"""scripts/launch_run.sh and run_manifest.py against a throwaway git repo and a fake `modal` (no network, no spend):
    python -m pytest test_launch_run.py -q"""
import hashlib
import os
import re
import shutil
import stat
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "scripts", "launch_run.sh")

MODAL_APP = '''import modal
WAVE1 = ("de", "es")
MODERATE_LANGS = ("uk", "sv", "cs")
RUN2_LANGS = WAVE1 + ("ru",) + MODERATE_LANGS[1:]
nemo_image = (modal.Image.x().add_local_python_source("common", "ft_train"))
tts_image = (modal.Image.x().add_local_python_source("readings"))
R3_ARMS = {"S-A": dict(lr=1e-4)}
R3_ARMS.update({"R10-main": dict(lr=1e-4), "R10-brC": dict(lr=1e-4), "R10-brE": dict(lr=1e-4), "R10-encF": dict(lr=1e-4)})
'''
FAKE_MODAL = '''#!/bin/bash
d=$(dirname "$0")
echo "$PWD :: $*" >> "$d/calls.txt"
if [ "$1 $2" = "volume ls" ]; then grep -qxF "$4" "$d/exists.txt"; exit $?; fi
if [ "$1 $2" = "run --detach" ]; then echo "View run at https://modal.com/apps/ws/main/ap-FAKE123abc"; exit 0; fi
exit 0
'''


def sh(*a, cwd=None, env=None):
    return subprocess.run(a, cwd=cwd, env=env, capture_output=True, text=True)


@pytest.fixture
def world(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pipeline").mkdir(parents=True)
    (repo / "pipeline" / "modal_app.py").write_text(MODAL_APP)
    for m in ("common", "ft_train"):
        (repo / "pipeline" / f"{m}.py").write_text(f"# {m} v1\n")
    for m in ("run_manifest.py", "run_watch.py"):
        shutil.copy(os.path.join(HERE, m), repo / "pipeline" / m)
    sh("git", "init", "-q", cwd=repo)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A", cwd=repo)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base", cwd=repo)
    sha = sh("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    (repo / "pipeline" / "common.py").write_text("# common DIRTY\n")  # an uncommitted edit must never be launched
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "modal").write_text(FAKE_MODAL)
    (fake / "modal").chmod(0o755)
    (fake / "exists.txt").write_text("")
    appr = tmp_path / "approvals.txt"
    appr.write_text("other_run\nR10_run1\n")
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}")

    def launch(*extra, run="R10_run1", steps="5000", log=None, arm="R10-main", sha_=None):
        log = log or tmp_path / "log"
        return sh("bash", SCRIPT, "--run", run, "--arm", arm, "--sha", sha_ or sha, "--max-steps", steps,
                  "--mix-dir", "mix4final10", "--repo", str(repo), "--approvals", str(appr), "--log-dir", str(log),
                  "--no-watch", *extra, env=env), log

    return dict(repo=repo, sha=sha, fake=fake, appr=appr, launch=launch, tmp=tmp_path)


def calls(w):
    p = w["fake"] / "calls.txt"
    return p.read_text() if p.exists() else ""


def test_unapproved_run_name_is_refused_before_anything_else(world):
    r, _ = world["launch"](run="R10_run2")  # not a line of the approvals file
    assert r.returncode == 3 and "no approval" in r.stderr
    assert "R10_run" not in calls(world) and "run --detach" not in calls(world)
    r, _ = world["launch"](run="R10_run")  # a prefix of an approved name is not an approval
    assert r.returncode == 3


def test_existing_run_directory_is_refused(world):
    (world["fake"] / "exists.txt").write_text("exp/R10_run1\n")
    r, _ = world["launch"]()
    assert r.returncode == 4 and "already exists" in r.stderr
    assert "run --detach" not in calls(world)


def test_other_refusals_leave_nothing_behind(world):
    r, log = world["launch"](steps="5100")
    assert r.returncode == 2 and "multiple of 250" in r.stderr
    r, log = world["launch"](arm="R10-nope")
    assert r.returncode == 2 and "not defined" in r.stderr
    assert not [p for p in os.listdir(log) if p.startswith("export.")]  # the export dir is removed on a refusal
    r, _ = world["launch"](sha_="deadbeef")
    assert r.returncode == 2 and "not found" in r.stderr
    assert "run --detach" not in calls(world)


def test_launch_uses_the_clean_export_and_the_standard_flags(world):
    r, log = world["launch"]("--part-weights", "mix4/tts_de.jsonl=3")
    assert r.returncode == 0, r.stderr
    line = [l for l in calls(world).splitlines() if "run --detach" in l][0]
    cwd, cmd = line.split(" :: ")
    assert re.search(r"/export\.[^/]+/pipeline$", cwd.replace("/private", ""))  # run from the export, not the checkout
    for flag in ("modal_app.py::retrain3_80_main", "--run R10_run1", "--arm R10-main", "--mix v4",
                 "--mix-dir mix4final10", "--langs de,es,ru,sv,cs", "--init-from ultra", "--anchor-init",
                 "--dur-weight 0", "--warmup 20", "--max-steps 5000", "--save-every 250", "--r3-sub-batch 1",
                 "--part-weights mix4/tts_de.jsonl=3"):
        assert flag in cmd, flag
    man = (log / "launch_manifest.txt").read_text()
    assert f"commit {world['sha']}" in man and "approval R10_run1" in man and "app ap-FAKE123abc" in man
    committed = hashlib.sha256(b"# common v1\n").hexdigest()  # the committed file, not the dirty working tree
    assert f"{committed}  common.py" in man
    assert hashlib.sha256(b"# common DIRTY\n").hexdigest() not in man
    assert re.search(r"[0-9a-f]{64}  modal_app\.py", man) and re.search(r"[0-9a-f]{64}  ft_train\.py", man)
    assert "readings.py" not in man  # only the training image's modules, not the TTS image's
    assert "FIRST 5 MINUTES" in r.stdout and "step250.pt.done" in r.stdout


def test_branch_arm_takes_the_main_runs_checkpoint(world):
    (world["fake"] / "exists.txt").write_text("exp/R10_main/step1000.pt\n")
    r, log = world["launch"]("--init-from", "R10_main@1000", arm="R10-brE")
    assert r.returncode == 0, r.stderr
    cmd = [l for l in calls(world).splitlines() if "run --detach" in l][0].split(" :: ")[1]
    assert "--init-from R10_main@1000" in cmd and "--anchor-init" in cmd and "--arm R10-brE" in cmd
    assert "--max-steps 5000" in cmd  # the caller's number; a branch passes --max-steps 500
    assert "INIT R10_main@1000" in r.stdout and "branch options" in r.stdout


def test_the_full_continuation_arm_is_a_branch_arm_too(world):
    (world["fake"] / "exists.txt").write_text("exp/R10_main/step1000.pt\n")
    r, _ = world["launch"]("--init-from", "R10_main@1000", arm="R10-encF", steps="4000")
    assert r.returncode == 0, r.stderr
    cmd = [l for l in calls(world).splitlines() if "run --detach" in l][0].split(" :: ")[1]
    assert "--arm R10-encF" in cmd and "--init-from R10_main@1000" in cmd and "--max-steps 4000" in cmd
    r, _ = world["launch"](arm="R10-encF", steps="4000")
    assert r.returncode == 2 and "needs --init-from" in r.stderr


def test_branch_start_is_checked_before_anything_launches(world):
    r, _ = world["launch"]("--init-from", "R10_main@1000", arm="R10-brC")  # the checkpoint is not on the volume
    assert r.returncode == 7 and "does not exist" in r.stderr
    r, _ = world["launch"](arm="R10-brC")  # a branch arm without a start checkpoint
    assert r.returncode == 2 and "needs --init-from" in r.stderr
    r, _ = world["launch"]("--init-from", "R10_run1@1000", arm="R10-brC")  # its own run
    assert r.returncode == 2
    r, _ = world["launch"]("--init-from", "R10_main@1000")  # a stock-start arm with a branch checkpoint
    assert r.returncode == 2 and "starts from stock" in r.stderr
    r, _ = world["launch"]("--init-from", "R10_main", arm="R10-brC")  # no step
    assert r.returncode == 2 and "<run>@<step>" in r.stderr
    assert "run --detach" not in calls(world)


def test_default_launch_keeps_the_stock_start_flags(world):
    r, _ = world["launch"]()
    cmd = [l for l in calls(world).splitlines() if "run --detach" in l][0].split(" :: ")[1]
    assert "--init-from ultra --anchor-init" in cmd and "INIT ultra" in r.stdout


def test_resume_needs_the_run_and_the_first_commit(world):
    r, log = world["launch"]("--resume")
    assert r.returncode == 4 and "does not exist" in r.stderr
    (world["fake"] / "exists.txt").write_text("exp/R10_run1\n")
    r, _ = world["launch"]("--resume", log=log)  # no first launch logged here
    assert r.returncode == 5
    (world["fake"] / "exists.txt").write_text("")
    first, log = world["launch"]()
    assert first.returncode == 0
    (world["fake"] / "exists.txt").write_text("exp/R10_run1\n")
    again, _ = world["launch"]("--resume", log=log)
    assert again.returncode == 0 and "resumed launch" in (log / "launch_manifest.txt").read_text()


def test_dry_run_checks_everything_and_launches_nothing(world):
    r, log = world["launch"]("--dry-run")
    assert r.returncode == 0 and "DRY RUN" in r.stdout
    assert "run --detach" not in calls(world) and not (log / "launch_manifest.txt").exists()


def test_script_has_no_local_paths_and_no_env_var_reads():
    text = open(SCRIPT).read()
    for bad in ("/Users/", "/home/", "~/", "$HOME", "${HOME", ".config", "/private", "/tmp/"):
        assert bad not in text, bad


def test_run_manifest_reads_the_export_without_importing_it(tmp_path):
    (tmp_path / "modal_app.py").write_text(MODAL_APP)
    (tmp_path / "common.py").write_text("x")
    (tmp_path / "ft_train.py").write_text("y")
    import run_manifest as RM

    assert RM.mounted_modules(str(tmp_path)) == ["common", "ft_train"]
    assert RM.run2_langs(str(tmp_path)) == "de,es,ru,sv,cs"
    assert RM.has_arm(str(tmp_path), "R10-main") and not RM.has_arm(str(tmp_path), "R10")
    assert [n for n, _ in RM.module_hashes(str(tmp_path))] == ["modal_app", "common", "ft_train"]
    os.remove(tmp_path / "ft_train.py")
    with pytest.raises(OSError):
        RM.module_hashes(str(tmp_path))


def test_the_real_modal_app_is_readable_by_the_launcher():
    import run_manifest as RM

    assert "ft_train" in RM.mounted_modules(HERE) and "mix_build" in RM.mounted_modules(HERE)
    assert RM.run2_langs(HERE).startswith("de,es,fr,it,nl,pt,ru")
    assert RM.has_arm(HERE, "R10-main")
    assert all(RM.has_arm(HERE, a) for a in ("R10-brC", "R10-brE", "R10-brEr", "R10-encF"))
