"""scripts/trigger.py against a fake volume, launcher, scorer and gate (no network, no spend, nothing launched):
    python -m pytest test_trigger.py -q"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import trigger as T  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


class FakeVol:
    def __init__(self):
        self.files = {}  # run -> set of file names
        self.fail = False

    def ls(self, path):
        if self.fail:
            return "error", []
        if path == "exp":
            return "ok", [f"exp/{r}" for r in sorted(self.files)]
        run = path.split("/", 1)[1]
        if run not in self.files:
            return "absent", []
        return "ok", [f"exp/{run}/{f}" for f in sorted(self.files[run])]

    def put(self, run, *steps, final=False):
        fs = self.files.setdefault(run, set())
        for s in steps:
            fs |= {f"step{s}.pt", f"step{s}.pt.done"}
        if final:
            fs.add("decoder_joint.pt")


class FakeHandle:
    def __init__(self, rc):
        self.rc, self.killed = rc, False

    def poll(self):
        return self.rc

    def kill(self):
        self.killed = True


class FakeExec:
    def __init__(self, tmp):
        self.tmp = tmp
        self.launches, self.gates, self.scores, self.stops = [], [], [], []
        self.launch_rc = []  # exit codes to return in order (default 0)
        self.gate_out = {}  # steps text -> output
        self.gate_rc = {}  # steps text -> exit code (0 GO, 1 NO-GO, 2 no verdict)
        self.score_rc = 0
        self.summaries = {}  # out dir basename -> summary dict
        self.on_launch = None

    def run(self, argv, timeout, log_path=None, cwd=None):
        if argv[:2] == ["bash", "LAUNCHER"]:
            self.launches.append(argv)
            rc = self.launch_rc.pop(0) if self.launch_rc else 0
            if rc == 0 and self.on_launch:
                self.on_launch(argv)
            return rc, f"launcher said {rc}\napp ap-X\n"
        if argv[:3] == ["modal", "app", "stop"]:
            self.stops.append(argv[3])
            return 0, "stopped"
        if argv[0] == "SUMMARY":
            d = os.path.basename(argv[1])
            return (0, json.dumps(self.summaries[d])) if d in self.summaries else (1, "no scores")
        self.gates.append(argv)
        steps = argv[argv.index("--steps") + 1]
        return self.gate_rc.get(steps, 2), self.gate_out.get(steps, "CANNOT DECIDE")

    def start(self, argv, log_path, cwd=None):
        out = argv[-1]
        self.scores.append(argv)
        s = self.summaries.get(os.path.basename(out))
        if s is not None:
            with open(os.path.join(out, "summary.json"), "w") as f:
                json.dump(s, f)
        return FakeHandle(self.score_rc)


@pytest.fixture
def world(tmp_path):
    (tmp_path / "scorer.sh").write_text("")
    (tmp_path / "gate.py").write_text("")
    (tmp_path / "main_log").mkdir()
    (tmp_path / "main_log" / "launch_manifest.txt").write_text(
        "run m\ncommit abc123def\narm R10-main\nlaunched x\n"
        "command modal run --detach modal_app.py::retrain3_80_main --run m --mix v4 --mix-dir mixdir --langs a,b "
        "--part-weights p1=2,p2=1 --extra-parts x:y:3\napp ap-MAINAPP\n")
    cfg = dict(
        volume="vol", launcher="LAUNCHER", repo="repo", approvals="appr", runs_log_root=str(tmp_path / "runs"),
        score_script=str(tmp_path / "scorer.sh"), gate_script=str(tmp_path / "gate.py"), score_root=str(tmp_path / "sc"),
        state_file=str(tmp_path / "state.json"), log_file=str(tmp_path / "trigger.log"),
        main_log_dir=str(tmp_path / "main_log"), brEr_flag=str(tmp_path / "brEr.flag"), stagger_s=300,
        runs=dict(main=dict(run="M", arm="R10-main"), lr05=dict(run="L", arm="R10-main-lr05"),
                  brC=dict(run="C", arm="R10-brC"), brE=dict(run="E", arm="R10-brE"), brEr=dict(run="Er", arm="R10-brEr"),
                  encF=dict(run="F", arm="R10-encF")),
        score_plan=dict(main=[1000, 1500, 2000, 2500, 3000, 5000], lr05=[2000, 3500, 5000], encF=[4000]),
        early_stop=dict(step=2500, v_limit_pp=3.0, summary_cmd=["SUMMARY", "{scores_dir}"]), score_concurrency=2)
    os.makedirs(cfg["score_root"])
    vol, clock, ex = FakeVol(), Clock(), FakeExec(tmp_path)

    def make(c=None, **kw):
        return T.Trigger(c or cfg, vol, ex, clock=clock, say=lambda m: None, **kw)

    return dict(cfg=cfg, vol=vol, clock=clock, ex=ex, make=make, tmp=tmp_path)


def launched_runs(w):
    return [a[a.index("--run") + 1] for a in w["ex"].launches]


def log_text(w):
    return (w["tmp"] / "trigger.log").read_text()


def test_nothing_is_launched_before_the_main_run_reaches_the_branch_step_and_never_a_main(world):
    w = world
    t = w["make"]()
    t.cycle()  # exp/M does not exist
    w["vol"].put("M", 250, 500, 750)
    t.cycle()
    assert w["ex"].launches == []
    w["vol"].put("M", 1000)
    t.cycle()
    assert launched_runs(w) == ["C"]
    for _ in range(3):
        w["clock"].t += 400
        t.cycle()
    assert set(launched_runs(w)) <= {"C", "E"}  # M, L (the mains) and F are never launched by this script


def test_branches_launch_five_minutes_apart_from_the_main_runs_commit_and_mix(world):
    w = world
    w["vol"].put("M", 250, 500, 750, 1000)
    t = w["make"]()
    t.cycle()
    assert launched_runs(w) == ["C"]
    w["clock"].t += 200
    t.cycle()
    assert launched_runs(w) == ["C"]  # only 200 s
    w["clock"].t += 150
    t.cycle()
    assert launched_runs(w) == ["C", "E"]
    c, e = w["ex"].launches
    for a, arm in ((c, "R10-brC"), (e, "R10-brE")):
        assert a[a.index("--arm") + 1] == arm and a[a.index("--init-from") + 1] == "M@1000"
        assert a[a.index("--max-steps") + 1] == "500" and a[a.index("--sha") + 1] == "abc123def"
        assert a[a.index("--mix-dir") + 1] == "mixdir" and a[a.index("--part-weights") + 1] == "p1=2,p2=1"
        assert a[a.index("--extra-parts") + 1] == "x:y:3"
        assert a[a.index("--approvals") + 1] == "appr" and a[a.index("--log-dir") + 1].endswith("/runs/" + a[a.index("--run") + 1])
    assert "ACTION launch brC" in log_text(w) and "ACTION launch brE" in log_text(w)


def test_restart_never_relaunches_and_an_existing_run_is_never_launched(world):
    w = world
    w["vol"].put("M", 1000)
    t = w["make"]()
    t.cycle()
    w["clock"].t += 400
    t.cycle()
    assert launched_runs(w) == ["C", "E"]
    t2 = w["make"]()  # a restart: state from disk, the volume now holds the two branch dirs
    w["vol"].put("C", 250)
    w["vol"].put("E", 250)
    for _ in range(3):
        w["clock"].t += 400
        t2.cycle()
    assert launched_runs(w) == ["C", "E"]
    # control: no state file at all, but exp/C and exp/E exist -> still no launch; only a missing run would launch
    os.remove(w["cfg"]["state_file"])
    t3 = w["make"]()
    t3.cycle()
    assert launched_runs(w) == ["C", "E"] and "already exists" in log_text(w)


def test_a_failed_volume_listing_is_never_read_as_absent(world):
    w = world
    w["vol"].put("M", 1000)
    w["vol"].fail = True
    t = w["make"]()
    for _ in range(3):
        t.cycle()
    assert w["ex"].launches == [] and "cannot list exp/" in log_text(w)
    w["vol"].fail = False  # control
    t.cycle()
    assert launched_runs(w) == ["C"]


def test_brEr_launches_only_when_the_flag_file_exists(world):
    w = world
    w["vol"].put("M", 1000)
    t = w["make"]()
    for _ in range(4):
        t.cycle()
        w["clock"].t += 400
    assert launched_runs(w) == ["C", "E"]
    open(w["cfg"]["brEr_flag"], "w").close()
    t.cycle()
    assert launched_runs(w) == ["C", "E", "Er"]  # more than 5 min after E
    settle(w, t, 3)
    assert launched_runs(w) == ["C", "E", "Er"]
    assert "R10-brEr" in w["ex"].launches[-1]


def test_a_launch_refused_for_a_missing_approval_is_retried_and_logged_once(world):
    w = world
    w["vol"].put("M", 1000)
    w["ex"].launch_rc = [3, 3, 0]
    t = w["make"]()
    t.cycle()
    t.cycle()  # within RETRY_S: no second call
    assert len(w["ex"].launches) == 1
    w["clock"].t += 301
    t.cycle()
    assert len(w["ex"].launches) == 2
    w["clock"].t += 301
    t.cycle()
    assert len(w["ex"].launches) == 3 and launched_runs(w)[2] == "C"
    assert log_text(w).count("launcher refused") == 1
    w["clock"].t += 400
    t.cycle()
    assert launched_runs(w).count("C") == 3 and "E" in launched_runs(w)  # C is settled now; E follows


def test_an_uncertain_launch_is_not_retried(world):
    w = world
    w["vol"].put("M", 1000)
    w["ex"].launch_rc = [6]
    t = w["make"]()
    t.cycle()
    for _ in range(3):
        w["clock"].t += 400
        t.cycle()
    assert launched_runs(w).count("C") == 1 and "ALARM brC" in log_text(w)


def test_markers_queue_the_planned_scores_and_a_missing_scorer_waits(world):
    w = world
    os.remove(w["cfg"]["score_script"])
    w["vol"].put("M", 250, 500, 750, 1000, 1250, 1500)
    w["vol"].put("L", 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
    t = w["make"]()
    t.cycle()
    assert w["ex"].scores == [] and log_text(w).count("scorer") >= 1
    assert log_text(w).count("does not exist yet") == 1
    open(w["cfg"]["score_script"], "w").close()
    t.cycle()
    runs_steps = {(a[1], a[2]) for a in w["ex"].scores}
    assert len(w["ex"].scores) <= 2  # concurrency
    for _ in range(6):
        t.cycle()  # finished fake jobs free their slots
    runs_steps = {(a[1], a[2]) for a in w["ex"].scores}
    assert runs_steps == {("M", "1000"), ("M", "1500"), ("L", "2000")}  # only the planned checkpoints, once each
    assert len(w["ex"].scores) == 3
    assert all(a[0] == w["cfg"]["score_script"] for a in w["ex"].scores)


def test_all_planned_checkpoints_of_the_three_runs_are_scored(world):
    w = world
    w["vol"].put("M", *range(250, 5001, 250))
    w["vol"].put("L", *range(250, 5001, 250))
    w["vol"].put("F", *range(250, 4001, 250))
    w["ex"].summaries = {}
    t = w["make"](dry_run=False)
    w["cfg"]["early_stop"] = {}
    for _ in range(20):
        t.cycle()
    got = {(a[1], int(a[2])) for a in w["ex"].scores}
    want = {("M", s) for s in (1000, 1500, 2000, 2500, 3000, 5000)} | {("L", s) for s in (2000, 3500, 5000)} | {("F", 4000)}
    assert got == want


def start_branches(w, steps=(250, 500)):
    w["vol"].put("M", 1000)
    w["vol"].put("C", *steps)
    w["vol"].put("E", *steps)


def settle(w, t, n=8):
    for _ in range(n):
        w["clock"].t += 400
        t.cycle()


def gate_says(w, k250, k500, both):
    w["ex"].gate_rc = {"250": k250, "500": k500, "250,500": both}
    w["ex"].gate_out = {k: ("==> GO for encF" if v == 0 else "==> NO-GO for encF" if v == 1 else "CANNOT DECIDE")
                        for k, v in w["ex"].gate_rc.items()}


def test_gate_go_at_250_500_and_over_both_launches_encF_once(world):
    w = world
    start_branches(w)
    gate_says(w, 0, 0, 0)
    t = w["make"]()
    settle(w, t)
    assert [g[g.index("--steps") + 1] for g in w["ex"].gates] == ["250", "500", "250,500"]
    g = w["ex"].gates[0]
    assert g[g.index("--steps") - 2:g.index("--steps")] == ["E", "C"]  # <E run> <C run> --steps ...
    assert t.state["encF"] == "go" and "F" in launched_runs(w)
    f = w["ex"].launches[launched_runs(w).index("F")]
    assert f[f.index("--arm") + 1] == "R10-encF" and f[f.index("--max-steps") + 1] == "4000"
    assert f[f.index("--init-from") + 1] == "M@1000"
    settle(w, t)
    assert launched_runs(w).count("F") == 1 and len(w["ex"].gates) == 3


@pytest.mark.parametrize("k250,k500,both", [(1, 0, 0), (0, 1, 1), (0, 0, 1)])  # a NO-GO anywhere, incl. the sign read
def test_gate_no_go_anywhere_never_launches_encF(world, k250, k500, both):
    w = world
    start_branches(w)
    gate_says(w, k250, k500, both)
    t = w["make"]()
    settle(w, t)
    assert t.state["encF"] == "nogo" and "F" not in launched_runs(w)
    assert "encF decision: NO" in log_text(w)


def test_the_gate_waits_for_both_branches_scores_at_each_step(world):
    w = world
    w["vol"].put("M", 1000)
    w["vol"].put("C", 250)
    w["vol"].put("E", 250)
    gate_says(w, 0, 0, 0)
    t = w["make"]()
    settle(w, t)
    assert [g[g.index("--steps") + 1] for g in w["ex"].gates] == ["250"] and t.state["encF"] == "pending"
    w["vol"].put("C", 500)
    settle(w, t)
    assert len(w["ex"].gates) == 1  # E has no step 500 yet
    w["vol"].put("E", 500)
    settle(w, t)
    assert t.state["encF"] == "go"


def test_a_missing_gate_waits_and_a_gate_without_a_verdict_is_retried_then_left_to_hand(world):
    w = world
    start_branches(w)
    os.remove(w["cfg"]["gate_script"])
    t = w["make"]()
    settle(w, t, 4)
    assert w["ex"].gates == [] and t.state["encF"] == "pending" and "interim gate" in log_text(w)
    open(w["cfg"]["gate_script"], "w").close()
    settle(w, t, 8)  # exit 2 = cannot decide: three tries (>= 5 min apart), then "decide by hand", never a launch
    assert "F" not in launched_runs(w) and "decide by hand" in log_text(w) and t.state["encF"] == "manual"
    assert len(w["ex"].gates) == 3


def test_early_stop_at_main_2500_stops_only_the_main_app(world):
    w = world
    w["vol"].put("M", 1000, 2500)
    w["ex"].summaries = {"M_step1000": {"pooled_v_vs_stock_pp": 1.2, "wer_ok": True},
                         "M_step2500": {"pooled_v_vs_stock_pp": 3.6, "wer_ok": True}}
    t = w["make"]()
    settle(w, t, 4)
    assert w["ex"].stops == ["ap-MAINAPP"] and t.state["early_stop"]["action"] == "stopped"
    w["vol"].put("M", 3000)
    settle(w, t, 3)
    assert w["ex"].stops == ["ap-MAINAPP"]  # once
    assert not any(a[1:3] == ["M", "3000"] for a in w["ex"].scores)  # later main checkpoints are not scored after a stop


@pytest.mark.parametrize("v2500,v1000,wer,stop", [
    (3.6, 1.2, True, True), (3.6, 4.0, True, False), (2.9, 0.5, True, False), (1.0, 0.5, False, True)])
def test_early_stop_rule_table(v2500, v1000, wer, stop):
    verdict, _ = T.early_stop_decision(v2500, v1000, wer, 3.0)
    assert (verdict == "stop") == stop


def test_early_stop_without_the_numbers_is_flagged_not_acted_on(world):
    w = world
    w["vol"].put("M", 1000, 2500)
    w["ex"].summaries = {"M_step1000": {}, "M_step2500": {}}
    t = w["make"]()
    settle(w, t, 4)
    assert w["ex"].stops == [] and "cannot be decided" in log_text(w)
    assert T.early_stop_decision(5.0, None, True, 3.0)[0] == "undetermined"
    assert T.early_stop_decision(4.0, 1.0, None, 3.0)[0] == "stop"  # V alone is enough
    assert T.early_stop_decision(1.0, 1.0, None, 3.0)[0] == "undetermined"  # an unknown WER is not "fine"
    assert T.early_stop_decision(None, 1.0, True, 3.0)[0] == "undetermined"


def test_dry_run_acts_on_nothing(world):
    w = world
    start_branches(w)
    t = w["make"](dry_run=True)
    settle(w, t, 3)
    assert w["ex"].launches == [] and w["ex"].scores == [] and w["ex"].stops == []
    assert "DRY-RUN" in log_text(w) and not os.path.exists(w["cfg"]["state_file"])


def test_a_failed_scorer_is_retried_once_then_given_up(world):
    w = world
    w["vol"].put("M", 1000)
    w["ex"].score_rc = 1
    t = w["make"]()
    settle(w, t, 6)
    assert [a[2] for a in w["ex"].scores].count("1000") == 2 and "giving up" in log_text(w)


def test_a_launch_command_that_cannot_be_read_in_full_never_feeds_a_branch(world, tmp_path):
    w = world
    man = tmp_path / "main_log" / "launch_manifest.txt"
    man.write_text("run m\ncommit abc123def\narm R10-main\nlaunched x\ncommand modal run --detach modal_app.py::f --run m "
                   "--mix v4 --mix-dir mixdir --save-every 250 --r3-sub-batch 1 --part-weights p1=2\np2=1 --extra-parts x:y:3\n"
                   "approval m\napp ap-MAINAPP\n")
    assert T.parse_manifest(str(man)).get("command_untrusted") is True
    w["vol"].put("M", 1000)
    t = w["make"]()
    t.cycle()
    assert w["ex"].launches == [] and "could not be read in full" in log_text(w)
    cfg = dict(w["cfg"], mix_dir="mixdir", part_weights="p1=2,p2=1", extra_parts="x:y:3")  # control: explicit values launch
    t = w["make"](cfg)
    t.cycle()
    a = w["ex"].launches[0]
    assert a[a.index("--part-weights") + 1] == "p1=2,p2=1" and a[a.index("--mix-dir") + 1] == "mixdir"


def test_helpers():
    assert T.parse_gate("x", 0) == "go" and T.parse_gate("x", 1) == "nogo" and T.parse_gate("CANNOT DECIDE", 2) is None
    assert T.parse_gate("a\n==> NO-GO for encF", None) == "nogo" and T.parse_gate("==> GO for encF", None) == "go"
    assert T.parse_gate("GOOD", None) is None and T.parse_gate("a traceback", 3) is None
    m = T.parse_manifest("/nonexistent")
    assert m == {}


def test_the_new_scripts_carry_no_local_paths_or_env_var_reads():
    for name in ("trigger.py", "trigger.sh", "launch_run.sh"):
        text = open(os.path.join(HERE, "..", "scripts", name)).read()
        for bad in ("/Users/", "/home/", "~/", "$HOME", "${HOME", "/private", "/tmp/", "os.environ", "getenv"):
            assert bad not in text, (name, bad)
    text = open(os.path.join(HERE, "run_watch.py")).read()
    for bad in ("/Users/", "~/", "os.environ", "getenv"):
        assert bad not in text, bad
