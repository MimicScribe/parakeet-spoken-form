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
        self.apps = []  # what `modal app list --json` returns (None = the call fails)
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
        if argv[:4] == ["modal", "app", "stop", "--yes"]:
            self.stops.append(argv[4])
            return 0, "stopped"
        if argv[:3] == ["modal", "app", "list"]:
            return (0, json.dumps(self.apps)) if self.apps is not None else (1, "network error")
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
    assert w["ex"].stops == ["ap-MAINAPP"] and t.state["early_stop"]["main"]["action"] == "stopped"
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


def app(app_id, tasks="1", state="ephemeral", desc="parakeet-x"):
    return dict(app_id=app_id, description=desc, state=state, tasks=tasks, created_at="2026-10-18 22:00:00-07:00")


def test_running_apps_keeps_only_apps_with_a_task_and_drops_the_nights_own():
    raw = json.dumps([app("ap-1"), app("ap-2", tasks="0"), app("ap-3", state="stopped"), app("ap-MAINAPP"),
                      app("ap-4", state="ephemeral (detached)", tasks="12")])
    got = T.running_apps(raw, {"ap-MAINAPP"})
    assert [a["app_id"] for a in got] == ["ap-1", "ap-4"]
    assert T.running_apps("not json") is None and T.running_apps("{}") is None
    assert "ap-4" in T.format_apps(got) and "tasks=12" in T.format_apps(got)


def test_a_busy_volume_delays_a_branch_launch_then_goes_on_after_the_wait(world):
    w = world
    w["vol"].put("M", 1000)
    cfg = dict(w["cfg"], quiet_max_other=0, quiet_wait_s=1800)
    w["ex"].apps = [app("ap-MAINAPP"), app("ap-DENSE", tasks="12", desc="parakeet-dense-scan")]  # main is the night's own
    t = w["make"](cfg)
    t.cycle()
    assert w["ex"].launches == [] and "waiting for a quiet volume" in log_text(w) and "ap-DENSE" in log_text(w)
    w["clock"].t += 600
    t.cycle()
    assert w["ex"].launches == [] and log_text(w).count("waiting for a quiet volume") == 1  # logged once
    w["ex"].apps = [app("ap-MAINAPP")]  # the readers stopped
    t.cycle()
    assert launched_runs(w) == ["C"]
    # control: a volume that never quiets is waited out, then the launch goes ahead with an alarm
    w2 = dict(w)
    w["vol"].files.clear()
    w["vol"].put("M", 1000)
    os.remove(cfg["state_file"])
    w["ex"].launches.clear()
    w["ex"].apps = [app("ap-MAINAPP"), app("ap-DENSE")]
    t = w["make"](cfg)
    t.cycle()
    assert w["ex"].launches == []
    w["clock"].t += 1900
    t.cycle()
    assert launched_runs(w) == ["C"] and "launching anyway" in log_text(w)


def test_a_failing_app_list_is_not_a_quiet_volume(world):
    w = world
    w["vol"].put("M", 1000)
    w["ex"].apps = None
    t = w["make"](dict(w["cfg"], quiet_max_other=0))
    t.cycle()
    assert w["ex"].launches == [] and "the app list failed" in log_text(w)


def test_without_quiet_max_other_the_launch_does_not_look_at_other_apps(world):
    w = world
    w["vol"].put("M", 1000)
    w["ex"].apps = [app("ap-DENSE")] * 5
    w["make"]().cycle()
    assert launched_runs(w) == ["C"]


def test_second_seed_control_branch_launches_5_min_after_brE_with_its_seed_and_does_not_enter_the_gate(world):
    w = world
    cfg = json.loads(json.dumps(w["cfg"]))
    cfg["runs"]["brCs2"] = dict(run="C2", arm="R10-brC", seed=777)
    w["vol"].put("M", 1000)
    t = w["make"](cfg)
    t.cycle()
    assert launched_runs(w) == ["C"]
    w["clock"].t += 301
    t.cycle()
    assert launched_runs(w) == ["C", "E"]
    w["clock"].t += 200
    t.cycle()
    assert launched_runs(w) == ["C", "E"]  # only 200 s after brE
    w["clock"].t += 101
    t.cycle()
    assert launched_runs(w) == ["C", "E", "C2"]
    c2 = w["ex"].launches[-1]
    assert c2[c2.index("--arm") + 1] == "R10-brC" and c2[c2.index("--init-from") + 1] == "M@1000"
    assert c2[c2.index("--max-steps") + 1] == "500" and c2[c2.index("--seed") + 1] == "777"
    assert all("--seed" not in a for a in w["ex"].launches[:2])  # brC and brE keep the arm's own seed
    # brEr (flag) goes after brCs2, not at the same time
    open(w["cfg"]["brEr_flag"], "w").close()
    t.cycle()
    assert launched_runs(w) == ["C", "E", "C2"]
    w["clock"].t += 301
    t.cycle()
    assert launched_runs(w) == ["C", "E", "C2", "Er"]
    # scored like brC, never read by the gate
    w["vol"].put("C", 250, 500)
    w["vol"].put("E", 250, 500)
    w["vol"].put("C2", 250, 500)
    gate_says(w, 0, 0, 0)
    settle(w, t, 10)
    assert {(a[1], a[2]) for a in w["ex"].scores} >= {("C2", "500")} and ("C2", "250") not in {(a[1], a[2]) for a in w["ex"].scores}
    assert all("C2" not in g for g in w["ex"].gates) and all(g[g.index("--steps") - 1] == "C" for g in w["ex"].gates)


def test_anchor_test_is_scored_with_the_main_runs_step_500_and_compared(world, tmp_path):
    w = world
    (tmp_path / "cmp.py").write_text("")
    cfg = json.loads(json.dumps(w["cfg"]))
    cfg["runs"]["anc1"] = dict(run="A1", arm="R10-anc1")
    cfg["compare"] = [dict(name="anc1_vs_main500", a="anc1@500", b="main@500", script=str(tmp_path / "cmp.py"),
                           cmd=["CMP", "{a_dir}", "{b_dir}"])]
    w["vol"].put("M", 250, 500)
    w["vol"].put("A1", 250, 500)
    seen = []
    orig = w["ex"].run
    w["ex"].run = lambda argv, *a, **k: (0, "metric A B A-B\nen_value +1 +2 -1\n") if argv[0] == "CMP" and not seen.append(argv) else orig(argv, *a, **k)
    t = w["make"](cfg)
    settle(w, t, 5)
    assert {(a[1], a[2]) for a in w["ex"].scores} >= {("A1", "500"), ("M", "500")}
    assert len(seen) == 1 and seen[0][1].endswith("A1@500__") is False and "A1_step500" in seen[0][1] and "M_step500" in seen[0][2]
    assert open(os.path.join(cfg["score_root"], "anc1_vs_main500.txt")).read().startswith("metric")
    settle(w, t, 3)
    assert len(seen) == 1  # once
    assert "compare anc1_vs_main500" in log_text(w)


def test_the_trimmed_plan_scores_only_the_decision_points(world):
    w = world
    cfg = json.loads(json.dumps(w["cfg"]))
    cfg["runs"]["cap"] = dict(run="CAP", arm="R10-main")
    cfg["runs"]["enfree2"] = dict(run="EF2", arm="R8-s-enfree")
    cfg["runs"].pop("lr05")
    cfg["runs"]["brCs2"] = dict(run="C2", arm="R10-brC", seed=777)
    cfg["score_plan"] = dict(main=[1000, 2500, 5000], cap=[2500, 5000], encF=[4000], enfree2=[500])
    cfg["early_stop"] = dict(step=2500, v_limit_pp=3.0, roles=["main", "cap"], summary_cmd=["SUMMARY", "{scores_dir}"],
                             ref={"cap": "main@1000"})
    for run, steps in (("M", range(250, 5001, 250)), ("CAP", range(250, 5001, 250)), ("EF2", (250, 500)),
                       ("C", (250, 500)), ("E", (250, 500)), ("C2", (250, 500)), ("F", range(250, 4001, 250))):
        w["vol"].put(run, *steps)
    w["ex"].summaries = {n: {"pooled_v_vs_stock_pp": 1.0, "wer_ok": True} for n in ("M_step1000", "M_step2500", "CAP_step2500")}
    gate_says(w, 0, 0, 0)
    t = w["make"](cfg)
    settle(w, t, 30)
    got = sorted((a[1], int(a[2])) for a in w["ex"].scores)
    assert got == sorted([("M", 1000), ("M", 2500), ("M", 5000), ("CAP", 2500), ("CAP", 5000), ("F", 4000), ("EF2", 500),
                          ("C", 250), ("C", 500), ("E", 250), ("E", 500), ("C2", 500)])  # nothing else is scored
    assert t.state["early_stop"]["main"]["action"] == "kept" and t.state["early_stop"]["cap"]["action"] == "kept"


def test_early_stop_applies_to_each_main_and_stops_only_that_runs_app(world):
    w = world
    cfg = json.loads(json.dumps(w["cfg"]))
    cfg["runs"]["cap"] = dict(run="CAP", arm="R10-main")
    cfg["early_stop"] = dict(step=2500, v_limit_pp=3.0, roles=["main", "cap"], summary_cmd=["SUMMARY", "{scores_dir}"],
                             ref={"cap": "main@1000"})
    capdir = w["tmp"] / "runs" / "CAP"
    capdir.mkdir(parents=True)
    (capdir / "launch_manifest.txt").write_text("run CAP\ncommit abc\napp ap-CAPAPP\n")
    w["vol"].put("M", 1000, 2500)
    w["vol"].put("CAP", 2500)
    w["ex"].summaries = {"M_step1000": {"pooled_v_vs_stock_pp": 1.2, "wer_ok": True},
                         "M_step2500": {"pooled_v_vs_stock_pp": 1.0, "wer_ok": True},    # main is fine
                         "CAP_step2500": {"pooled_v_vs_stock_pp": 3.6, "wer_ok": True}}  # cap: > +3 and worse than main@1000
    t = w["make"](cfg)
    settle(w, t, 4)
    assert w["ex"].stops == ["ap-CAPAPP"]
    assert t.state["early_stop"]["main"]["action"] == "kept" and t.state["early_stop"]["cap"]["action"] == "stopped"


def test_a_role_mix_overrides_the_config_mix_in_the_templates(world):
    w = world
    cfg = json.loads(json.dumps(w["cfg"]))
    cfg.update(mix="mixA", score_dir_template="{score_root}/{run}@{step}__{mix}", score_argv=["S", "{run}", "{step}", "--mix", "{mix}"])
    cfg["runs"]["cap"] = dict(run="CAP", arm="R10-main", mix="mixCAP")
    cfg["score_plan"] = dict(main=[1000], cap=[1000])
    w["vol"].put("M", 1000)
    w["vol"].put("CAP", 1000)
    t = w["make"](cfg)
    settle(w, t, 3)
    by = {a[1]: a for a in w["ex"].scores}
    assert by["M"][-1] == "mixA" and by["CAP"][-1] == "mixCAP"
    assert t.out_dir("cap", 1000).endswith("CAP@1000__mixCAP")


def test_runs_launched_by_hand_that_never_appear_do_not_keep_the_orchestrator_alive(world):
    w = world
    cfg = json.loads(json.dumps(w["cfg"]))
    cfg["runs"]["cap"] = dict(run="CAP", arm="R10-main")
    cfg["runs"]["anc1"] = dict(run="A1", arm="R10-anc1")
    cfg["score_plan"] = dict(main=[1000], cap=[2500])
    cfg["early_stop"] = {}
    cfg["compare"] = [dict(name="c", a="cap@2500", b="main@1000", cmd=["CMP"]), dict(name="d", a="anc1@500", b="main@500", cmd=["CMP"])]
    start_branches(w, steps=(250, 500))
    w["vol"].put("M", 500, 1000)
    gate_says(w, 0, 0, 0)
    t = w["make"](cfg)
    settle(w, t, 12)
    assert t.finished()  # nothing waits for CAP or A1: they were never launched


def test_without_the_anchor_test_nothing_waits_for_it(world):
    w = world
    t = w["make"]()
    assert "anc1" not in {r for _, r, _ in t.plan()} and ("main", 500) not in {(r, s) for _, r, s in t.plan()}


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


def test_trigger_sh_quiet_lists_the_running_apps_and_exits_nonzero_when_busy(tmp_path):
    import subprocess

    fake = tmp_path / "bin"
    fake.mkdir()
    doc = json.dumps([app("ap-DENSE", tasks="12", desc="parakeet-dense-scan"), app("ap-OLD", state="stopped", tasks="0")])
    (fake / "modal").write_text(f"#!/bin/bash\necho '{doc}'\n")
    (fake / "modal").chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}")
    script = os.path.join(HERE, "..", "scripts", "trigger.sh")
    r = subprocess.run(["bash", script, "quiet"], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "ap-DENSE" in r.stdout and "ap-OLD" not in r.stdout and "NOT QUIET" in r.stdout
    (fake / "modal").write_text("#!/bin/bash\necho '[]'\n")
    r = subprocess.run(["bash", script, "quiet"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "QUIET" in r.stdout and "none" in r.stdout
    (fake / "modal").write_text("#!/bin/bash\necho boom >&2; exit 1\n")
    r = subprocess.run(["bash", script, "quiet"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "UNKNOWN" in r.stdout
