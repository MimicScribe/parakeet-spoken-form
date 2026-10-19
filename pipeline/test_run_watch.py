"""Run watchdog decisions and loop (no network, no modal CLI): python -m pytest test_run_watch.py -q"""
import json
import os

import run_watch as W

MIN = 60.0


def watch_(**kw):
    """A StallWatch with the default limits (stall 40 min, grace 75 min) seen once at t=0 with nothing on the volume."""
    w = W.StallWatch(0.0, kw.pop("stall", 40 * MIN), kw.pop("grace", 75 * MIN), **kw)
    w.observe(0.0, set(), tasks=1)
    return w


def test_grace_before_the_first_marker_is_75_minutes_then_stall_is_40():
    w = watch_()
    assert w.observe(74 * MIN, set(), tasks=1) == "ok"
    assert w.observe(76 * MIN, set(), tasks=1) == "stop" and w.reason == "no_first_marker"
    w = watch_()
    assert w.observe(60 * MIN, {250}, tasks=1) == "ok"  # the first marker ends the grace
    assert w.observe(99 * MIN, {250}, tasks=1) == "ok"  # 39 min since it
    assert w.observe(101 * MIN, {250}, tasks=1) == "stop" and w.reason == "stall"  # 41 min
    w = watch_()
    w.observe(60 * MIN, {250}, tasks=1)
    assert w.observe(99 * MIN, {250, 500}, tasks=1) == "ok"  # a new marker restarts the clock
    assert w.observe(138 * MIN, {250, 500}, tasks=1) == "ok"
    assert w.observe(140 * MIN, {250, 500}, tasks=1) == "stop"


def test_time_without_a_running_task_does_not_count_toward_the_grace():
    w = watch_()
    assert w.observe(30 * MIN, set(), tasks=0) == "ok"  # the app is in the GPU queue
    assert w.observe(300 * MIN, set(), tasks=0) == "ok"  # five hours queued: still fine for the grace clock
    assert w.observe(370 * MIN, set(), tasks=1) == "ok"  # the task starts; the 70 min until it is queue time
    assert w.observe(374 * MIN, set(), tasks=1) == "ok"
    assert w.observe(450 * MIN, set(), tasks=1) == "stop"  # 80 min of task time without a marker
    # control: the same run counted by wall clock would have stopped at the first 76-minute poll
    assert W.StallWatch(0.0, 40 * MIN, 75 * MIN).observe(76 * MIN, set()) == "stop"


def test_a_preemption_retry_resets_the_stall_clock_and_the_gap_is_not_counted():
    w = watch_()
    w.observe(10 * MIN, {250}, tasks=1)
    assert w.observe(45 * MIN, {250}, tasks=1) == "ok"  # 35 min of stall so far
    assert w.observe(50 * MIN, {250}, tasks=0) == "ok"  # preempted; the retry waits for a GPU
    assert w.observe(120 * MIN, {250}, tasks=0) == "ok"
    assert w.observe(122 * MIN, {250}, tasks=1) == "ok"  # the retry runs: the clock restarts
    assert w.attempt == 1
    assert w.observe(160 * MIN, {250}, tasks=1) == "ok"  # 38 min after the restart; without the reset, 150 min
    assert w.observe(164 * MIN, {250}, tasks=1) == "stop"
    # control: no gap in the task count (a hang, not a preemption) is a stall
    w = watch_()
    w.observe(10 * MIN, {250}, tasks=1)
    assert w.observe(51 * MIN, {250}, tasks=1) == "stop"


def test_max_run_time_counts_task_time_only_and_stops():
    w = watch_(max_run_s=4.5 * 3600)
    for m, step in ((60, 250), (120, 500), (180, 750), (240, 1000)):
        assert w.observe(m * MIN, {step}, tasks=1) == "ok"
    assert w.observe(265 * MIN, {1250}, tasks=1) == "ok"  # 4 h 25 min
    assert w.observe(271 * MIN, {1500}, tasks=1) == "stop" and w.reason == "max_run"
    w = watch_(max_run_s=4.5 * 3600)  # the same wall clock, but 3 h of it queued: no stop
    w.observe(10 * MIN, set(), tasks=0)
    w.observe(190 * MIN, {250}, tasks=1)
    assert w.observe(271 * MIN, {500}, tasks=1) == "ok"


def test_default_max_run_for_the_main_run_is_about_4_5_hours():
    assert abs(W.default_max_run_s(5000, 1.83) / 3600 - 4.5) < 0.2
    assert W.default_max_run_s(500, 1.9) < 1.2 * 3600
    assert W.default_max_run_s(0, 1.83) == 4.5 * 3600


def test_speed_alarm_above_1_5x_expected_and_not_below():
    # 250 steps per 8 min = 1.92 s/step vs expected 1.83: fine
    w = watch_(expected_s_per_step=1.83)
    for i, t in enumerate((10, 18, 26, 34), start=1):
        w.observe(t * MIN, {250 * i}, tasks=1)
    assert w.take_alarms() == [] and abs(w.s_per_step - 1.92) < 0.01
    # 250 steps per 15 min = 3.6 s/step: alarm (and the run keeps going)
    w = watch_(expected_s_per_step=1.83)
    verdicts = [w.observe(t * MIN, {250 * i}, tasks=1) for i, t in enumerate((10, 25, 40), start=1)]
    assert verdicts == ["ok"] * 3
    alarms = w.take_alarms()
    assert alarms and "SPEED ALARM" in alarms[-1] and w.s_per_step > 1.5 * 1.83
    assert w.take_alarms() == []


def test_speed_is_not_measured_across_a_retry_or_over_markers_already_present():
    w = W.StallWatch(0.0, 40 * MIN, 75 * MIN, expected_s_per_step=1.83)
    w.observe(0.0, {1500}, tasks=1)  # a resumed run: the markers already there are a baseline, not progress
    w.observe(30 * MIN, {1500, 1750}, tasks=1)  # staging time is inside this first interval: no speed from one marker
    assert w.s_per_step is None and w.take_alarms() == []
    w.observe(38 * MIN, {1500, 1750, 2000}, tasks=1)
    assert abs(w.s_per_step - 1.92) < 0.01
    w.observe(40 * MIN, {2000}, tasks=0)
    w.observe(100 * MIN, {2000}, tasks=1)  # the retry restages; this long gap must not show up as slow steps
    w.observe(110 * MIN, {2000, 2250}, tasks=1)
    w.observe(118 * MIN, {2000, 2250, 2500}, tasks=1)
    assert abs(w.s_per_step - 1.92) < 0.01 and w.take_alarms() == []


def test_a_final_checkpoint_or_a_stopped_app_is_finished_never_a_stall():
    w = watch_()
    assert w.observe(10 ** 6, {250}, final=True, tasks=1) == "finished"
    assert w.observe(10 ** 6, {250}, app_state="stopped", tasks=0) == "finished"
    assert watch_().observe(10 ** 6, set(), app_state="ephemeral", tasks=1) == "stop"  # control


def test_queue_limit_stops_an_app_that_never_gets_a_gpu():
    w = watch_(max_queue_s=240 * MIN)
    assert w.observe(200 * MIN, set(), tasks=0) == "ok"
    assert w.observe(250 * MIN, set(), tasks=0) == "stop" and w.reason == "queue"


def test_uncounted_polls_are_paused_time():
    w = watch_()
    w.observe(10 * MIN, {250}, tasks=1)
    assert w.observe(90 * MIN, {250}, tasks=1, counted=False) == "ok"  # the CLI was failing for 80 min
    assert w.observe(100 * MIN, {250}, tasks=1) == "ok"  # only 10 of the 90 min counted since the marker


class FakeOps:
    def __init__(self, script, clock, info=None):
        self.script, self.clock, self.stopped, self.log_calls = script, clock, False, []
        self.info = info or (lambda t: ("ephemeral", 1))

    def listing(self):
        return self.script(self.clock.t)

    def app_info(self):
        return self.info(self.clock.t)

    def logs(self, since, until):
        self.log_calls.append((since, until))
        return f"line {len(self.log_calls)}\n"

    def stop(self):
        self.stopped = True
        return True


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def run(tmp_path, script, info=None, stall=40, grace=75, max_run=float("inf"), expected=0.0, **kw):
    c = Clock()
    t0 = c.t
    ops = FakeOps(lambda t: script(t - t0), c, info and (lambda t: info(t - t0)))
    msgs = []
    rc = W.watch(ops, str(tmp_path), stall * MIN, grace * MIN, 120.0, kw.get("logs_every", 600.0), max_run, expected,
                 clock=c, sleep=c.sleep, say=msgs.append)
    return rc, ops, msgs


def test_loop_stops_a_stalled_run_and_keeps_its_logs(tmp_path):
    rc, ops, msgs = run(tmp_path, lambda t: ({250, 500} if t > 20 * MIN else {250} if t > 10 * MIN else set(), False))
    assert rc == 2 and ops.stopped
    assert any("STALL" in m for m in msgs)
    text = open(os.path.join(tmp_path, "app.log")).read()
    assert text.startswith("line 1\n") and text.count("line") >= 3  # periodic fetches plus the one at the stop


def test_loop_finishes_cleanly_and_a_failed_listing_is_not_a_stall(tmp_path):
    def script(t):
        if 5 * MIN < t < 200 * MIN:
            return None  # the CLI is failing: not evidence of a stall
        return ({250}, t > 200 * MIN)

    rc, ops, msgs = run(tmp_path, script)
    assert rc == 0 and not ops.stopped
    assert any("listing failed" in m for m in msgs)


def test_a_failed_app_list_is_paused_time_too(tmp_path):
    rc, ops, msgs = run(tmp_path, lambda t: ({250}, t > 300 * MIN), info=lambda t: None if 5 * MIN < t < 290 * MIN else ("e", 1))
    assert rc == 0 and not ops.stopped and any("app list failed" in m for m in msgs)


def test_loop_stops_at_the_max_run_time_with_exit_4_and_an_alarm_file(tmp_path):
    rc, ops, msgs = run(tmp_path, lambda t: ({250 * int(t // (8 * MIN))} - {0}, False), max_run=3 * 3600)
    assert rc == 4 and ops.stopped
    assert any("MAX RUN TIME" in m for m in msgs)
    assert "max_run" in open(os.path.join(tmp_path, "ALARM.txt")).read()
    st = json.load(open(os.path.join(tmp_path, "status.json")))
    assert st["verdict"] == "stop" and st["run_min"] > 180


def test_loop_logs_a_speed_alarm_and_keeps_running(tmp_path):
    # a marker every 15 min = 3.6 s/step against 1.83 expected, until the run finishes at 2 h
    rc, ops, msgs = run(tmp_path, lambda t: ({250 * int(t // (15 * MIN)), 250} - {0}, t > 120 * MIN),
                        expected=1.83)
    assert rc == 0 and not ops.stopped
    assert any("SPEED ALARM" in m for m in msgs)
    assert "SPEED ALARM" in open(os.path.join(tmp_path, "ALARM.txt")).read()


def test_loop_survives_a_queue_and_a_retry_gap(tmp_path):
    def info(t):  # queued for 3 h, runs, preempted at 1 h of running, back after 50 min
        return ("e", 0 if t < 180 * MIN or 240 * MIN < t < 290 * MIN else 1)

    def script(t):
        return ({250 * (i + 1) for i, at in enumerate((200, 230, 320, 350)) if t > at * MIN}, t > 380 * MIN)

    rc, ops, msgs = run(tmp_path, script, info=info, stall=40, grace=75)
    assert rc == 0 and not ops.stopped


def test_log_ranges_do_not_overlap(tmp_path):
    rc, ops, _ = run(tmp_path, lambda t: ({250}, t > 40 * MIN), logs_every=600.0)
    ends = [u for _, u in ops.log_calls]
    starts = [s for s, _ in ops.log_calls]
    assert len(ops.log_calls) >= 3
    for prev_end, nxt_start in zip(ends, starts[1:]):
        assert nxt_start <= prev_end  # each range starts at (just before) the previous end: no gap


def test_modal_app_info_reads_tasks_and_distinguishes_failure_from_absence(monkeypatch):
    m = W.Modal("vol", "run", "ap-X")
    listing = json.dumps([{"app_id": "ap-X", "state": "ephemeral (detached)", "tasks": "1"},
                          {"app_id": "ap-Y", "state": "stopped", "tasks": "0"}])
    monkeypatch.setattr(m, "_run", lambda *a, **k: listing)
    assert m.app_info() == ("ephemeral (detached)", 1)
    m.app = "ap-Y"
    assert m.app_info() == ("stopped", 0)
    m.app = "ap-Z"
    assert m.app_info() == ("", -1)  # not named: the count is unknown, assume running
    monkeypatch.setattr(m, "_run", lambda *a, **k: None)
    assert m.app_info() is None  # the CLI failed
    monkeypatch.setattr(m, "_run", lambda *a, **k: "not json")
    assert m.app_info() is None


def test_modal_stop_passes_yes(monkeypatch):
    m = W.Modal("vol", "run", "ap-X")
    seen = []
    monkeypatch.setattr(m, "_run", lambda *a, **k: seen.append(a) or "")
    assert m.stop() and seen == [("app", "stop", "--yes", "ap-X")]  # red-on-base: the prompt aborts a non-interactive stop


def test_stage_watch_reads_the_rate_and_judges_only_unfinished_staging():
    sw = W.StageWatch(300.0, 600.0, 900.0)
    assert sw.check(700) == "ok" or sw.check(700) == "unknown"
    assert sw.check(100) == "ok" and sw.check(1000) == "unknown"  # no rate line at all: unknown, never "slow"
    sw.feed("noise\nstaging 5000/521000 files 65 files/s\n")
    assert sw.rate == 65.0 and (sw.done, sw.total) == (5000, 521000)
    assert sw.check(300) == "ok"  # too early to judge
    assert sw.check(601) == "slow"
    sw.feed("staging 10000/521000 files 800 files/s")
    assert sw.rate == 800.0 and sw.check(601) == "ok"
    sw.feed("staged 521000 files to /tmp/stage in 40s")
    assert sw.finished and sw.check(10 ** 6) == "ok" and abs(sw.rate - 13025) < 1
    slow_done = W.StageWatch()
    slow_done.feed("staged 1000 files to /tmp/stage in 100s")  # finished slowly: reported, not stopped
    assert slow_done.check(10 ** 6) == "ok" and slow_done.rate == 10.0
    off = W.StageWatch(0.0)
    off.feed("staging 5000/9000 files 1 files/s")
    assert off.check(10 ** 6) == "ok"
    sw.reset()
    assert sw.rate is None and not sw.finished


class StagingOps(FakeOps):
    def __init__(self, script, clock, info=None, lines=None):
        super().__init__(script, clock, info)
        self.lines, self.t0 = lines, clock.t

    def logs(self, since, until):
        super().logs(since, until)
        return self.lines(self.clock.t - self.t0)


def run_staging(tmp_path, lines, script=lambda t: (set(), False), min_rate=300.0, info=None):
    c = Clock()
    t0 = c.t
    ops = StagingOps(lambda t: script(t - t0), c, info and (lambda t: info(t - t0)), lines)
    msgs = []
    rc = W.watch(ops, str(tmp_path), 40 * MIN, 75 * MIN, 120.0, 600.0, stage=W.StageWatch(min_rate, 600.0, 900.0),
                 clock=c, sleep=c.sleep, say=msgs.append)
    return rc, ops, msgs


def test_slow_staging_stops_the_app_after_ten_minutes_and_says_why(tmp_path):
    rc, ops, msgs = run_staging(tmp_path, lambda t: f"staging {int(t * 65)}/521000 files 65 files/s\n")
    assert rc == 6 and ops.stopped
    assert any("SLOW STAGING" in m and "65 files/s" in m for m in msgs)
    st = json.load(open(os.path.join(tmp_path, "status.json")))
    assert st["staging_rate"] == 65.0 and st["run_min"] < 13  # stopped soon after the 10 min mark
    assert "slow_staging" in open(os.path.join(tmp_path, "ALARM.txt")).read()


def test_fast_staging_does_not_stop_and_the_run_goes_on(tmp_path):
    def lines(t):
        if t < 5 * MIN:
            return ""
        return "staging 100000/521000 files 800 files/s\n" + ("staged 521000 files to /tmp/stage in 650s\n" if t > 12 * MIN else "")

    rc, ops, msgs = run_staging(tmp_path, lines, script=lambda t: ({250}, t > 60 * MIN))
    assert rc == 0 and not ops.stopped


def test_staging_that_is_only_slow_in_the_queue_does_not_count_queue_time(tmp_path):
    rc, ops, msgs = run_staging(tmp_path, lambda t: "staging 5000/521000 files 65 files/s\n" if t > 200 * MIN else "",
                                info=lambda t: ("e", 0 if t < 180 * MIN else 1), script=lambda t: (set(), t > 188 * MIN))
    assert rc == 0 and not ops.stopped  # slow, but the run is over before 10 min of task time have passed
