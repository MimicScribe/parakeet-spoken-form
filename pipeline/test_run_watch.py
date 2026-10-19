"""Stall watchdog decisions and loop (no network, no modal CLI): python -m pytest test_run_watch.py -q"""
import os

import run_watch as W

MIN = 60.0


def test_progress_resets_the_clock_and_a_stall_stops():
    w = W.StallWatch(0.0, 25 * MIN, 45 * MIN)
    assert w.observe(10 * MIN, set()) == "ok"
    assert w.observe(44 * MIN, set()) == "ok"  # grace before the first marker
    assert w.observe(46 * MIN, set()) == "stop"  # no first marker at all
    w = W.StallWatch(0.0, 25 * MIN, 45 * MIN)
    assert w.observe(30 * MIN, {250}) == "ok"
    assert w.observe(54 * MIN, {250}) == "ok"  # 24 min since step 250
    assert w.observe(56 * MIN, {250}) == "stop"  # 26 min
    w = W.StallWatch(0.0, 25 * MIN, 45 * MIN)
    w.observe(30 * MIN, {250})
    assert w.observe(54 * MIN, {250, 500}) == "ok"  # a new marker at 54 min restarts the clock
    assert w.observe(78 * MIN, {250, 500}) == "ok"
    assert w.observe(80 * MIN, {250, 500}) == "stop"


def test_a_final_checkpoint_or_a_stopped_app_is_finished_never_a_stall():
    w = W.StallWatch(0.0, 25 * MIN, 45 * MIN)
    assert w.observe(10 ** 6, {250}, final=True) == "finished"
    assert w.observe(10 ** 6, {250}, app_state="stopped") == "finished"
    assert W.StallWatch(0.0, 25 * MIN, 45 * MIN).observe(10 ** 6, set(), app_state="ephemeral") == "stop"  # control


class FakeOps:
    def __init__(self, script, clock):
        self.script, self.clock, self.stopped, self.log_calls = script, clock, False, []

    def listing(self):
        return self.script(self.clock.t)

    def state(self):
        return ""

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


def run(tmp_path, script, **kw):
    c = Clock()
    t0 = c.t
    ops = FakeOps(lambda t: script(t - t0), c)
    msgs = []
    rc = W.watch(ops, str(tmp_path), 25 * MIN, 45 * MIN, 120.0, kw.get("logs_every", 600.0),
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
        if 5 * MIN < t < 100 * MIN:
            return None  # the CLI is failing: not evidence of a stall
        return ({250}, t > 100 * MIN)

    rc, ops, msgs = run(tmp_path, script)
    assert rc == 0 and not ops.stopped
    assert any("listing failed" in m for m in msgs)


def test_log_ranges_do_not_overlap(tmp_path):
    rc, ops, _ = run(tmp_path, lambda t: ({250}, t > 40 * MIN), logs_every=600.0)
    ends = [u for _, u in ops.log_calls]
    starts = [s for s, _ in ops.log_calls]
    assert len(ops.log_calls) >= 3
    for prev_end, nxt_start in zip(ends, starts[1:]):
        assert nxt_start <= prev_end  # each range starts at (just before) the previous end: no gap
