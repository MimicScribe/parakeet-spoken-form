"""Cost guard for a long training run: stop the Modal app when checkpoints stop appearing or the run takes too long,
and keep its logs.

    python3 run_watch.py --app ap-XXXX --run <run name> --log-dir <dir> [--steps N]   (beside the launch, under caffeinate -i)

Every --poll-s it lists exp/<run> on the volume and reads the app's task count. Only time with a RUNNING TASK counts: while
the app has none (the GPU queue, the gap between a preemption and its retry) the grace, stall and spend clocks are paused.
  * grace: no step marker before --grace-min (default 75) of task time -> stop (staging and loading come first);
  * stall: no new `step<N>.pt.done` for --stall-min (default 40) of task time -> stop. A retry (the task count goes
    1 -> 0 -> 1) resets this clock, since the retry stages the data and loads the checkpoint again;
  * spend tripwire: more than --max-run-h of task time -> stop. Default from --steps: steps * --expected-s-per-step * 1.5
    plus 45 min of staging / retry slack (5,000 steps at 1.83 s/step -> 4.6 h); without --steps, 4.5 h;
  * speed alarm: the s/step measured between step markers of one attempt is above 1.5 x --expected-s-per-step -> a SPEED
    ALARM line (and <log-dir>/ALARM.txt); an alarm never stops the run, the max-run tripwire does;
  * queue limit: more than --max-queue-min (default 240) with no running task in total -> stop (nothing is billed meanwhile,
    but nothing is happening either).
It appends the app's logs to <log-dir>/app.log every --logs-every-s (non-overlapping time ranges) so the evidence survives
a stop, and rewrites <log-dir>/status.json each poll. Exit codes: 0 the run finished (decoder_joint.pt exists, or the app
stopped by itself), 2 stalled or no first marker, 3 interrupted, 4 max run time, 5 queue limit. A failing `modal` call
(network, CLI) is never read as a stall: it is logged and the clocks are paused for that poll.
The decision logic (StallWatch) is separate from the `modal` calls so it is tested without a network.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

DONE = re.compile(r"step(\d+)\.pt\.done")
SLACK_S = 45 * 60  # staging + one retry's reload, added to the default max run time
DEFAULT_MAX_RUN_S = 4.5 * 3600


def default_max_run_s(steps: int, expected_s_per_step: float, factor: float = 1.5) -> float:
    return steps * expected_s_per_step * factor + SLACK_S if steps else DEFAULT_MAX_RUN_S


class StallWatch:
    """What to do given the checkpoint markers and task count seen so far.

    observe() returns "ok", "finished", or "stop"; after a "stop", .reason is "stall", "no_first_marker", "max_run" or
    "queue". Time is credited to the run only when the app had a running task at both ends of the interval (tasks == 0 is
    the queue or a retry gap; tasks == -1, the default, means "not known, assume running"; counted=False skips an interval
    whose polls failed). Markers already present at the first observation (a resumed run) are a baseline, not progress.
    """

    def __init__(self, start: float, stall_s: float, grace_s: float, max_run_s: float = float("inf"),
                 expected_s_per_step: float = 0.0, speed_factor: float = 1.5, max_queue_s: float = float("inf")):
        self.stall_s, self.grace_s, self.max_run_s = stall_s, grace_s, max_run_s
        self.expected, self.factor, self.max_queue_s = expected_s_per_step, speed_factor, max_queue_s
        self.last_t, self.prev_running = start, None
        self.run_s = self.queue_s = self.idle_s_ = 0.0  # task time, no-task time, task time since the last progress
        self.max_step, self.new_marker, self.baseline = 0, False, None
        self.attempt, self.ever_ran, self.in_gap = 0, False, False
        self.window = []  # (step, run_s) of the markers seen in this attempt
        self.s_per_step = None
        self.alarms = []
        self.reason = ""

    def observe(self, now: float, steps, final: bool = False, app_state: str = "", tasks: int = -1,
                counted: bool = True) -> str:
        dt, self.last_t = max(0.0, now - self.last_t), now
        if counted:
            running = tasks != 0
            if running and self.prev_running is not False:
                self.run_s += dt
                self.idle_s_ += dt
            else:
                self.queue_s += dt
            self.prev_running = running
            if running:
                if self.in_gap:  # a retry: it stages and loads again, so the stall clock and the speed window restart
                    self.attempt, self.idle_s_, self.window, self.in_gap = self.attempt + 1, 0.0, [], False
                self.ever_ran = True
            elif self.ever_ran:
                self.in_gap = True
        if final or app_state == "stopped":
            return "finished"
        top = max(steps, default=0)
        if self.baseline is None:
            self.baseline = self.max_step = top
        elif top > self.max_step:
            self.max_step, self.idle_s_, self.new_marker = top, 0.0, True
            self.window.append((top, self.run_s))
            self._speed()
        limit = self.stall_s if self.new_marker else max(self.grace_s, self.stall_s)
        if self.idle_s_ > limit:
            self.reason = "stall" if self.new_marker else "no_first_marker"
            return "stop"
        if self.run_s > self.max_run_s:
            self.reason = "max_run"
            return "stop"
        if self.queue_s > self.max_queue_s:
            self.reason = "queue"
            return "stop"
        return "ok"

    def _speed(self) -> None:
        if len(self.window) < 2:
            return
        (s0, t0), (s1, t1) = self.window[0], self.window[-1]
        self.s_per_step = (t1 - t0) / (s1 - s0)
        if self.expected and self.s_per_step > self.factor * self.expected:
            self.alarms.append(f"SPEED ALARM: {self.s_per_step:.2f} s/step over steps {s0}-{s1} of this attempt, "
                               f"expected {self.expected:.2f} (limit {self.factor * self.expected:.2f})")

    def idle_s(self, now: float = 0.0) -> float:
        return self.idle_s_

    def take_alarms(self) -> list:
        out, self.alarms = self.alarms, []
        return out


class Modal:
    """The `modal` calls the watchdog needs; each returns None on failure instead of raising."""

    def __init__(self, volume: str, run: str, app: str):
        self.volume, self.run, self.app = volume, run, app

    def _run(self, *args, timeout=120):
        try:
            r = subprocess.run(["modal", *args], capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None

    def listing(self):
        """(steps, final) of exp/<run>, or None when the listing failed."""
        out = self._run("volume", "ls", self.volume, f"exp/{self.run}")
        if out is None:
            return None
        return {int(m.group(1)) for m in DONE.finditer(out)}, "decoder_joint.pt" in out

    def app_info(self):
        """(state, running task count) of the app; ("", -1) when the list does not name it (count unknown, assume
        running); None when the call or its JSON failed."""
        out = self._run("app", "list", "--json", "--limit", "200")
        try:
            apps = json.loads(out) if out is not None else None
            if not isinstance(apps, list):
                return None
            for a in apps:
                if a.get("app_id") == self.app:
                    return str(a.get("state", "")), int(str(a.get("tasks", "0")) or 0)
        except (ValueError, TypeError):
            return None
        return "", -1

    def logs(self, since: str, until: str):
        return self._run("app", "logs", self.app, "--since", since, "--until", until, "--timestamps", timeout=300)

    def stop(self) -> bool:
        return self._run("app", "stop", "--yes", self.app) is not None  # without --yes the CLI asks and aborts (no terminal)


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


REASON_TEXT = {
    "stall": "STALL: no new step marker for {idle:.0f} min of task time (last step {step})",
    "no_first_marker": "NO FIRST MARKER after {idle:.0f} min of task time",
    "max_run": "MAX RUN TIME: {run:.0f} min of task time (limit {limit:.0f} min; last step {step})",
    "queue": "QUEUE LIMIT: {queue:.0f} min without a running task",
}
EXIT_CODE = {"stall": 2, "no_first_marker": 2, "max_run": 4, "queue": 5}


def watch(ops, log_dir: str, stall_s: float, grace_s: float, poll_s: float, logs_every_s: float,
          max_run_s: float = float("inf"), expected_s_per_step: float = 0.0, max_queue_s: float = float("inf"),
          clock=time.time, sleep=time.sleep, say=print) -> int:
    os.makedirs(log_dir, exist_ok=True)
    w = StallWatch(clock(), stall_s, grace_s, max_run_s, expected_s_per_step, max_queue_s=max_queue_s)
    last_logs, bad, last_beat = clock(), 0, 0.0
    steps, final = set(), False
    log_path = os.path.join(log_dir, "app.log")

    def persist(now):
        nonlocal last_logs
        text = ops.logs(iso(last_logs - 1), iso(now))
        if text is not None:
            with open(log_path, "a") as f:
                f.write(text)
            last_logs = now

    def status(now, verdict, tasks):
        doc = dict(time=iso(now), verdict=verdict, last_step=w.max_step, tasks=tasks, attempt=w.attempt,
                   run_min=round(w.run_s / 60, 1), queue_min=round(w.queue_s / 60, 1),
                   idle_min=round(w.idle_s_ / 60, 1), s_per_step=w.s_per_step and round(w.s_per_step, 3),
                   max_run_min=None if max_run_s == float("inf") else round(max_run_s / 60, 1))
        tmp = os.path.join(log_dir, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, os.path.join(log_dir, "status.json"))
        return doc

    try:
        while True:
            now = clock()
            got, info = ops.listing(), ops.app_info()
            ok = got is not None and info is not None
            if got is not None:
                steps, final = got
            if not ok:
                bad += 1
                say(f"{iso(now)} {'listing' if got is None else 'app list'} failed ({bad} in a row); the clocks are paused")
            else:
                bad = 0
            state, tasks = info if info is not None else ("", -1)
            verdict = w.observe(now, steps, final, state, tasks, counted=ok)
            for alarm in w.take_alarms():
                say(f"{iso(now)} {alarm}")
                with open(os.path.join(log_dir, "ALARM.txt"), "a") as f:
                    f.write(f"{iso(now)} {alarm}\n")
            doc = status(now, verdict, tasks)
            if now - last_beat >= 1800:
                last_beat = now
                say(f"{iso(now)} heartbeat: step {doc['last_step']}, {doc['run_min']} min of task time, "
                    f"{doc['queue_min']} min queued, {doc['s_per_step']} s/step, attempt {doc['attempt']}")
            if now - last_logs >= logs_every_s or verdict != "ok":
                persist(now)
            if verdict == "finished":
                how = "finished (decoder_joint.pt written)" if final else "app stopped on its own"
                say(f"{iso(now)} run {how}, last step marker {w.max_step}; logs in {log_path}")
                return 0
            if verdict == "stop":
                say(f"{iso(now)} " + REASON_TEXT[w.reason].format(
                    idle=w.idle_s_ / 60, step=w.max_step, run=w.run_s / 60, limit=max_run_s / 60, queue=w.queue_s / 60)
                    + " -> stopping the app")
                stopped = ops.stop()
                say(f"{iso(clock())} modal app stop {'ok' if stopped else 'FAILED, stop it by hand'}")
                with open(os.path.join(log_dir, "ALARM.txt"), "a") as f:
                    f.write(f"{iso(now)} app stopped by the watchdog: {w.reason}\n")
                persist(clock())
                return EXIT_CODE[w.reason]
            sleep(poll_s)
    except KeyboardInterrupt:
        say("interrupted; the app keeps running")
        return 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--app", required=True, help="the Modal app id (ap-...) of the launch")
    ap.add_argument("--run", required=True, help="run name (exp/<run> on the volume)")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--volume", default="parakeet-spoken-form")
    ap.add_argument("--stall-min", type=float, default=40.0, help="no new step marker for this much task time -> stop")
    ap.add_argument("--grace-min", type=float, default=75.0, help="the same, before the FIRST marker (staging, loading)")
    ap.add_argument("--steps", type=int, default=0, help="the launch's --max-steps (sets the default max run time)")
    ap.add_argument("--expected-s-per-step", type=float, default=1.83)
    ap.add_argument("--max-run-h", type=float, default=0.0, help="task-time limit; 0 = derived from --steps")
    ap.add_argument("--max-queue-min", type=float, default=240.0, help="total time without a running task -> stop")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--logs-every-s", type=float, default=600.0)
    a = ap.parse_args(argv)
    max_run = a.max_run_h * 3600 if a.max_run_h else default_max_run_s(a.steps, a.expected_s_per_step)
    return watch(Modal(a.volume, a.run, a.app), a.log_dir, a.stall_min * 60, a.grace_min * 60, a.poll_s,
                 a.logs_every_s, max_run, a.expected_s_per_step, a.max_queue_min * 60)


if __name__ == "__main__":
    sys.exit(main())
