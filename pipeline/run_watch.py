"""Cost guard for a long training run: stop the Modal app when checkpoints stop appearing, and keep its logs.

    python3 run_watch.py --app ap-XXXX --run <run name> --log-dir <dir>     (leave it running beside the launch)

Every --poll-s it lists exp/<run> on the volume. A new `step<N>.pt.done` (SaveEvery writes one per checkpoint) resets the
clock. No new marker within --stall-min (--grace-min before the first one: file staging and model loading come first)
-> `modal app stop`. It also appends the app's logs to <log-dir>/app.log every --logs-every-s (non-overlapping time
ranges), so the evidence survives a stop. Exit codes: 0 the run finished (decoder_joint.pt exists, or the app
stopped by itself), 2 stopped because of a stalled checkpoint cadence, 3 interrupted. A failing `modal` call (network,
CLI) is never read as a stall: it is logged and the stall clock is paused for that poll.
The decision logic (StallWatch) is separate from the `modal` calls so it is tested without a network.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

DONE = re.compile(r"step(\d+)\.pt\.done")


class StallWatch:
    """What to do given the checkpoint markers seen so far. observe() returns "ok", "finished" or "stop"."""

    def __init__(self, start: float, stall_s: float, grace_s: float):
        self.start, self.stall_s, self.grace_s = start, stall_s, grace_s
        self.last_progress, self.max_step = start, 0

    def observe(self, now: float, steps, final: bool = False, app_state: str = "") -> str:
        if final or app_state == "stopped":
            return "finished"
        top = max(steps, default=0)
        if top > self.max_step:
            self.max_step, self.last_progress = top, now
        limit = self.stall_s if self.max_step else max(self.grace_s, self.stall_s)
        return "stop" if now - self.last_progress > limit else "ok"

    def idle_s(self, now: float) -> float:
        return now - self.last_progress

    def pause(self, dt: float) -> None:
        """A poll that could not read the volume tells nothing about the run: that interval does not count."""
        self.last_progress += dt


class Modal:
    """The three `modal` calls the watchdog needs; each returns None on failure instead of raising."""

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

    def state(self) -> str:
        out = self._run("app", "list", "--json", "--limit", "200")
        try:
            for a in json.loads(out or "[]"):
                if a.get("app_id") == self.app:
                    return str(a.get("state", ""))
        except ValueError:
            pass
        return ""

    def logs(self, since: str, until: str):
        return self._run("app", "logs", self.app, "--since", since, "--until", until, "--timestamps", timeout=300)

    def stop(self) -> bool:
        return self._run("app", "stop", self.app) is not None


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def watch(ops, log_dir: str, stall_s: float, grace_s: float, poll_s: float, logs_every_s: float,
          clock=time.time, sleep=time.sleep, say=print) -> int:
    import os

    os.makedirs(log_dir, exist_ok=True)
    w = StallWatch(clock(), stall_s, grace_s)
    last_logs, bad, last_poll = clock(), 0, clock()
    steps, final = set(), False
    log_path = os.path.join(log_dir, "app.log")

    def persist(now):
        nonlocal last_logs
        text = ops.logs(iso(last_logs - 1), iso(now))
        if text is not None:
            with open(log_path, "a") as f:
                f.write(text)
            last_logs = now

    try:
        while True:
            now = clock()
            got = ops.listing()
            if got is None:
                bad += 1
                w.pause(now - last_poll)
                say(f"{iso(now)} listing failed ({bad} in a row); the stall clock is paused")
            else:
                bad, (steps, final) = 0, got
            last_poll = now
            verdict = w.observe(now, steps, final, ops.state())
            if now - last_logs >= logs_every_s or verdict != "ok":
                persist(now)
            if verdict == "finished":
                how = "finished (decoder_joint.pt written)" if final else "app stopped on its own"
                say(f"{iso(now)} run {how}, last step marker {w.max_step}; logs in {log_path}")
                return 0
            if verdict == "stop":
                say(f"{iso(now)} STALL: no new step marker for {w.idle_s(now) / 60:.0f} min (last step {w.max_step}) -> "
                    f"stopping the app")
                ok = ops.stop()
                say(f"{iso(clock())} modal app stop {'ok' if ok else 'FAILED, stop it by hand'}")
                persist(clock())
                return 2
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
    ap.add_argument("--stall-min", type=float, default=25.0, help="no new step marker for this long -> stop")
    ap.add_argument("--grace-min", type=float, default=45.0, help="the same, before the FIRST marker (staging, loading)")
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--logs-every-s", type=float, default=600.0)
    a = ap.parse_args(argv)
    return watch(Modal(a.volume, a.run, a.app), a.log_dir, a.stall_min * 60, a.grace_min * 60, a.poll_s, a.logs_every_s)


if __name__ == "__main__":
    sys.exit(main())
