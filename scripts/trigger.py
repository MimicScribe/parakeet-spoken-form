"""Local orchestrator for one night's run set: launches the follow-on runs and scores checkpoints as their markers appear.

    python3 trigger.py --config <config.json> [--once] [--dry-run]        (normally through trigger.sh, under caffeinate -i)

It never launches a main run: those are started by hand through launch_run.sh. It watches the volume for
exp/<main>/step<branch_step>.pt.done and then, through launch_run.sh only,
  * launches the two branch runs (decoder-only control `brC`, then the encoder branch `brE`, `stagger_s` apart);
  * launches the real-only encoder branch `brEr` only if the flag file named in the config exists;
  * scores every branch checkpoint (step 250 and 500) and runs the interim gate on C vs E at each; when the gate prints GO
    at every gate step and the printed `signs` lines agree, it launches the full encoder continuation `encF`;
  * scores the planned checkpoints of the main, half-lr and encF runs as each `step<N>.pt.done` appears;
  * applies the early-stop rule to the main run at `early_stop.step` (stops its app).
Every action goes to the log file; the state file makes a restart harmless: a run whose exp/<run> exists on the volume (or
whose launch the state records) is never launched again, a listing that fails is never read as "absent", and a scoring
job already done is not repeated. The approvals file is only read by launch_run.sh: this script never writes it, so a run
without its approval line is refused by the launcher and retried (logged once) until the line appears.

Contracts with the scoring tools (argv templates `score_argv` / `gate_argv` in the config; {score_script} {gate_script}
{run} {step} {out} {c_dir} {e_dir} {python} are filled in; the defaults are `<scorer> <run> <step> <out>` and
`python <gate> --step <step> --c-dir <dir> --e-dir <dir>`):
  * the scorer is run once per checkpoint and writes <out>/summary.json; exit 0 = scored. The early-stop rule reads
    `pooled_v_vs_stock_pp` (number) and `wer_ok` (bool) from the summary of the main run's early_stop step and of its
    branch_step checkpoint (key names are config values);
  * the gate is run with the two score dirs of one step; its output holds a verdict token GO or NO-GO, and optionally a
    line `signs: <list>` (the sign of each E-C effect); the lists of the gate steps must be equal for the sign agreement.
A missing scorer or gate is logged once and waited for, never treated as a failure.
"""
import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime

MARK = re.compile(r"step(\d+)\.pt\.done$")
RETRY_S = 300  # a refused / transiently failed launch is tried again no sooner than this
MAX_ATTEMPTS = 3  # for a launcher refusal that retrying cannot fix (bad arguments, wrong commit)
ROLES = ("main", "lr05", "brC", "brE", "brEr", "encF")


def local_time(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# ---------------------------------------------------------------------------------------------------------------- I/O


class Volume:
    """`modal volume ls`; a failed call is "error", never "absent"."""

    def __init__(self, volume: str):
        self.volume = volume

    def ls(self, path: str):
        """("ok", [entries]) | ("absent", []) | ("error", [])"""
        try:
            r = subprocess.run(["modal", "volume", "ls", self.volume, path], capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError):
            return "error", []
        if r.returncode == 0:
            return "ok", [l.strip() for l in r.stdout.splitlines() if l.strip()]
        if "does not exist" in (r.stdout + r.stderr):
            return "absent", []
        return "error", []


class Exec:
    """Blocking and background commands."""

    def run(self, argv, timeout, log_path=None, cwd=None):
        """(returncode, output); returncode None on timeout or when the command cannot start."""
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        except subprocess.TimeoutExpired as e:
            return None, f"timeout after {timeout}s: {e.stdout or ''}"
        except OSError as e:
            return None, f"cannot start: {e}"
        out = r.stdout + r.stderr
        if log_path:
            with open(log_path, "a") as f:
                f.write(out)
        return r.returncode, out

    def start(self, argv, log_path, cwd=None):
        f = open(log_path, "a")
        return subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)


# ----------------------------------------------------------------------------------------------------------- the logic


def parse_manifest(path: str) -> dict:
    """What the main run's launch logged: commit, app id, and the training command's mix flags."""
    out = {}
    try:
        text = open(path).read()
    except OSError:
        return out
    m = re.search(r"^commit (\S+)", text, re.M)
    if m:
        out["sha"] = m.group(1)
    apps = re.findall(r"^app (ap-\S+)", text, re.M)
    if apps:
        out["app"] = apps[0]
    m = re.search(r"^command (.*)$", text, re.M)
    if m:
        try:
            toks = shlex.split(m.group(1))
        except ValueError:
            toks = []
        for flag, key in (("--mix-dir", "mix_dir"), ("--part-weights", "part_weights"), ("--extra-parts", "extra_parts")):
            if flag in toks and toks.index(flag) + 1 < len(toks):
                out[key] = toks[toks.index(flag) + 1]
    return out


def parse_gate(output: str):
    """(verdict, signs): verdict "go" | "nogo" | None; signs the text after a `signs:` line, or None."""
    verdict = None
    if re.search(r"\bNO[-_ ]?GO\b", output, re.I):
        verdict = "nogo"
    elif re.search(r"\bGO\b", output):
        verdict = "go"
    m = re.search(r"^\s*signs?\s*[:=]\s*(.+?)\s*$", output, re.I | re.M)
    return verdict, (re.sub(r"\s+", "", m.group(1)) if m else None)


def early_stop_decision(v_now, v_ref, wer_ok, limit_pp: float):
    """("stop"|"keep"|"undetermined", reason). Stop when WER is broken, or the pooled V vs stock is above limit_pp and
    worse than at the reference checkpoint."""
    if wer_ok is False:
        return "stop", "WER broken"
    if v_now is None or wer_ok is None:
        return "undetermined", "summary lacks the pooled V or the WER verdict"
    if v_now > limit_pp:
        if v_ref is None:
            return "undetermined", f"V {v_now:+.1f} pp > {limit_pp:+.1f} but the reference checkpoint has no V"
        if v_now > v_ref:
            return "stop", f"pooled V {v_now:+.1f} pp > {limit_pp:+.1f} and worse than {v_ref:+.1f} at the reference"
        return "keep", f"pooled V {v_now:+.1f} pp > {limit_pp:+.1f} but not worse than {v_ref:+.1f}"
    return "keep", f"pooled V {v_now:+.1f} pp within {limit_pp:+.1f}"


class Trigger:
    def __init__(self, cfg: dict, vol, ex, clock=time.time, dry_run=False, say=None):
        self.cfg, self.vol, self.ex, self.clock, self.dry = cfg, vol, ex, clock, dry_run
        self.say_out = say or print
        self.runs = cfg["runs"]  # role -> {"run": name, "arm": arm}
        self.state_path = cfg["state_file"]
        self.state = self.load_state()
        self.jobs = {}  # score key -> (handle, started)
        self.warned = set()
        self.markers = {}  # run name -> set of steps with a .done marker (None when the listing failed)
        self.exists = {}  # run name -> bool, from the exp/ listing
        self.listing_ok = False
        self.launch_cfg = {}
        self.last_beat = 0.0

    # --- logging and state -----------------------------------------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"{local_time(self.clock())} {msg}"
        self.say_out(line)
        try:
            with open(self.cfg["log_file"], "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def once(self, key: str, msg: str) -> None:
        """Log a waiting / problem condition once per episode (clear_warn() when it clears)."""
        if key not in self.warned:
            self.warned.add(key)
            self.log(msg)

    def clear_warn(self, key: str) -> None:
        self.warned.discard(key)

    def load_state(self) -> dict:
        try:
            with open(self.state_path) as f:
                st = json.load(f)
        except (OSError, ValueError):
            st = {}
        st.setdefault("launched", {})
        st.setdefault("scores", {})
        st.setdefault("gates", {})
        st.setdefault("encF", "pending")
        st.setdefault("early_stop", {})
        for sc in st["scores"].values():
            if sc.get("status") == "running":  # a previous process was killed mid-job: run it again
                sc["status"] = "pending"
        return st

    def save(self) -> None:
        if self.dry:
            return
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1, sort_keys=True)
        os.replace(tmp, self.state_path)

    # --- the volume ------------------------------------------------------------------------------------------

    def refresh(self) -> None:
        status, entries = self.vol.ls("exp")
        self.listing_ok = status == "ok"
        if not self.listing_ok:
            self.once("exp-listing", f"cannot list exp/ ({status}); nothing is launched or decided this cycle")
            return
        self.clear_warn("exp-listing")
        names = {e.split("/")[-1] for e in entries}
        for role, r in self.runs.items():
            run = r["run"]
            self.exists[run] = run in names
            if not self.exists[run]:
                self.markers[run] = set()
                continue
            st, files = self.vol.ls(f"exp/{run}")
            if st == "ok":
                self.markers[run] = {int(m.group(1)) for f in files for m in [MARK.search(f)] if m}
                self.clear_warn("ls-" + run)
            elif st == "absent":
                self.markers[run] = set()
            else:
                self.markers[run] = None
                self.once("ls-" + run, f"cannot list exp/{run}; its markers are unknown this cycle")

    def has(self, role: str, step: int) -> bool:
        return step in (self.markers.get(self.runs[role]["run"]) or ())

    # --- launching -------------------------------------------------------------------------------------------

    def launch_params(self) -> dict:
        c = self.cfg
        man = parse_manifest(os.path.join(c["main_log_dir"], "launch_manifest.txt")) if c.get("main_log_dir") else {}
        p = {k: c.get(k) or man.get(k) or "" for k in ("sha", "mix_dir", "part_weights", "extra_parts")}
        if man.get("sha") and c.get("sha") and man["sha"] != c["sha"] and not man["sha"].startswith(c["sha"]):
            self.once("sha-mismatch", f"ALARM the config sha {c['sha']} differs from the main run's launch {man['sha']}; "
                                      "the branches use the main run's")
        if man.get("sha"):
            p["sha"] = man["sha"]
        p["app"] = man.get("app", "")
        return p

    def try_launch(self, role: str, steps: int, init_from: str = "") -> None:
        """Launch `role` unless its exp/<run> exists or the state says it was launched. Idempotent."""
        run, arm = self.runs[role]["run"], self.runs[role]["arm"]
        rec = self.state["launched"].get(role)
        if rec and rec.get("status") in ("launched", "exists", "uncertain", "failed", "dry"):
            return
        if not self.listing_ok:
            return
        if self.exists.get(run):
            self.log(f"{role}: exp/{run} already exists on the volume; not launching, recording it")
            self.state["launched"][role] = dict(status="exists", ts=self.clock())
            self.save()
            return
        if rec and self.clock() - rec.get("last_try", 0) < RETRY_S:
            return
        p = self.launch_params()
        missing = [k for k in ("sha", "mix_dir") if not p[k]]
        if missing:
            self.once(f"params-{role}", f"{role}: cannot launch yet, no {missing} (set them in the config or log the main "
                                        "run's launch under main_log_dir)")
            return
        argv = ["bash", self.cfg["launcher"], "--run", run, "--arm", arm, "--sha", p["sha"], "--max-steps", str(steps),
                "--mix-dir", p["mix_dir"], "--repo", self.cfg["repo"], "--approvals", self.cfg["approvals"],
                "--log-dir", os.path.join(self.cfg["runs_log_root"], run)]
        if p["part_weights"]:
            argv += ["--part-weights", p["part_weights"]]
        if p["extra_parts"]:
            argv += ["--extra-parts", p["extra_parts"]]
        if init_from:
            argv += ["--init-from", init_from]
        if self.dry:
            self.log(f"DRY-RUN {role}: would run {shlex.join(argv)}")
            self.state["launched"][role] = dict(status="dry", ts=self.clock())
            return
        self.log(f"ACTION launch {role} ({run}, {arm}, {steps} steps, init {init_from or 'stock'}): {shlex.join(argv)}")
        rc, out = self.ex.run(argv, timeout=600)
        now = self.clock()
        tail = " | ".join(out.strip().splitlines()[-2:]) if out else ""
        attempts = (rec or {}).get("attempts", 0) + 1
        if rc == 0:
            self.state["launched"][role] = dict(status="launched", ts=now, rc=0)
            self.log(f"{role}: launched ({tail})")
            self.clear_warn(f"refused-{role}")
        elif rc == 4:
            self.state["launched"][role] = dict(status="exists", ts=now, rc=4)
            self.log(f"{role}: the launcher says exp/{run} exists; recorded, not retried")
        elif rc == 6 or rc is None:
            self.state["launched"][role] = dict(status="uncertain", ts=now, rc=rc)
            self.log(f"ALARM {role}: the launch may have started (exit {rc}: {tail}); NOT retrying, look at "
                     f"{self.cfg['runs_log_root']}/{run} by hand")
        elif rc in (3, 7, 8, 9):
            why = {3: "no approval line yet", 7: "start checkpoint not on the volume", 8: "volume listing failed",
                   9: "caffeinate missing"}[rc]
            self.state["launched"][role] = dict(status="waiting", last_try=now, attempts=(rec or {}).get("attempts", 0),
                                               ts=(rec or {}).get("ts", now), why=why)
            self.once(f"refused-{role}", f"{role}: launcher refused ({why}); will retry every {RETRY_S // 60} min. {tail}")
        else:
            status = "failed" if attempts >= MAX_ATTEMPTS else "waiting"
            self.state["launched"][role] = dict(status=status, last_try=now, attempts=attempts, rc=rc, ts=now)
            self.log(f"{'ALARM ' if status == 'failed' else ''}{role}: launcher exit {rc} (attempt {attempts}); {tail}"
                     + (" -> giving up" if status == "failed" else ""))
        self.save()

    def launch_step(self) -> None:
        c, st = self.cfg, self.state["launched"]
        bstep = c.get("branch_step", 1000)
        init = f"{self.runs['main']['run']}@{bstep}"
        if self.has("main", bstep):
            self.try_launch("brC", c.get("branch_steps", 500), init)
            if self.settled("brC"):
                if self.after_stagger("brC"):
                    self.try_launch("brE", c.get("branch_steps", 500), init)
                flag = c.get("brEr_flag")
                if flag and os.path.exists(flag) and self.settled("brE") and self.after_stagger("brE"):
                    self.try_launch("brEr", c.get("branch_steps", 500), init)
                elif flag and not os.path.exists(flag):
                    self.clear_warn("brEr-flag")
        if self.state["encF"] == "go":
            self.try_launch("encF", c.get("encF_steps", 4000), init)

    def settled(self, role: str) -> bool:
        """The role's launch attempt has an outcome (launched, exists, uncertain, failed) or the run exists."""
        rec = self.state["launched"].get(role)
        return bool(rec and rec.get("status") in ("launched", "exists", "uncertain", "failed", "dry"))

    def after_stagger(self, prev_role: str) -> bool:
        ts = self.state["launched"][prev_role].get("ts", 0)
        return self.clock() - ts >= self.cfg.get("stagger_s", 300)

    # --- scoring ---------------------------------------------------------------------------------------------

    def plan(self):
        """[(priority, role, step)] of every checkpoint that should be scored."""
        c = self.cfg
        items = []
        for role in ("brC", "brE"):
            for s in (250, 500):
                items.append((0, role, s))
        items.append((1, "brEr", 500))
        es = c.get("early_stop", {})
        if es:
            items += [(1, "main", es["step"]), (1, "main", c.get("branch_step", 1000))]
        for role, steps in c.get("score_plan", {}).items():
            items += [(2, role, s) for s in steps]
        seen, out = set(), []
        for it in sorted(items, key=lambda x: (x[0], x[2], x[1])):
            if it[1:] not in seen and it[1] in self.runs:
                seen.add(it[1:])
                out.append(it)
        return out

    def key(self, role: str, step: int) -> str:
        return f"{role}@{step}"

    def out_dir(self, role: str, step: int) -> str:
        return os.path.join(self.cfg["score_root"], f"{self.runs[role]['run']}_step{step}")

    def tool_ready(self, name: str, path: str) -> bool:
        if os.path.exists(path):
            self.clear_warn("missing-" + name)
            return True
        self.once("missing-" + name, f"{name} {path} does not exist yet; waiting for it (checkpoints are queued, not lost)")
        return False

    def fill(self, template, **kw):
        kw.setdefault("python", sys.executable)
        return [str(t).format(**kw) for t in template]

    def score_step(self) -> None:
        c = self.cfg
        # collect finished jobs
        for key, (h, started) in list(self.jobs.items()):
            rc = h.poll()
            timed_out = rc is None and self.clock() - started > c.get("score_timeout_s", 7200)
            if rc is None and not timed_out:
                continue
            if timed_out:
                h.kill()
            sc = self.state["scores"][key]
            del self.jobs[key]
            if rc == 0:
                sc["status"] = "done"
                self.log(f"scored {key} ({int(self.clock() - started)} s) -> {sc['out']}")
            else:
                sc["attempts"] = sc.get("attempts", 0) + 1
                sc["status"] = "failed" if sc["attempts"] >= 2 else "pending"
                self.log(f"{'ALARM ' if sc['status'] == 'failed' else ''}score {key} "
                         f"{'timed out' if timed_out else 'exited ' + str(rc)} (attempt {sc['attempts']})"
                         + (" -> giving up" if sc["status"] == "failed" else ""))
            self.save()
        # queue the new markers, then start what fits
        scorer = c["score_script"]
        for _, role, step in self.plan():
            key = self.key(role, step)
            if self.has(role, step) and key not in self.state["scores"] and not self.skipped(role, step):
                self.state["scores"][key] = dict(status="pending", out=self.out_dir(role, step))
                self.log(f"marker seen: {self.runs[role]['run']} step{step}.pt.done -> score queued")
        pend = [(p, r, s) for p, r, s in self.plan()
                if self.state["scores"].get(self.key(r, s), {}).get("status") == "pending"]
        for _, role, step in pend:
            if len(self.jobs) >= c.get("score_concurrency", 2):
                break
            key = self.key(role, step)
            if key in self.jobs or not self.tool_ready("scorer", scorer):
                continue
            out = self.state["scores"][key]["out"]
            argv = self.fill(c.get("score_argv") or ["{score_script}", "{run}", "{step}", "{out}"], score_script=scorer,
                             run=self.runs[role]["run"], step=step, out=out, role=role)
            if self.dry:
                self.log(f"DRY-RUN would score {key}: {shlex.join(argv)}")
                self.state["scores"][key]["status"] = "dry"
                continue
            os.makedirs(out, exist_ok=True)
            self.log(f"ACTION score {key}: {shlex.join(argv)}")
            try:
                h = self.ex.start(argv, os.path.join(out, "score.log"), cwd=c.get("score_cwd"))
            except OSError as e:
                self.log(f"ALARM cannot start the scorer: {e}")
                continue
            self.jobs[key] = (h, self.clock())
            self.state["scores"][key]["status"] = "running"
            self.save()

    def skipped(self, role: str, step: int) -> bool:
        es = self.state["early_stop"]
        return role == "main" and es.get("action") == "stopped" and step > self.cfg["early_stop"]["step"]

    def done(self, role: str, step: int) -> bool:
        return self.state["scores"].get(self.key(role, step), {}).get("status") == "done"

    # --- gates, encF, early stop -------------------------------------------------------------------------------

    def gate_step(self) -> None:
        c = self.cfg
        if self.state["encF"] != "pending" or not c.get("gate_script"):
            return
        for step in (250, 500):
            k = str(step)
            if k in self.state["gates"] or not (self.done("brC", step) and self.done("brE", step)):
                continue
            if not self.tool_ready("interim gate", c["gate_script"]):
                return
            argv = self.fill(c.get("gate_argv") or ["{python}", "{gate_script}", "--step", "{step}", "--c-dir", "{c_dir}",
                                                    "--e-dir", "{e_dir}"], gate_script=c["gate_script"], step=step, c_dir=self.out_dir("brC", step), e_dir=self.out_dir("brE", step),
                             out=os.path.join(c["score_root"], f"gate_step{step}.txt"))
            if self.dry:
                self.log(f"DRY-RUN would run the gate at {step}: {shlex.join(argv)}")
                continue
            self.log(f"ACTION interim gate at step {step}: {shlex.join(argv)}")
            rc, out = self.ex.run(argv, timeout=900, log_path=os.path.join(c["score_root"], f"gate_step{step}.log"),
                                  cwd=c.get("score_cwd"))
            verdict, signs = parse_gate(out)
            if verdict is None:
                n = self.state["gates"].setdefault("errors", {}).get(k, 0) + 1
                self.state["gates"]["errors"][k] = n
                self.log(f"{'ALARM ' if n >= 3 else ''}gate at {step}: no GO / NO-GO in the output (exit {rc}); "
                         f"{'giving up, decide by hand' if n >= 3 else 'will retry'}: {' | '.join(out.strip().splitlines()[-3:])}")
                if n >= 3:
                    self.state["gates"][k] = dict(verdict="error", signs=None)
                self.save()
                continue
            self.state["gates"][k] = dict(verdict=verdict, signs=signs)
            self.log(f"gate at step {step}: {verdict.upper()} signs={signs} (exit {rc})")
            self.save()
        g = self.state["gates"]
        verdicts = {s: g[s]["verdict"] for s in ("250", "500") if s in g}
        if any(v != "go" for v in verdicts.values()):
            self.state["encF"] = "nogo"
            self.log(f"encF decision: NO (gate verdicts {verdicts}); no full encoder run, diagnose by hand")
        elif len(verdicts) == 2:
            a, b = g["250"]["signs"], g["500"]["signs"]
            if a is None or b is None:
                self.log("note: the gate printed no `signs:` line; sign agreement is taken from the GO verdicts alone")
            if a != b:
                self.state["encF"] = "nogo"
                self.log(f"encF decision: NO (GO at both steps but the signs differ: {a} vs {b})")
            else:
                self.state["encF"] = "go"
                self.log("encF decision: GO at 250 and 500 with sign agreement")
        self.save()

    def early_stop_step(self) -> None:
        es = self.cfg.get("early_stop")
        st = self.state["early_stop"]
        if not es or st.get("action") in ("stopped", "kept"):
            return
        step, ref = es["step"], self.cfg.get("branch_step", 1000)
        if not self.done("main", step):
            return
        read = lambda s: self.read_summary(self.out_dir("main", s))
        now, then = read(step), read(ref)
        v_key, w_key = es.get("v_key", "pooled_v_vs_stock_pp"), es.get("wer_key", "wer_ok")
        verdict, why = early_stop_decision(now.get(v_key), then.get(v_key), now.get(w_key), es.get("v_limit_pp", 3.0))
        if verdict == "undetermined":
            self.once("early-stop-undetermined", f"ALARM early stop at main step {step} cannot be decided: {why} "
                      f"(wanted {v_key} and {w_key} in the summary of main@{step}; reference main@{ref}); decide by hand")
            return
        self.log(f"early-stop rule at main step {step}: {verdict.upper()} ({why})")
        if verdict == "keep":
            st["action"] = "kept"
            self.save()
            return
        app = self.launch_params().get("app")
        if es.get("action", "stop") != "stop" or not app:
            self.log(f"ALARM early stop says STOP main; {'action is log-only' if app else 'no app id in its launch manifest'}"
                     f" -> stop it by hand{': modal app stop ' + app if app else ''}")
            st["action"] = "stopped"
            self.save()
            return
        if self.dry:
            self.log(f"DRY-RUN would stop app {app}")
            return
        self.log(f"ACTION stop the main run's app {app}")
        rc, out = self.ex.run(["modal", "app", "stop", app], timeout=180)
        self.log(f"modal app stop exit {rc}: {out.strip()[-200:]}" + ("" if rc == 0 else "  ALARM stop it by hand"))
        if rc == 0:
            st["action"] = "stopped"
        self.save()

    @staticmethod
    def read_summary(out_dir: str) -> dict:
        try:
            with open(os.path.join(out_dir, "summary.json")) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    # --- the loop --------------------------------------------------------------------------------------------

    def finished(self) -> bool:
        if not (self.settled("brC") and self.settled("brE")) or self.state["encF"] == "pending":
            return False
        if self.state["encF"] == "go" and not self.settled("encF"):
            return False
        for _, role, step in self.plan():
            if role == "brEr" and not self.settled("brEr"):
                continue  # only scored when the flag launched it
            if role == "encF" and self.state["encF"] != "go":
                continue
            if self.skipped(role, step):
                continue
            if self.state["scores"].get(self.key(role, step), {}).get("status") not in ("done", "failed"):
                return False
        return True

    def beat(self) -> None:
        if self.clock() - self.last_beat < 1800:
            return
        self.last_beat = self.clock()
        launched = {r: v["status"] for r, v in self.state["launched"].items()}
        scores = {}
        for k, v in self.state["scores"].items():
            scores.setdefault(v["status"], []).append(k)
        self.log(f"heartbeat: launched {launched}; encF {self.state['encF']}; scores {scores}; early stop "
                 f"{self.state['early_stop'].get('action')}")

    def cycle(self) -> None:
        self.refresh()
        if self.listing_ok:
            self.launch_step()
        self.score_step()
        self.gate_step()
        self.early_stop_step()
        if self.listing_ok:
            self.launch_step()  # a gate that just said GO launches encF in the same cycle
        self.beat()

    def loop(self, once=False, sleep=time.sleep) -> int:
        poll, end = self.cfg.get("poll_s", 120), self.clock() + self.cfg.get("max_hours", 14) * 3600
        self.log(f"trigger started (dry-run={self.dry}); runs: " + ", ".join(f"{r}={v['run']}" for r, v in self.runs.items()))
        while True:
            try:
                self.cycle()
            except Exception as e:  # one bad cycle must not end the night
                self.log(f"ALARM cycle error {type(e).__name__}: {e}")
            if once:
                return 0
            if self.finished():
                self.log("all planned launches and scores are settled; exiting")
                return 0
            if self.clock() > end:
                self.log("ALARM max_hours reached with work still pending; exiting (the state file keeps the progress)")
                return 1
            sleep(poll)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--once", action="store_true", help="one cycle, then exit")
    ap.add_argument("--dry-run", action="store_true", help="log what would be launched / scored / stopped; do nothing")
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = json.load(f)
    for k in ("runs", "launcher", "repo", "approvals", "runs_log_root", "score_script", "score_root", "state_file",
              "log_file", "volume"):
        if k not in cfg:
            print(f"config lacks '{k}'", file=sys.stderr)
            return 2
    for role in cfg["runs"]:
        if role not in ROLES or not {"run", "arm"} <= set(cfg["runs"][role]):
            print(f"config runs.{role}: roles are {ROLES}, each with run and arm", file=sys.stderr)
            return 2
    for k in ("score_root", "runs_log_root"):
        os.makedirs(cfg[k], exist_ok=True)
    lock = open(cfg["state_file"] + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another trigger holds the lock; exiting", file=sys.stderr)
        return 3
    t = Trigger(cfg, Volume(cfg["volume"]), Exec(), dry_run=a.dry_run)
    return t.loop(once=a.once)


if __name__ == "__main__":
    sys.exit(main())
