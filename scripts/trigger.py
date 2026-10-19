"""Local orchestrator for one night's run set: launches the follow-on runs and scores checkpoints as their markers appear.

    python3 trigger.py --config <config.json> [--once] [--dry-run]        (normally through trigger.sh, under caffeinate -i)

It never launches a main run: those are started by hand through launch_run.sh. It watches the volume for
exp/<main>/step<branch_step>.pt.done and then, through launch_run.sh only,
  * launches the two branch runs (decoder-only control `brC`, then the encoder branch `brE`, `stagger_s` apart);
  * with a `brCs2` role in the config, launches the control branch again on another sampler seed (`seed` of that role, same
    rows, another batch order) `stagger_s` after brE: a noise-floor read, scored like brC and not part of the gate;
  * launches the real-only encoder branch `brEr` only if the flag file named in the config exists;
  * scores every branch checkpoint (step 250 and 500) and runs the interim gate on E vs C at 250, at 500 and over both
    (sign agreement); when all three say GO it launches the full encoder continuation `encF`;
  * scores the planned checkpoints of the main, half-lr and encF runs as each `step<N>.pt.done` appears;
  * applies the early-stop rule to each run of `early_stop.roles` (default the main run) at `early_stop.step` (stops its app);
  * with an `anc1` role (a run launched by hand), scores it at step 500 together with the main run's step 500, and runs each
    configured `compare` command (a, b = "<role>@<step>") once both are scored.
Every action goes to the log file; the state file makes a restart harmless: a run whose exp/<run> exists on the volume (or
whose launch the state records) is never launched again, a listing that fails is never read as "absent", and a scoring
job already done is not repeated. The approvals file is only read by launch_run.sh: this script never writes it, so a run
without its approval line is refused by the launcher and retried (logged once) until the line appears.

Contracts with the scoring tools (argv templates `score_argv` / `gate_argv` in the config; {score_script} {gate_script}
{run} {step} {out} {mix} {e_run} {c_run} {steps} {python} are filled in; the defaults are `<scorer> <run> <step> <out>` and
`python <gate> <E run> <C run> --steps <steps>`):
  * the scorer is run once per checkpoint; exit 0 = scored. Its results must be where `score_dir_template` says (it is
    also the directory the gate reads);
  * the gate is run over step 250, over step 500 and over both (its sign-agreement read): exit 0 = GO, 1 = NO-GO, anything
    else = no verdict (retried, then left to hand). The encoder run launches only when all three say GO;
  * the early-stop rule reads `pooled_v_vs_stock_pp` (number) and `wer_ok` (bool) of the run's early_stop step and of its
    reference checkpoint (early_stop.ref[role] = "<role>@<step>", default its own branch_step checkpoint), from the JSON object printed by `early_stop.summary_cmd` (argv template, {scores_dir}) or,
    without one, from <scores dir>/summary.json (key names are config values).
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
SWITCHES = ("--anchor-init", "--detach")  # flags of the launcher's training command that take no value
HAND_LAUNCHED = ("anc1", "cap", "enfree2")  # runs started by hand: scored when they exist, not waited for otherwise
GATE_KEYS = ("250", "500", "250,500")  # the step sets the interim gate is read over
ROLES = ("main", "lr05", "cap", "enfree2", "brC", "brCs2", "brE", "brEr", "encF", "anc1")  # brCs2: brC again on another sampler seed; anc1: launched by hand


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
    m = re.search(r"^command (.*?)\n(?=approval |module sha256|app |launched |# resumed|\Z)", text, re.M | re.S)
    if m:
        try:
            toks = shlex.split(m.group(1))
        except ValueError:
            toks = []
        start = toks.index("--run") if "--run" in toks else len(toks)
        i, ok = start, start < len(toks)
        # every flag of the logged command takes exactly one value (except the switches): a value that was split (a space
        # or a line break inside it) breaks that pattern, and then nothing taken from this line can be trusted
        while ok and i < len(toks):
            if not toks[i].startswith("--"):
                ok = False
            elif toks[i] in SWITCHES:
                i += 1
            elif i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                i += 2
            else:
                ok = False
        if not ok:
            return {k: v for k, v in out.items() if k in ("sha", "app")} | {"command_untrusted": True}
        for flag, key in (("--mix-dir", "mix_dir"), ("--part-weights", "part_weights"), ("--extra-parts", "extra_parts")):
            if flag in toks and toks.index(flag) + 1 < len(toks):
                out[key] = toks[toks.index(flag) + 1]
    return out


def parse_gate(output: str, rc=None):
    """"go" | "nogo" | None. The gate's exit code decides (0 GO, 1 NO-GO; anything else, e.g. 2 = cannot decide, is no
    verdict unless the output still holds an explicit GO / NO-GO line)."""
    if rc == 0:
        return "go"
    if rc == 1:
        return "nogo"
    if re.search(r"==>\s*NO[-_ ]?GO\b", output):
        return "nogo"
    if re.search(r"==>\s*GO\b", output):
        return "go"
    return None


def early_stop_decision(v_now, v_ref, wer_ok, limit_pp: float):
    """("stop"|"keep"|"undetermined", reason). Stop when WER is broken, or the pooled V vs stock is above limit_pp and
    worse than at the reference checkpoint. A missing number is "undetermined", never "keep"."""
    if wer_ok is False:
        return "stop", "WER broken"
    if v_now is None:
        return "undetermined", "no pooled V in the summary"
    if v_now > limit_pp:
        if v_ref is None:
            return "undetermined", f"V {v_now:+.1f} pp > {limit_pp:+.1f} but the reference checkpoint has no V"
        if v_now > v_ref:
            return "stop", f"pooled V {v_now:+.1f} pp > {limit_pp:+.1f} and worse than {v_ref:+.1f} at the reference"
    if wer_ok is None:
        return "undetermined", "no WER verdict in the summary"
    return "keep", f"pooled V {v_now:+.1f} pp (limit {limit_pp:+.1f}, reference {v_ref}), WER ok"


def running_apps(raw: str, exclude=()) -> list:
    """The apps of a `modal app list --json` listing that hold a running task and are not in `exclude`; None if raw is no list."""
    try:
        apps = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(apps, list):
        return None
    out = []
    for a in apps:
        try:
            tasks = int(str(a.get("tasks", "0")) or 0)
        except ValueError:
            tasks = 0
        if tasks > 0 and not str(a.get("state", "")).startswith("stopped") and a.get("app_id") not in exclude:
            out.append(dict(app_id=a.get("app_id"), description=a.get("description", ""), state=a.get("state", ""),
                            tasks=tasks, created_at=a.get("created_at", "")))
    return out


def format_apps(apps: list) -> str:
    return "\n".join(f"  {a['app_id']}  {a['description']:<28s} tasks={a['tasks']}  {a['state']}  started {a['created_at']}"
                     for a in apps)


def own_app_ids(cfg: dict) -> set:
    """App ids of the runs launched through the launcher (their launch manifests), which are not 'other' readers."""
    ids, dirs = set(), []
    root = cfg.get("runs_log_root")
    if root and os.path.isdir(root):
        dirs += [os.path.join(root, d) for d in os.listdir(root)]
    if cfg.get("main_log_dir"):
        dirs.append(cfg["main_log_dir"])
    for d in dirs:
        ids.add(parse_manifest(os.path.join(d, "launch_manifest.txt")).get("app"))
    ids.discard(None)
    return ids


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

    def quiet_ok(self, role: str) -> bool:
        """With `quiet_max_other` in the config, wait (up to `quiet_wait_s`, then go on with an alarm) while more than that
        many apps besides this night's own runs hold a running task: the launched run stages its audio from the same volume
        and a busy volume leaves its GPU idle."""
        limit = self.cfg.get("quiet_max_other")
        if limit is None:
            return True
        rc, out = self.ex.run(["modal", "app", "list", "--json", "--limit", "200"], timeout=120)
        apps = running_apps(out, own_app_ids(self.cfg)) if rc == 0 else None
        waited = self.state.setdefault("quiet_since", {})
        if apps is not None and len(apps) <= limit:
            self.clear_warn(f"quiet-{role}")
            return True
        since = waited.setdefault(role, self.clock())
        late = self.clock() - since > self.cfg.get("quiet_wait_s", 2700)
        what = "the app list failed" if apps is None else f"{len(apps)} other apps are running (limit {limit}):\n{format_apps(apps)}"
        if late:
            self.log(f"ALARM {role}: the volume was not quiet for {self.cfg.get('quiet_wait_s', 2700) // 60} min; launching "
                     f"anyway. {what}")
            return True
        self.once(f"quiet-{role}", f"{role}: waiting for a quiet volume before the launch: {what}")
        self.save()
        return False

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
        if man.get("command_untrusted") and not (c.get("mix_dir") and "part_weights" in c and "extra_parts" in c):
            self.once("untrusted-cmd", "ALARM the main run's logged launch command could not be read in full; the branches "
                                       "would risk a different mix. Set mix_dir, part_weights and extra_parts in the config.")
            p["mix_dir"] = ""
        return p

    def try_launch(self, role: str, steps: int, init_from: str = "", seed: int = 0) -> None:
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
        if not self.dry and not self.quiet_ok(role):
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
        if seed:
            argv += ["--seed", str(seed)]
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
                last = "brE"  # each further branch starts stagger_s after the one before it
                if "brCs2" in self.runs and self.settled("brE") and self.after_stagger("brE"):
                    self.try_launch("brCs2", c.get("branch_steps", 500), init, self.runs["brCs2"].get("seed", 0))
                if "brCs2" in self.runs:
                    last = "brCs2"
                flag = c.get("brEr_flag")
                if flag and os.path.exists(flag) and self.settled(last) and self.after_stagger(last):
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
        items.append((1, "brCs2", 500))
        if "anc1" in self.runs and self.exists.get(self.runs["anc1"]["run"]):  # the anchor test and its matched control (main@500)
            items += [(1, "anc1", 500), (1, "main", 500)]
        es = c.get("early_stop", {})
        if es:
            for role in es.get("roles", ["main"]):
                ref_role, ref_step = self.es_ref(role)
                items += [(1, role, es["step"]), (1, ref_role, ref_step)]
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
        tpl = self.cfg.get("score_dir_template") or "{score_root}/{run}_step{step}"
        return tpl.format(score_root=self.cfg["score_root"], run=self.runs[role]["run"], step=step, mix=self.mix_of(role))

    def mix_of(self, role: str) -> str:
        """The mix a run's held-out read uses ({mix} in the templates): the role's own `mix`, else the config's."""
        return self.runs.get(role, {}).get("mix") or self.cfg.get("mix", "")

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
                             mix=self.mix_of(role),
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

    def es_ref(self, role: str):
        """(role, step) of the checkpoint the early-stop rule compares `role`'s step against: early_stop.ref[role] =
        "<role>@<step>", default its own checkpoint at branch_step."""
        ref = self.cfg.get("early_stop", {}).get("ref", {}).get(role)
        if ref:
            r, st = ref.split("@")
            return r, int(st)
        return role, self.cfg.get("branch_step", 1000)

    def skipped(self, role: str, step: int) -> bool:
        es = self.state["early_stop"].get(role, {})
        return es.get("action") == "stopped" and step > self.cfg["early_stop"]["step"]

    def done(self, role: str, step: int) -> bool:
        return self.state["scores"].get(self.key(role, step), {}).get("status") == "done"

    def compare_step(self) -> None:
        """Run each configured `compare` command (a: "<role>@<step>", b: ..., cmd argv with {a_dir} {b_dir}, script) once both
        checkpoints are scored; its output goes to <score_root>/<name>.txt and the first lines to the log."""
        c = self.cfg
        done = self.state.setdefault("compares", {})
        for cmp_ in c.get("compare", []):
            name = cmp_["name"]
            rec = done.get(name, {})
            if rec.get("status") in ("done", "failed") or self.clock() - rec.get("last", 0) < RETRY_S:
                continue
            (ra, sa), (rb, sb) = (x.split("@") for x in (cmp_["a"], cmp_["b"]))
            if ra not in self.runs or rb not in self.runs or not (self.done(ra, int(sa)) and self.done(rb, int(sb))):
                continue
            if cmp_.get("script") and not self.tool_ready("compare " + name, cmp_["script"]):
                continue
            argv = self.fill(cmp_["cmd"], a_dir=self.out_dir(ra, int(sa)), b_dir=self.out_dir(rb, int(sb)))
            out_path = os.path.join(c["score_root"], f"{name}.txt")
            if self.dry:
                self.log(f"DRY-RUN would compare {name}: {shlex.join(argv)}")
                continue
            self.log(f"ACTION compare {name}: {shlex.join(argv)}")
            rc, out = self.ex.run(argv, timeout=600, cwd=c.get("score_cwd"))
            n = rec.get("n", 0) + 1
            if rc == 0:
                with open(out_path, "w") as f:
                    f.write(out)
                done[name] = dict(status="done")
                self.log(f"compare {name} -> {out_path}\n" + out.rstrip())
            else:
                done[name] = dict(status="failed" if n >= 2 else "retry", n=n, last=self.clock())
                self.log(f"{'ALARM ' if n >= 2 else ''}compare {name} exit {rc}: {' | '.join(out.strip().splitlines()[-2:])}")
            self.save()

    # --- gates, encF, early stop -------------------------------------------------------------------------------

    def gate_step(self) -> None:
        """The interim gate, read at step 250, at step 500 and over both (its sign-agreement read); encF needs all three GO."""
        c = self.cfg
        if self.state["encF"] != "pending" or not c.get("gate_script"):
            return
        g = self.state["gates"]
        for key in GATE_KEYS:
            steps = [int(x) for x in key.split(",")]
            if key in g or not all(self.done(r, s) for s in steps for r in ("brC", "brE")):
                continue
            err = g.setdefault("errors", {}).get(key, {})
            if self.clock() - err.get("last", 0) < RETRY_S:
                continue
            if not self.tool_ready("interim gate", c["gate_script"]):
                return
            argv = self.fill(c.get("gate_argv") or ["{python}", "{gate_script}", "{e_run}", "{c_run}", "--steps", "{steps}"],
                             gate_script=c["gate_script"], mix=self.mix_of("brE"), e_run=self.runs["brE"]["run"], c_run=self.runs["brC"]["run"],
                             steps=key, out=os.path.join(c["score_root"], f"gate_{key.replace(',', '_')}.json"))
            if self.dry:
                self.log(f"DRY-RUN would run the gate over steps {key}: {shlex.join(argv)}")
                continue
            self.log(f"ACTION interim gate over steps {key}: {shlex.join(argv)}")
            rc, out = self.ex.run(argv, timeout=1800, cwd=c.get("score_cwd"),
                                  log_path=os.path.join(c["score_root"], f"gate_{key.replace(',', '_')}.log"))
            verdict = parse_gate(out, rc)
            if verdict is None:
                n = err.get("n", 0) + 1
                g["errors"][key] = dict(n=n, last=self.clock())
                self.log(f"{'ALARM ' if n >= 3 else ''}gate over {key}: no verdict (exit {rc}); "
                         f"{'giving up, decide by hand' if n >= 3 else 'will retry'}: {' | '.join(out.strip().splitlines()[-3:])}")
                if n >= 3:
                    g[key] = dict(verdict="error")
                self.save()
                break  # the gate cannot decide yet: do not try the other step sets on top of it
            else:
                g[key] = dict(verdict=verdict)
                self.log(f"gate over steps {key}: {verdict.upper()} (exit {rc}): {' | '.join(out.strip().splitlines()[-1:])}")
            self.save()
        verdicts = {k: g[k]["verdict"] for k in GATE_KEYS if k in g}
        if "error" in verdicts.values():
            self.state["encF"] = "manual"
            self.log(f"ALARM encF decision left to hand: the gate gave no verdict ({verdicts})")
        elif any(v == "nogo" for v in verdicts.values()):
            self.state["encF"] = "nogo"
            self.log(f"encF decision: NO (gate verdicts {verdicts}); no full encoder run, diagnose by hand")
        elif len(verdicts) == len(GATE_KEYS):
            self.state["encF"] = "go"
            self.log("encF decision: GO at 250, at 500 and over both (sign agreement)")
        self.save()

    def early_stop_step(self) -> None:
        es = self.cfg.get("early_stop")
        if not es:
            return
        for role in es.get("roles", ["main"]):
            if role in self.runs:
                self.early_stop_role(role, es)

    def early_stop_role(self, role: str, es: dict) -> None:
        st = self.state["early_stop"].setdefault(role, {})
        if st.get("action") in ("stopped", "kept"):
            return
        step = es["step"]
        ref_role, ref = self.es_ref(role)
        if not self.done(role, step):
            return
        now, then = self.read_summary(role, step), self.read_summary(ref_role, ref)
        v_key, w_key = es.get("v_key", "pooled_v_vs_stock_pp"), es.get("wer_key", "wer_ok")
        verdict, why = early_stop_decision(now.get(v_key), then.get(v_key), now.get(w_key), es.get("v_limit_pp", 3.0))
        name = self.runs[role]["run"]
        if verdict == "undetermined":
            self.once(f"early-stop-undetermined-{role}", f"ALARM early stop at {name} step {step} cannot be decided: {why} "
                      f"(wanted {v_key} and {w_key} in the summary of {role}@{step}; reference {ref_role}@{ref}); decide by hand")
            return
        self.log(f"early-stop rule at {name} step {step} (reference {ref_role}@{ref}): {verdict.upper()} ({why})")
        if verdict == "keep":
            st["action"] = "kept"
            self.save()
            return
        app = self.app_of(role)
        if es.get("action", "stop") != "stop" or not app:
            self.log(f"ALARM early stop says STOP {name}; {'action is log-only' if app else 'no app id in its launch manifest'}"
                     f" -> stop it by hand{': modal app stop --yes ' + app if app else ''}")
            st["action"] = "stopped"
            self.save()
            return
        if self.dry:
            self.log(f"DRY-RUN would stop app {app}")
            return
        self.log(f"ACTION stop {name}'s app {app}")
        rc, out = self.ex.run(["modal", "app", "stop", "--yes", app], timeout=180)  # --yes: without it the CLI asks and aborts
        self.log(f"modal app stop exit {rc}: {out.strip()[-200:]}" + ("" if rc == 0 else "  ALARM stop it by hand"))
        if rc == 0:
            st["action"] = "stopped"
        self.save()

    def app_of(self, role: str) -> str:
        """The Modal app id of a run launched through the launcher: its launch manifest under runs_log_root/<run> (the main
        run's under main_log_dir as well)."""
        run = self.runs[role]["run"]
        paths = [os.path.join(self.cfg["runs_log_root"], run, "launch_manifest.txt")]
        if role == "main" and self.cfg.get("main_log_dir"):
            paths.append(os.path.join(self.cfg["main_log_dir"], "launch_manifest.txt"))
        for p in paths:
            app = parse_manifest(p).get("app")
            if app:
                return app
        return ""

    def read_summary(self, role: str, step: int) -> dict:
        """{pooled V vs stock, WER verdict} of a scored checkpoint: from the early-stop `summary_cmd` (argv template with
        {scores_dir}; prints one JSON object) or, without one, <scores dir>/summary.json."""
        d = self.out_dir(role, step)
        cmd = self.cfg.get("early_stop", {}).get("summary_cmd")
        try:
            if cmd:
                rc, out = self.ex.run(self.fill(cmd, scores_dir=d, run=self.runs[role]["run"], step=step), timeout=300,
                                      cwd=self.cfg.get("score_cwd"))
                doc = json.loads(out) if rc == 0 else {}
            else:
                with open(os.path.join(d, "summary.json")) as f:
                    doc = json.load(f)
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}

    # --- the loop --------------------------------------------------------------------------------------------

    def finished(self) -> bool:
        if not (self.settled("brC") and self.settled("brE")) or self.state["encF"] == "pending":
            return False
        if self.state["encF"] == "go" and not self.settled("encF"):
            return False
        for _, role, step in self.plan():
            if role in ("brEr", "brCs2") and not self.settled(role):
                continue  # only scored when launched
            if role in HAND_LAUNCHED and not self.exists.get(self.runs[role]["run"]):
                continue  # launched by hand, or not at all
            if role == "encF" and self.state["encF"] != "go":
                continue
            if self.skipped(role, step):
                continue
            if self.state["scores"].get(self.key(role, step), {}).get("status") not in ("done", "failed"):
                return False
        for cmp_ in self.cfg.get("compare", []):
            ra, rb = (x.split("@")[0] for x in (cmp_["a"], cmp_["b"]))
            if any(r in HAND_LAUNCHED and not self.exists.get(self.runs[r]["run"]) for r in (ra, rb)):
                continue
            if self.state.get("compares", {}).get(cmp_["name"], {}).get("status") not in ("done", "failed"):
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
                 f"{ {r: v.get('action') for r, v in self.state['early_stop'].items()} }")

    def cycle(self) -> None:
        self.refresh()
        if self.listing_ok:
            self.launch_step()
        self.score_step()
        self.compare_step()
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


def quiet_check(cfg: dict) -> int:
    """Print who else is running on the volume; exit 0 when that is no more than quiet_max_other, 1 otherwise, 2 if unknown."""
    try:
        r = subprocess.run(["modal", "app", "list", "--json", "--limit", "200"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        r = None
    apps = running_apps(r.stdout, own_app_ids(cfg)) if r is not None and r.returncode == 0 else None
    if apps is None:
        print("cannot list the running apps (network or CLI failure): the volume state is UNKNOWN")
        return 2
    limit = cfg.get("quiet_max_other", 0)
    print(f"{len(apps)} other app(s) holding a running task (this night's own runs excluded; limit {limit}):")
    print(format_apps(apps) or "  none")
    print("QUIET" if len(apps) <= limit else "NOT QUIET: pause the readers above (or wait) before launching; "
          "a busy volume stages at a tenth of its speed and leaves the GPU idle")
    return 0 if len(apps) <= limit else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--once", action="store_true", help="one cycle, then exit")
    ap.add_argument("--config")
    ap.add_argument("--quiet-check", action="store_true", help="print the apps holding a running task and exit 1 if "
                    "more than quiet_max_other (config, default 0) are not this night's own runs")
    ap.add_argument("--dry-run", action="store_true", help="log what would be launched / scored / stopped; do nothing")
    a = ap.parse_args(argv)
    cfg = {}
    if a.config:
        with open(a.config) as f:
            cfg = json.load(f)
    if a.quiet_check:
        return quiet_check(cfg)
    if not a.config:
        print("--config is required", file=sys.stderr)
        return 2
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
