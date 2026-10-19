#!/bin/bash
# Launch a training run of the current recipe from a clean export of one commit, behind three locks.
#
#   scripts/launch_run.sh --run <name> --arm <arm> --sha <commit> --max-steps <N> --mix-dir <dir on the volume> \
#       --repo <git checkout> --approvals <file> --log-dir <dir> [--part-weights "<path>=<n>,..."] \
#       [--extra-parts "<path>:<tag>:<n>,..."] [--en-free-repeat <n>] [--langs <l1,l2,...>] [--volume <name>] [--init-from <run>@<step>]
#       [--resume] [--dry-run] [--no-watch]
#
# Locks (a refusal exits non-zero and starts nothing):
#   1. the exact run name must be a line of the approvals file (exit 3);
#   2. exp/<run> must not exist on the volume (exit 4); with --resume it must exist, and the commit must be the one the
#      first launch logged in --log-dir (exit 5). The check reads the volume's exp/ listing: a failed listing (network,
#      CLI) refuses the launch (exit 8); it is never read as "absent";
#   3. the arm must exist in the commit, max-steps must be a multiple of the checkpoint interval (250), the run name
#      must be a plain name (exit 2).
#   4. --init-from (default ultra = a stock start): a branch arm (R10-br*, R10-encF) needs --init-from <other run>@<step> whose
#      step<N>.pt exists on the volume; every other arm needs the stock start (exit 2; exit 7 when the file is missing).
# The run is launched from `git archive <sha>` of --repo (never the working tree). The commit and the sha256 of
# modal_app.py and of every module the training image mounts go to <log-dir>/launch_manifest.txt before anything starts.
# The training function is retrain3_80_main (A100 80 GB, 6 h cap per attempt) with the standard recipe flags. Unless
# --no-watch, run_watch.py starts beside it, under `caffeinate -i` (a refusal, exit 9, if caffeinate is missing), and stops
# the app on a stall (no first marker in 75 min, none for 40 min; time with no running task does not count) or when the
# task time passes the spend tripwire (steps x s/step x 1.5 + 45 min; 4.6 h for 5,000 steps), and logs a speed alarm above
# 1.5 x the expected s/step, and stops the app when the audio staging runs below 300 files/s after 10 min of task time (a
# volume busy with other readers; the GPU would sit idle). Run `scripts/trigger.sh quiet` before a launch to see who else reads
# the volume. Paths come from the arguments only.
set -u
SAVE_EVERY=250
VOLUME=parakeet-spoken-form
RUN=""; ARM=""; SHA=""; STEPS=""; MIXDIR=""; REPO=""; APPROVALS=""; LOG=""; PW=""; XP=""; LANGS=""; EFR=""
INIT=ultra; RESUME=0; DRY=0; WATCH=1
die() { echo "REFUSED: $2" >&2; exit "$1"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2;; --arm) ARM=$2; shift 2;; --sha) SHA=$2; shift 2;; --max-steps) STEPS=$2; shift 2;;
    --mix-dir) MIXDIR=$2; shift 2;; --repo) REPO=$2; shift 2;; --approvals) APPROVALS=$2; shift 2;;
    --log-dir) LOG=$2; shift 2;; --part-weights) PW=$2; shift 2;; --extra-parts) XP=$2; shift 2;;
    --langs) LANGS=$2; shift 2;; --en-free-repeat) EFR=$2; shift 2;; --volume) VOLUME=$2; shift 2;; --init-from) INIT=$2; shift 2;;
    --resume) RESUME=1; shift;; --dry-run) DRY=1; shift;; --no-watch) WATCH=0; shift;;
    *) die 2 "unknown argument $1";;
  esac
done
for v in RUN ARM SHA STEPS MIXDIR REPO APPROVALS LOG; do [ -n "${!v}" ] || die 2 "missing --$(echo $v | tr 'A-Z_' 'a-z-') (see the header of this script)"; done
[[ $RUN =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die 2 "run name '$RUN' is not a plain name"
[[ $STEPS =~ ^[0-9]+$ ]] && [ "$STEPS" -gt 0 ] && [ $((STEPS % SAVE_EVERY)) -eq 0 ] || die 2 "--max-steps $STEPS is not a multiple of $SAVE_EVERY"

[[ $INIT == ultra || $INIT =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*@[0-9]+$ ]] || die 2 "--init-from '$INIT' is not ultra or <run>@<step>"
if [[ $ARM == R10-br* || $ARM == R10-encF ]]; then
  [ "$INIT" != ultra ] && [ "${INIT%@*}" != "$RUN" ] || die 2 "branch arm $ARM needs --init-from <other run>@<step> (the main run's checkpoint)"
else
  [ "$INIT" = ultra ] || die 2 "--init-from $INIT is for the branch arms (R10-br*, R10-encF); arm $ARM starts from stock"
fi

[ $WATCH -eq 0 ] || command -v caffeinate >/dev/null 2>&1 || die 9 "caffeinate not found: the watchdog must run under caffeinate -i (or pass --no-watch)"
EXPORT=""
die() { echo "REFUSED: $2" >&2; [ -z "$EXPORT" ] || rm -rf "$EXPORT"; exit "$1"; }

# lock 1: an explicit approval for THIS run name (one exact line)
[ -f "$APPROVALS" ] && grep -qxF -- "$RUN" "$APPROVALS" || die 3 "run '$RUN' has no approval line in $APPROVALS"

# lock 2: a new run name (or, with --resume, the same run and the same commit)
FULL=$(git -C "$REPO" rev-parse --verify --quiet "$SHA^{commit}") || die 2 "commit $SHA not found in $REPO"
# the listing must SUCCEED to say anything: "absent" is only "the listing worked and does not name it"
LISTING=$(modal volume ls "$VOLUME" exp 2>/dev/null) || die 8 "cannot list exp/ on $VOLUME (network or CLI failure); not launching on an unknown state"
if printf '%s\n' "$LISTING" | grep -qxF -- "exp/$RUN"; then
  [ $RESUME -eq 1 ] || die 4 "exp/$RUN already exists on $VOLUME (a launch would silently resume it); use a new run name, or --resume on purpose"
else
  [ $RESUME -eq 0 ] || die 4 "--resume: exp/$RUN does not exist on $VOLUME"
fi
if [ $RESUME -eq 1 ]; then
  FIRST=$(sed -n 's/^commit //p' "$LOG/launch_manifest.txt" 2>/dev/null | head -1)
  [ "$FIRST" = "$FULL" ] || die 5 "--resume must use the commit of the first launch (${FIRST:-none logged in $LOG}), not $FULL"
fi

# a branch's start checkpoint must exist on the volume
if [ "$INIT" != ultra ]; then
  INITDIR=$(modal volume ls "$VOLUME" "exp/${INIT%@*}" 2>&1) || {
    printf '%s' "$INITDIR" | grep -q "does not exist" && die 7 "exp/${INIT%@*} does not exist on $VOLUME"
    die 8 "cannot list exp/${INIT%@*} on $VOLUME (network or CLI failure)"
  }
  printf '%s\n' "$INITDIR" | grep -qxF -- "exp/${INIT%@*}/step${INIT#*@}.pt" || die 7 "exp/${INIT%@*}/step${INIT#*@}.pt does not exist on $VOLUME"
fi

# lock 3: the arm exists in the commit
mkdir -p "$LOG" || die 2 "cannot create $LOG"
EXPORT=$(mktemp -d "$LOG/export.XXXXXX") || die 2 "cannot create an export directory in $LOG"
git -C "$REPO" archive "$FULL" | tar -x -C "$EXPORT" || die 2 "git archive failed"
PIPE=$EXPORT/pipeline
python3 "$PIPE/run_manifest.py" "$PIPE" arm "$ARM" || die 2 "arm '$ARM' is not defined in modal_app.py at $FULL"
[ -n "$LANGS" ] || LANGS=$(python3 "$PIPE/run_manifest.py" "$PIPE" langs) || die 2 "cannot read the language list"

CMD=(modal run --detach modal_app.py::retrain3_80_main --run "$RUN" --arm "$ARM" --mix v4 --mix-dir "$MIXDIR"
     --langs "$LANGS" --init-from "$INIT" --anchor-init --dur-weight 0 --warmup 20 --max-steps "$STEPS"
     --save-every $SAVE_EVERY --r3-sub-batch 1)
# --en-free-repeat n: the arm lists the English number-free parts twice; the build may have used another repeat to land the
# English number share in 0.35-0.40, so the weights set the TOTAL count for the three parts (the part-weights string logged below)
if [ -n "$EFR" ]; then
  [[ $EFR =~ ^[0-9]+$ ]] && [ "$EFR" -ge 1 ] || die 2 "--en-free-repeat $EFR is not a count"
  PW="${PW:+$PW,}replay/icsi_train.jsonl=$EFR,mix3b/d1p_v.jsonl=$EFR,mix3b/d10p_v.jsonl=$EFR"
fi
[ -z "$PW" ] || CMD+=(--part-weights "$PW")
[ -z "$XP" ] || CMD+=(--extra-parts "$XP")
HASHES=$(python3 "$PIPE/run_manifest.py" "$PIPE" hashes) || die 2 "cannot hash the mounted modules"
if [ $RESUME -eq 1 ]; then
  HEADER="# resumed launch at $(date -u +%FT%TZ)"
else
  HEADER="run $RUN
commit $FULL
arm $ARM
launched $(date -u +%FT%TZ)
command ${CMD[*]}
approval $(grep -xF -- "$RUN" "$APPROVALS")
module sha256 (modal_app.py + the modules the training image mounts):
$HASHES"
fi
echo "$HEADER"
if [ $DRY -eq 1 ]; then rm -rf "$EXPORT"; echo "DRY RUN: every check passed, nothing launched"; exit 0; fi
echo "export $EXPORT"
echo "$HEADER" >> "$LOG/launch_manifest.txt"

(cd "$PIPE" && nohup "${CMD[@]}" > "$LOG/train.log" 2>&1 &)
APP=""
for _ in $(seq 60); do
  APP=$(grep -oE "ap-[A-Za-z0-9]+" "$LOG/train.log" 2>/dev/null | head -1)
  [ -n "$APP" ] && break
  sleep 2
done
[ -n "$APP" ] || die 6 "launched, but no app id in $LOG/train.log after 2 min; find and watch it by hand"
echo "app $APP" | tee -a "$LOG/launch_manifest.txt"
if [ $WATCH -eq 1 ]; then
  EXPECT=1.83; [[ $ARM == R10-brE* || $ARM == R10-encF ]] && EXPECT=1.9  # s/step: decoder-only arms, arms with the top encoder blocks
  nohup caffeinate -i python3 "$PIPE/run_watch.py" --app "$APP" --run "$RUN" --volume "$VOLUME" --log-dir "$LOG/watch" \
    --steps "$STEPS" --expected-s-per-step "$EXPECT" > "$LOG/watch.log" 2>&1 &
  echo "watchdog pid $! under caffeinate -i (stops the app on a stall or the max task time, logs a speed alarm; $LOG/watch.log, $LOG/watch/status.json)"
fi
cat <<EOF

FIRST 5 MINUTES: look in $LOG/train.log for these, in order (anything else first = stop and read it):
  1. BUILD $MIXDIR []                           the list is empty (no language without rows)
  2. L2SP ANCHOR .../ultra/state.pt INIT $INIT SUB_BATCH 1   (the anchor is stock also for a branch)
  3. MIX {...} ARM $ARM {...}                   the arm's settings (R10-main: en_teacher, en_kl_scale 3.0, wf_en_weight 3.0)
     branch arms: 'branch options: sampler_seed 4242 lr_offset 1000 lr_total 5000 ...' (E arms: also UNFREEZE_TOP 2,
     'param group 1 ... peak lr 2e-05', then 'lr step 0: group0 ... group1 ...' and 'grad_norm pre-clip dec ... enc ...')
  4. SHARE total ... english part 0.35-0.40     the English-share assert passed
  5. mix <part>: N rows, H h                    one line per listed part; a weighted part appears that many times
  6. staging N/M files R files/s ... staged N files in Ns   R should be several hundred (the watchdog stops the app below
                                                300 files/s after 10 min); then 'train mix: H h' and 'manifest sha256 ... recorded'
  7. en_teacher ...@...: N tensors              the English KL teacher loaded
  8. step 10 loss ... then every 10 steps       about 2 s per step; 'r3 rnnt ... kl_b ...' lines with finite numbers
  9. step250.pt.done in exp/$RUN                first checkpoint, about 10 min after the first step
Red flags: AssertionError, Traceback, 'mix drift', 'Runner failed', nan, no step line 15 min after 'train mix'.
Stop by hand: modal app stop --yes $APP
EOF
