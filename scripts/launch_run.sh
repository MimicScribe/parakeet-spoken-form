#!/bin/bash
# Launch a training run of the current recipe from a clean export of one commit, behind three locks.
#
#   scripts/launch_run.sh --run <name> --arm <arm> --sha <commit> --max-steps <N> --mix-dir <dir on the volume> \
#       --repo <git checkout> --approvals <file> --log-dir <dir> [--part-weights "<path>=<n>,..."] \
#       [--extra-parts "<path>:<tag>:<n>,..."] [--langs <l1,l2,...>] [--volume <name>] [--resume] [--dry-run] [--no-watch]
#
# Locks (a refusal exits non-zero and starts nothing):
#   1. the exact run name must be a line of the approvals file (exit 3);
#   2. exp/<run> must not exist on the volume (exit 4); with --resume it must exist, and the commit must be the one the
#      first launch logged in --log-dir (exit 5);
#   3. the arm must exist in the commit, max-steps must be a multiple of the checkpoint interval (250), the run name
#      must be a plain name (exit 2).
# The run is launched from `git archive <sha>` of --repo (never the working tree). The commit and the sha256 of
# modal_app.py and of every module the training image mounts go to <log-dir>/launch_manifest.txt before anything starts.
# The training function is retrain3_80_main (A100 80 GB, 6 h cap per attempt) with the standard recipe flags. Unless
# --no-watch, run_watch.py starts beside it and stops the app if checkpoints stall. Paths come from the arguments only.
set -u
SAVE_EVERY=250
VOLUME=parakeet-spoken-form
RUN=""; ARM=""; SHA=""; STEPS=""; MIXDIR=""; REPO=""; APPROVALS=""; LOG=""; PW=""; XP=""; LANGS=""
RESUME=0; DRY=0; WATCH=1
die() { echo "REFUSED: $2" >&2; exit "$1"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=$2; shift 2;; --arm) ARM=$2; shift 2;; --sha) SHA=$2; shift 2;; --max-steps) STEPS=$2; shift 2;;
    --mix-dir) MIXDIR=$2; shift 2;; --repo) REPO=$2; shift 2;; --approvals) APPROVALS=$2; shift 2;;
    --log-dir) LOG=$2; shift 2;; --part-weights) PW=$2; shift 2;; --extra-parts) XP=$2; shift 2;;
    --langs) LANGS=$2; shift 2;; --volume) VOLUME=$2; shift 2;;
    --resume) RESUME=1; shift;; --dry-run) DRY=1; shift;; --no-watch) WATCH=0; shift;;
    *) die 2 "unknown argument $1";;
  esac
done
for v in RUN ARM SHA STEPS MIXDIR REPO APPROVALS LOG; do [ -n "${!v}" ] || die 2 "missing --$(echo $v | tr 'A-Z_' 'a-z-') (see the header of this script)"; done
[[ $RUN =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die 2 "run name '$RUN' is not a plain name"
[[ $STEPS =~ ^[0-9]+$ ]] && [ "$STEPS" -gt 0 ] && [ $((STEPS % SAVE_EVERY)) -eq 0 ] || die 2 "--max-steps $STEPS is not a multiple of $SAVE_EVERY"

EXPORT=""
die() { echo "REFUSED: $2" >&2; [ -z "$EXPORT" ] || rm -rf "$EXPORT"; exit "$1"; }

# lock 1: the owner's approval for THIS run name (one exact line)
[ -f "$APPROVALS" ] && grep -qxF -- "$RUN" "$APPROVALS" || die 3 "run '$RUN' has no approval line in $APPROVALS"

# lock 2: a new run name (or, with --resume, the same run and the same commit)
FULL=$(git -C "$REPO" rev-parse --verify --quiet "$SHA^{commit}") || die 2 "commit $SHA not found in $REPO"
if modal volume ls "$VOLUME" "exp/$RUN" >/dev/null 2>&1; then
  [ $RESUME -eq 1 ] || die 4 "exp/$RUN already exists on $VOLUME (a launch would silently resume it); use a new run name, or --resume on purpose"
else
  [ $RESUME -eq 0 ] || die 4 "--resume: exp/$RUN does not exist on $VOLUME"
fi
if [ $RESUME -eq 1 ]; then
  FIRST=$(sed -n 's/^commit //p' "$LOG/launch_manifest.txt" 2>/dev/null | head -1)
  [ "$FIRST" = "$FULL" ] || die 5 "--resume must use the commit of the first launch (${FIRST:-none logged in $LOG}), not $FULL"
fi

# lock 3: the arm exists in the commit
mkdir -p "$LOG" || die 2 "cannot create $LOG"
EXPORT=$(mktemp -d "$LOG/export.XXXXXX") || die 2 "cannot create an export directory in $LOG"
git -C "$REPO" archive "$FULL" | tar -x -C "$EXPORT" || die 2 "git archive failed"
PIPE=$EXPORT/pipeline
python3 "$PIPE/run_manifest.py" "$PIPE" arm "$ARM" || die 2 "arm '$ARM' is not defined in modal_app.py at $FULL"
[ -n "$LANGS" ] || LANGS=$(python3 "$PIPE/run_manifest.py" "$PIPE" langs) || die 2 "cannot read the language list"

CMD=(modal run --detach modal_app.py::retrain3_80_main --run "$RUN" --arm "$ARM" --mix v4 --mix-dir "$MIXDIR"
     --langs "$LANGS" --init-from ultra --anchor-init --dur-weight 0 --warmup 20 --max-steps "$STEPS"
     --save-every $SAVE_EVERY --r3-sub-batch 1)
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
  nohup python3 "$PIPE/run_watch.py" --app "$APP" --run "$RUN" --volume "$VOLUME" --log-dir "$LOG/watch" \
    > "$LOG/watch.log" 2>&1 &
  echo "watchdog pid $! (stops the app when no new step marker appears; $LOG/watch.log)"
fi
cat <<EOF

FIRST 5 MINUTES: look in $LOG/train.log for these, in order (anything else first = stop and read it):
  1. BUILD $MIXDIR []                           the list is empty (no language without rows)
  2. L2SP ANCHOR .../ultra/state.pt INIT ultra SUB_BATCH 1
  3. MIX {...} ARM $ARM {...}                   the arm's settings (R10-main: en_teacher, en_kl_scale 3.0, wf_en_weight 3.0)
  4. SHARE total ... english part 0.35-0.40     the English-share assert passed
  5. mix <part>: N rows, H h                    one line per listed part; a weighted part appears that many times
  6. staged N files in Ns                         then 'train mix: H h' and 'manifest sha256 ... recorded'
  7. en_teacher ...@...: N tensors              the English KL teacher loaded
  8. step 10 loss ... then every 10 steps       about 2 s per step; 'r3 rnnt ... kl_b ...' lines with finite numbers
  9. step250.pt.done in exp/$RUN                first checkpoint, about 10 min after the first step
Red flags: AssertionError, Traceback, 'mix drift', 'Runner failed', nan, no step line 15 min after 'train mix'.
Stop by hand: modal app stop $APP
EOF
