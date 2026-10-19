#!/bin/bash
# Run the run-set orchestrator under caffeinate -i, restarting it if it dies. State and logs come from the config file.
#
#   scripts/trigger.sh <config.json> [--dry-run] [--once]
#
# trigger.py exits 0 when everything planned is settled and 1 when its time limit passes with work pending; exit 2
# (bad config) and 3 (another copy holds the state lock) end this script too. Any other exit (a crash, a kill signal)
# restarts it after 30 s.
#   scripts/trigger.sh quiet [config.json]     list the apps holding a running task besides this night's own runs; exit 1 if
#                                              there are more than the config's quiet_max_other (default 0): run it before a launch
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "${1:-}" = quiet ]; then
  shift
  exec python3 "$HERE/trigger.py" --quiet-check ${1:+--config "$1"}
fi
[ $# -ge 1 ] || { echo "usage: trigger.sh <config.json> [--dry-run] [--once]" >&2; exit 2; }
command -v caffeinate >/dev/null 2>&1 || { echo "caffeinate not found: the orchestrator must run under caffeinate -i" >&2; exit 9; }
if [ "$1" != "--inner" ]; then
  exec caffeinate -i bash "$0" --inner "$@"
fi
shift
CONFIG=$1; shift
while true; do
  python3 "$HERE/trigger.py" --config "$CONFIG" "$@"
  rc=$?
  case $rc in 0|1|2|3) exit $rc;; esac
  echo "trigger.py exited $rc; restarting in 30 s" >&2
  sleep 30
done
