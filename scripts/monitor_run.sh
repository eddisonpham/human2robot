#!/usr/bin/env bash
# Report progress on the named runs: step, rate, ETA, stall detection, and
# whether each metrics stream still passes the audit.
#
# Usage: scripts/monitor_run.sh <interval_seconds> <run_name> [run_name ...]
set -u

INTERVAL="${1:?usage: monitor_run.sh <interval_seconds> <run_name>...}"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

while true; do
  uv run python scripts/monitor_run.py "$@" 2>&1 | grep -v "reward functions"
  sleep "$INTERVAL"
done
