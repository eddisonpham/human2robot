#!/usr/bin/env bash
# Pause one named training run.
#
# Usage: scripts/pause_run.sh <run_name> [--now]
#
# Creates results/<run_name>/PAUSE, which the training loop polls, so the run
# stops at a clean checkpoint instead of being killed. Windows cannot deliver
# SIGTERM to a detached process, so the sentinel file is the pause channel.
#
# --now skips the graceful wait and kills immediately, which rewinds the run to
# its last periodic checkpoint and loses up to checkpoint_interval steps.
set -u

RUN_NAME="${1:?usage: pause_run.sh <run_name> [--now]}"
FORCE_KILL="${2:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

RUN_DIR="${RUN_DIR_OVERRIDE:-results}/${RUN_NAME}"
PAUSE_FILE="$RUN_DIR/PAUSE"

if [ ! -d "$RUN_DIR" ]; then
  echo "no run directory at $RUN_DIR" >&2
  exit 1
fi

pids_for() {
  powershell -NoProfile -Command \
    "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '${RUN_NAME}' } | Select-Object -ExpandProperty ProcessId" \
    2>/dev/null | tr -d '\r'
}

report_point() {
  echo
  echo "=== $RUN_NAME ==="
  LATEST=$(ls -t "$RUN_DIR"/checkpoints/step_*.pt 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    STEP=$(basename "$LATEST" | sed 's/step_//; s/\.pt$//')
    echo "resume point: step $STEP"
    if [ -f "$RUN_DIR/metrics.jsonl" ]; then
      REACHED=$(tail -1 "$RUN_DIR/metrics.jsonl" | sed -n 's/.*"step": *\([0-9]*\).*/\1/p')
      if [ -n "$REACHED" ] && [ "$REACHED" -gt "$STEP" ] 2>/dev/null; then
        echo "rewind on resume: $((REACHED - STEP)) steps"
      else
        echo "rewind on resume: none"
      fi
    fi
  else
    echo "no checkpoints; a resume would start fresh"
  fi
  if [ -f "$RUN_DIR/run_status.json" ]; then
    echo "status: $(tr -d '\r\n ' < "$RUN_DIR/run_status.json")"
  fi
}

PIDS=$(pids_for)
if [ -z "$PIDS" ]; then
  echo "no running process for $RUN_NAME"
  rm -f "$PAUSE_FILE"
  report_point
  exit 0
fi

if [ "$FORCE_KILL" = "--now" ]; then
  echo "killing immediately (--now)"
  for pid in $PIDS; do taskkill //F //PID "$pid" >/dev/null 2>&1; done
  sleep 3
  rm -f "$RUN_DIR/.run.lock" 2>/dev/null
  report_point
  exit 0
fi

echo "requesting a clean pause (writing $PAUSE_FILE)"
touch "$PAUSE_FILE"

# The loop polls every 100 steps, so a checkpointed stop lands within seconds.
for _ in $(seq 1 40); do
  sleep 3
  if [ -z "$(pids_for)" ]; then
    break
  fi
done

if [ -n "$(pids_for)" ]; then
  echo "still running after 120s; killing (rewinds to last checkpoint)"
  for pid in $(pids_for); do taskkill //F //PID "$pid" >/dev/null 2>&1; done
  sleep 3
  touch "$PAUSE_FILE"
else
  echo "paused cleanly"
fi

rm -f "$RUN_DIR/.run.lock" 2>/dev/null
report_point
