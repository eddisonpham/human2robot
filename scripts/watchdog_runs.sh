#!/usr/bin/env bash
# Watchdog for Tier B ablation runs.
#
# Addresses the undiagnosed hang that has wedged four runs so far: the process
# stays alive and burns CPU but stops writing metrics for 25-45 minutes. This
# watches for that signature and restarts the run from its latest checkpoint.
#
# Usage: scripts/watchdog_runs.sh <stall_minutes> <run_name> [run_name ...]
set -u

STALL_MINUTES="${1:?usage: watchdog_runs.sh <stall_minutes> <run_name>...}"
shift

STALL_SECONDS=$((STALL_MINUTES * 60))
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

log() { echo "[watchdog $(date +%H:%M:%S)] $*"; }

is_running() {
  local run_name="$1"
  powershell -NoProfile -Command \
    "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '${run_name}' } | Select-Object -ExpandProperty ProcessId" \
    2>/dev/null | tr -d '\r' | grep -q '[0-9]'
}

kill_run() {
  local run_name="$1"
  local pids
  pids=$(powershell -NoProfile -Command \
    "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match '${run_name}' } | Select-Object -ExpandProperty ProcessId" \
    2>/dev/null | tr -d '\r')
  for pid in $pids; do
    taskkill //F //PID "$pid" >/dev/null 2>&1
  done
  log "killed $run_name (pids: ${pids:-none})"
}

config_for() {
  case "$1" in
    tier_b_pickup_cond_c_s*) echo configs/tier_b_cond_c.yaml ;;
    tier_b_pickup_cond_d_s*) echo configs/tier_b_cond_d.yaml ;;
    tier_b_pickup_cond_e_s*) echo configs/tier_b_cond_e.yaml ;;
    tier_b_pickup_cond_b_s*) echo configs/tier_b_cond_b.yaml ;;
    *) echo configs/tier_b_pickup.yaml ;;
  esac
}

seed_for() {
  echo "$1" | sed -n 's/.*_s\([0-9]\+\)$/\1/p'
}

log "watching ${#} runs with a ${STALL_MINUTES} minute stall threshold: $*"

while true; do
  sleep 300
  for run_name in "$@"; do
    metrics="results/${run_name}/metrics.jsonl"
    status="results/${run_name}/run_status.json"
    [ -f "$metrics" ] || continue

    # A finished run is done; nothing to watch.
    if grep -q '"status": "completed"' "$status" 2>/dev/null; then
      continue
    fi

    # Completed runs get their status file written last, but a run that stopped
    # on its own without completing is the hang we care about.
    if ! is_running "$run_name"; then
      if ! grep -q '"status": "completed"' "$status" 2>/dev/null; then
        log "$run_name exited without completing; restarting from checkpoint"
        kill_run "$run_name"
        nohup uv run human2robot-train --config "$(config_for "$run_name")" \
          --seed "$(seed_for "$run_name")" --run-name "$run_name" --resume \
          >>"logs/${run_name}_watchdog.log" 2>&1 &
      fi
      continue
    fi

    age=$(( $(date +%s) - $(stat -c %Y "$metrics") ))
    if [ "$age" -gt "$STALL_SECONDS" ]; then
      log "$run_name has not written metrics for $((age / 60)) min; restarting from checkpoint"
      kill_run "$run_name"
      sleep 5
      nohup uv run human2robot-train --config "$(config_for "$run_name")" \
        --seed "$(seed_for "$run_name")" --run-name "$run_name" --resume \
        >>"logs/${run_name}_watchdog.log" 2>&1 &
      sleep 30
    fi
  done
done
