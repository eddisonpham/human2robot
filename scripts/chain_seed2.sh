#!/usr/bin/env bash
# Launch the seed 2 runs once the current wave finishes.
#
# Wave 1 (C s1, D s0/s1, E s0/s1) shares 24 cores, so running seed 2 alongside
# it would slow everything down. This waits for the D and E runs to reach
# completion, then starts seed 2 and hands the new runs to the watchdog.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

wait_for() {
  local run_name="$1"
  while true; do
    if grep -q '"status": "completed"' "results/${run_name}/run_status.json" 2>/dev/null; then
      return 0
    fi
    sleep 300
  done
}

log "waiting for wave 1 to complete (D s0, D s1, E s0, E s1)"
wait_for tier_b_pickup_cond_d_s0
wait_for tier_b_pickup_cond_d_s1
wait_for tier_b_pickup_cond_e_s0
wait_for tier_b_pickup_cond_e_s1
log "wave 1 done; launching seed 2 for C, D, E"

for c in c d e; do
  run_name="tier_b_pickup_cond_${c}_s2"
  rm -rf "results/${run_name}"
  nohup uv run human2robot-train --config "configs/tier_b_cond_${c}.yaml" \
    --seed 2 --run-name "$run_name" >"logs/fix_cond_${c}_s2.log" 2>&1 &
  log "launched $run_name (pid $!)"
  sleep 5
done

# Hand the seed 2 runs to their own watchdog.
nohup bash scripts/watchdog_runs.sh 45 \
  tier_b_pickup_cond_c_s2 tier_b_pickup_cond_d_s2 tier_b_pickup_cond_e_s2 \
  >logs/watchdog_wave2.log 2>&1 &
log "watchdog for wave 2 started (pid $!)"
