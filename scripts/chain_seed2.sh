#!/usr/bin/env bash
# Launch Condition C seed 2 once C seed 1 completes.
#
# D and E seeds are deliberately absent: the residual dynamics conditions are
# degenerate while the nominal physics model is the exact simulator, so those
# runs cannot answer their question until domain randomization exists. See
# docs/FINDINGS_residual_degeneracy.md.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

log() { echo "[chain $(date +%H:%M:%S)] $*"; }

while ! grep -q '"status": "completed"' \
  "results/tier_b_pickup_cond_c_s1/run_status.json" 2>/dev/null; do
  sleep 120
done

log "C s1 complete; launching C s2"
rm -rf results/tier_b_pickup_cond_c_s2
nohup uv run human2robot-train --config configs/tier_b_cond_c.yaml \
  --seed 2 --run-name tier_b_pickup_cond_c_s2 >logs/fix_cond_c_s2.log 2>&1 &
log "launched tier_b_pickup_cond_c_s2 (pid $!)"

nohup bash scripts/watchdog_runs.sh 45 tier_b_pickup_cond_c_s2 \
  >logs/watchdog_c_s2.log 2>&1 &
log "watchdog started for C s2 (pid $!)"
