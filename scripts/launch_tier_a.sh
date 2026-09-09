#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/logs

# Runs execute one at a time: concurrent training processes contend for
# CPU cores and drop aggregate throughput about fivefold on this machine.
# Run names match the experiment_id in each config so --resume finds
# existing checkpoints.

run() {
  local config="$1" seed="$2"
  local name
  name=$(basename "$config" .yaml)
  local log="results/logs/${name}.log"
  echo "[$(date +%H:%M:%S)] starting $name (seed $seed)"
  uv run dynhand-train --config "$config" --seed "$seed" --resume >> "$log" 2>&1
  echo "[$(date +%H:%M:%S)] finished $name with exit code $?"
}

run configs/tier_a_relocate.yaml 0
run configs/tier_a_relocate_demo.yaml 0
echo "[$(date +%H:%M:%S)] queue drained"
