#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/logs

# Runs execute one at a time: concurrent training processes contend for
# CPU cores and drop aggregate throughput about fivefold on this machine.
# Run names match the experiment_id in each config so --resume finds
# existing checkpoints. Pass config paths as arguments to run a subset:
#   bash scripts/launch_tier_a.sh configs/tier_a_relocate_demo.yaml

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(configs/tier_a_relocate.yaml configs/tier_a_relocate_demo.yaml)
fi

run() {
  local config="$1" seed="$2"
  local name log exit_code
  name=$(basename "$config" .yaml)
  log="results/logs/${name}.log"
  echo "[$(date +%H:%M:%S)] starting $name (seed $seed)"
  uv run dynhand-train --config "$config" --seed "$seed" --resume >> "$log" 2>&1
  exit_code=$?
  echo "[$(date +%H:%M:%S)] finished $name with exit code $exit_code"
  if [ "$exit_code" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] aborting queue: $name failed, see $log"
    exit "$exit_code"
  fi
}

for config in "${CONFIGS[@]}"; do
  run "$config" 0
done
echo "[$(date +%H:%M:%S)] queue drained"
