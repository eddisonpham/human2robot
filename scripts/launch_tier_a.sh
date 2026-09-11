#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/logs

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(configs/tier_a_relocate.yaml configs/tier_a_relocate_demo.yaml)
fi

run() {
  local config="$1" seed="$2"
  local config_name run_name log exit_code
  config_name=$(basename "$config" .yaml)
  run_name="${config_name}_seed${seed}"
  log="results/logs/${run_name}.log"
  echo "[$(date +%H:%M:%S)] starting $run_name"
  uv run dynhand-train \
    --config "$config" \
    --seed "$seed" \
    --run-name "$run_name" \
    --resume >> "$log" 2>&1
  exit_code=$?
  echo "[$(date +%H:%M:%S)] finished $run_name with exit code $exit_code"
  if [ "$exit_code" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] aborting queue: $run_name failed, see $log"
    exit "$exit_code"
  fi
}

for config in "${CONFIGS[@]}"; do
  run "$config" 0
done
echo "[$(date +%H:%M:%S)] queue drained"
