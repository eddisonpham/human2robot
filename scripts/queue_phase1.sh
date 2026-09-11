#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/logs

run_job() {
  local config="$1" seed="$2" name="$3"
  local log="results/logs/${name}.log"
  local exit_code
  echo "[$(date +%H:%M:%S)] starting ${name}"
  uv run dynhand-train --config "$config" --seed "$seed" --run-name "$name" --resume \
    >> "$log" 2>&1
  exit_code=$?
  echo "[$(date +%H:%M:%S)] finished ${name} with exit code ${exit_code}"
  if [ "$exit_code" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] queue stopped; inspect ${log}"
    exit "$exit_code"
  fi
}

run_job configs/tier_a_relocate_demo.yaml 0 tier_a_relocate_demo_seed0
run_job configs/tier_a_relocate.yaml 0 tier_a_relocate_seed0

if uv run --no-sync python -c "import stable_baselines3" 2>/dev/null; then
  uv run dynhand-sb3-check \
    --config configs/tier_a_relocate.yaml \
    --run-name tier_a_sb3_seed0 \
    >> results/logs/tier_a_sb3_seed0.log 2>&1
  exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] queue stopped; inspect results/logs/tier_a_sb3_seed0.log"
    exit "$exit_code"
  fi
else
  echo "SB3 is not installed. Run: uv sync --group sb3"
  exit 2
fi

for seed in 1 2; do
  run_job configs/tier_a_relocate.yaml "$seed" "tier_a_relocate_seed${seed}"
  run_job configs/tier_a_relocate_demo.yaml "$seed" "tier_a_relocate_demo_seed${seed}"
done

echo "[$(date +%H:%M:%S)] phase 1 queue drained"
