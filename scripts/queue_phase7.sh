#!/usr/bin/env bash
set -u
mkdir -p results logs
seeds=(0 1 2)
for seed in "${seeds[@]}"; do
  id="tier_b_pickup_cond_a_s${seed}"
  echo "=== launching seed ${seed} as ${id} ==="
  uv run human2robot-train --config configs/tier_b_pickup.yaml \
    --seed "${seed}" --run-name "${id}" \
    > "logs/${id}.log" 2>&1
  code=$?
  if [ ${code} -ne 0 ]; then
    echo "seed ${seed} FAILED with exit ${code}; stopping queue"
    exit ${code}
  fi
  echo "seed ${seed} finished"
done
echo "phase 7 queue complete"
