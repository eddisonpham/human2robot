#!/usr/bin/env bash
# Resume one named training run from its latest checkpoint.
#
# Usage: scripts/resume_run.sh <run_name> [config.yaml]
#
# Set SEED=<n> when the run name does not end in _s<n>. Set RUN_DIR_OVERRIDE when
# the run lives outside results/. Set DEVICE=cuda to opt back into the GPU.
#
# The config is inferred from the run name when not given. Resuming restores SAC
# weights, the replay buffer, the dynamics ensemble, and RNG state, so training
# continues from the checkpoint step rather than restarting.
set -u

RUN_NAME="${1:?usage: resume_run.sh <run_name> [config.yaml]}"
CONFIG="${2:-}"

# CPU by default. A killed training process can leave a CUDA context behind,
# after which the next process blocks forever inside cudaStreamSynchronize
# while PyTorch validates a distribution argument. Measured cost of CPU over
# CUDA on a 40k-step Condition C run is 129s versus 120s, about 7 percent.
DEVICE="${DEVICE:-cpu}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

if [ -z "$CONFIG" ]; then
  case "$RUN_NAME" in
    tier_b_pickup_cond_c_s*) CONFIG=configs/tier_b_cond_c.yaml ;;
    tier_b_pickup_cond_d_s*) CONFIG=configs/tier_b_cond_d.yaml ;;
    tier_b_pickup_cond_e_s*) CONFIG=configs/tier_b_cond_e.yaml ;;
    tier_b_pickup_cond_b_s*) CONFIG=configs/tier_b_cond_b.yaml ;;
    tier_b_pickup_cond_a_s*) CONFIG=configs/tier_b_pickup.yaml ;;
    tier_a_*) CONFIG=configs/tier_a_relocate.yaml ;;
    *) echo "cannot infer config for $RUN_NAME; pass it as the second argument" >&2
       exit 2 ;;
  esac
fi

SEED="${SEED:-$(echo "$RUN_NAME" | sed -n 's/.*_s\([0-9]\+\)$/\1/p')}"
if [ -z "$SEED" ]; then
  echo "cannot infer seed from $RUN_NAME; set SEED=<n> in the environment" >&2
  exit 2
fi

RUN_DIR="${RUN_DIR_OVERRIDE:-results}/${RUN_NAME}"
if [ ! -d "$RUN_DIR/checkpoints" ]; then
  echo "no checkpoints under $RUN_DIR; starting fresh" >&2
  RESUME_FLAG=""
else
  LATEST=$(ls -t "$RUN_DIR"/checkpoints/step_*.pt 2>/dev/null | head -1)
  if [ -z "$LATEST" ]; then
    echo "no checkpoint files found; starting fresh" >&2
    RESUME_FLAG=""
  else
    echo "resuming $RUN_NAME from $(basename "$LATEST") (seed $SEED)"
    RESUME_FLAG="--resume"
  fi
fi

nohup uv run human2robot-train --config "$CONFIG" --seed "$SEED" \
  --run-name "$RUN_NAME" --device "$DEVICE" $RESUME_FLAG \
  >"logs/${RUN_NAME}.log" 2>&1 &
echo "launched pid $! on $DEVICE; log: logs/${RUN_NAME}.log"
