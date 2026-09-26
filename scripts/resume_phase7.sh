#!/usr/bin/env bash
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resume Phase 7 Tier B Condition A training for all seeds.
# Per seed: resume from the latest checkpoint if checkpoints exist,
# otherwise start fresh. Seeds already completed are skipped.

SEEDS=(0 1 2)
CONFIG="configs/tier_b_pickup.yaml"

mkdir -p logs

uv run python - "$CONFIG" "${SEEDS[@]}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

config = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]]
results = Path("results")

failures = []
for seed in seeds:
    run_name = f"tier_b_pickup_cond_a_s{seed}"
    run_dir = results / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt")) if checkpoint_dir.exists() else []

    status = None
    status_path = run_dir / "run_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text()).get("status")
        except json.JSONDecodeError:
            status = None

    if status == "completed":
        print(f"seed {seed}: already completed, skipping")
        continue

    args = [
        "uv", "run", "human2robot-train",
        "--config", config,
        "--seed", str(seed),
        "--run-name", run_name,
    ]
    if checkpoints:
        args.append("--resume")
        print(f"seed {seed}: resuming from {checkpoints[-1].name}")
    else:
        print(f"seed {seed}: no checkpoints, starting fresh")

    with open(f"logs/{run_name}.log", "a", encoding="utf-8") as log:
        code = subprocess.call(args, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        print(f"seed {seed}: FAILED with exit code {code}, stopping queue")
        failures.append(seed)
        break
    print(f"seed {seed}: finished")

sys.exit(1 if failures else 0)
PY
