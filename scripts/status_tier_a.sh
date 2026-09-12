#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python_count=$(tasklist 2>/dev/null | grep -ci python || true)
echo "== python processes: ${python_count:-0} =="

check_run() {
  local name="$1"
  local run_dir="results/${name}"
  local log="results/logs/${name}.log"
  echo "== ${name} =="
  if [ ! -d "$run_dir" ]; then
    echo "status: not started"
    return
  fi
  if [ -f "$run_dir/run_status.json" ]; then
    status=$(uv run --no-sync python -c "import json; print(json.load(open('$run_dir/run_status.json'))['status'])")
    echo "status: ${status}"
  elif [ -f "$log" ] && grep -q "Training complete" "$log"; then
    echo "status: complete"
  else
    echo "status: active or failed"
  fi
  uv run --no-sync python - "$run_dir" <<'PY'
import sys
from dynhand.evaluation.audit import summarize

report = summarize(sys.argv[1])
print(f"metrics: eval_lines={report.eval_lines} healthy={report.healthy}")
if not report.healthy:
    print(f"duplicates={report.duplicate_steps}")
    print(f"out_of_order={report.out_of_order_transitions}")
PY
  latest=$(ls -v "$run_dir/checkpoints"/step_*.pt 2>/dev/null | tail -1 || true)
  [ -n "$latest" ] && echo "latest checkpoint: $latest"
}

check_run tier_a_relocate_seed0
check_run tier_a_relocate_demo_seed0
check_run tier_a_sb3_seed0
