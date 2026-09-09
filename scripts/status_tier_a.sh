#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== trainer processes: $(tasklist 2>/dev/null | grep -ci python || echo 0) python =="
tail -2 results/logs/queue.log 2>/dev/null

check_run() {
  local log="$1" results="$2"
  if [ -f "$log" ]; then
    grep -E "Resumed|Training complete" "$log" | tail -2
    latest_eval=$(grep "eval_return_mean" "$results/metrics.jsonl" 2>/dev/null | tail -1)
    [ -n "$latest_eval" ] && echo "latest eval: $latest_eval"
    latest_ckpt=$(ls "$results/checkpoints/" 2>/dev/null | tail -1)
    [ -n "$latest_ckpt" ] && echo "latest checkpoint: $latest_ckpt"
  else
    echo "no log yet"
  fi
}

echo "== condition A (results/tier_a_relocate_cond_a) =="
check_run results/logs/tier_a_relocate.log results/tier_a_relocate_cond_a

echo "== condition B (results/tier_a_relocate_cond_b) =="
check_run results/logs/tier_a_relocate_demo.log results/tier_a_relocate_cond_b
