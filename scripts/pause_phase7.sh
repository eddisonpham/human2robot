#!/usr/bin/env bash
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Gracefully pause Phase 7 training: stop the queue wrapper, wait for the
# in-flight trainer to pass its next checkpoint boundary if one is close,
# then terminate it. Safe to run repeatedly.

QUEUE_LOG="logs/phase7_queue.log"

# 1. Stop the queue wrapper first so it does not launch the next seed.
pids=$(ps -ef | grep -E "bash.*queue_phase7\.sh" | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then
  echo "stopping queue wrapper: $pids"
  kill $pids 2>/dev/null || true
  sleep 2
  kill -9 $pids 2>/dev/null || true
else
  echo "no queue wrapper running"
fi

# 2. Ask the trainer nicely to stop, then force after a grace period.
uv run python - <<'PY'
import os
import time

import psutil

current = os.getpid()
trainers = []
for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
    if process.pid == current:
        continue
    try:
        command = " ".join(process.info["cmdline"] or [])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
    if "human2robot-train" in command or "human2robot.rl.train" in command:
        trainers.append(process)

if not trainers:
    print("no trainer processes running")
    raise SystemExit(0)

for process in trainers:
    print(f"sending terminate to trainer pid {process.pid}")
    process.terminate()

_, alive = psutil.wait_procs(trainers, timeout=30)
for process in alive:
    print(f"force killing trainer pid {process.pid}")
    process.kill()
print(f"paused {len(trainers)} trainer(s); resume with scripts/resume_phase7.sh")
PY

# 3. Clear any stale lock file left behind so resume is instant.
uv run python - <<'PY'
from pathlib import Path

for lock in Path("results").glob("*/.run.lock"):
    lock.unlink(missing_ok=True)
    print(f"cleared {lock}")
PY

echo "pause complete"
