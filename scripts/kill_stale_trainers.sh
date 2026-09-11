#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv run python - <<'PY'
import os
import psutil

current = os.getpid()
matched = []
for process in psutil.process_iter(["pid", "cmdline"]):
    if process.pid == current:
        continue
    try:
        command = " ".join(process.info["cmdline"] or [])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
    if "dynhand-train" in command or "dynhand.rl.train" in command:
        matched.append(process)

for process in matched:
    try:
        process.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

_, alive = psutil.wait_procs(matched, timeout=10)
for process in alive:
    try:
        process.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

print(f"stopped {len(matched)} trainer process(es)")
PY
