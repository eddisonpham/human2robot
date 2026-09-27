"""Report training progress for one or more runs.

Shows the current step, the rate measured over a short window, an ETA to the
configured step budget, whether the run looks stalled, and whether its metrics
stream still passes the audit. Intended to be called repeatedly by
``scripts/monitor_run.sh``.

Usage:

    uv run python scripts/monitor_run.py <run_name> [run_name ...]
"""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from human2robot.evaluation.audit import audit_metrics
from human2robot.utils.seed import seed_everything

RESULTS = Path("results")
SAMPLE_SECONDS = 60.0
STALL_SECONDS = 600.0


def _read(run_dir: Path) -> tuple[int, int, float]:
    """Return (step, record count, seconds since the last write)."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return 0, 0, float("inf")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        return 0, 0, float("inf")
    try:
        step = int(json.loads(lines[-1])["step"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        step = 0
    return step, len(lines), time.time() - path.stat().st_mtime


def _target_steps(run_dir: Path) -> int:
    """Read the configured step budget from the run's saved config."""
    config = run_dir / "config.yaml"
    if not config.exists():
        return 0
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.startswith("total_env_steps:"):
            return int(line.split(":", 1)[1].strip())
    return 0


def main() -> None:
    seed_everything(0)
    names = sys.argv[1:]
    if not names:
        print("usage: monitor_run.py <run_name> [run_name ...]")
        raise SystemExit(2)

    targets = [(name, _target_steps(RESULTS / name)) for name in names]
    before = {name: _read(RESULTS / name) for name in names}
    print(f"[{time.strftime('%H:%M:%S')}] sampling for {SAMPLE_SECONDS:.0f}s")
    time.sleep(SAMPLE_SECONDS)

    for name, target in targets:
        run_dir = RESULTS / name
        step, records, age = _read(run_dir)
        prior = before[name][0]
        rate = (step - prior) / (SAMPLE_SECONDS / 3600.0) if step > prior else 0.0

        status = "unknown"
        status_file = run_dir / "run_status.json"
        if status_file.exists():
            try:
                status = json.loads(status_file.read_text())["status"]
            except (json.JSONDecodeError, KeyError):
                status = "unreadable"

        stalled = "STALLED" if age > STALL_SECONDS else "ok"
        if status == "completed" or rate == 0.0 and status != "running":
            stalled = "done"

        eta = "n/a"
        if target and step and rate > 0:
            hours = (target - step) / rate
            finish = datetime.now() + timedelta(hours=hours)
            eta = f"{finish:%H:%M} ({hours:.2f}h)"

        try:
            healthy = audit_metrics(run_dir / "metrics.jsonl").healthy
        except OSError:
            healthy = False
        audit = "clean" if healthy else "UNHEALTHY"

        print(
            f"  {name:<28} step={step:>9,} rec={records:>5} "
            f"rate={rate:>10,.0f}/h eta={eta:<18} "
            f"stall={stalled:<8} audit={audit:<11} status={status}"
        )


if __name__ == "__main__":
    main()
