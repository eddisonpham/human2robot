"""Rebuild evaluation records in a run's metrics stream from TensorBoard.

The training loop writes every evaluation to both ``metrics.jsonl`` and a
TensorBoard event file, so ``eval/return_mean`` can be recovered from the
latter when the JSONL copy is damaged.

Only ``eval_return_mean`` is recoverable: TensorBoard never saw the return
standard deviation or the episode length.

Usage:

    uv run python scripts/rebuild_eval_metrics.py [run_name ...]
"""

import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVAL_TAG = "eval/return_mean"


def eval_points(run_dir: Path) -> dict[int, float]:
    """Return step -> eval return, merged across every event file in the run."""
    points: dict[int, float] = {}
    for event_file in sorted((run_dir / "tb").glob("events.out.tfevents*")):
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        if EVAL_TAG not in accumulator.Tags()["scalars"]:
            continue
        for scalar in accumulator.Scalars(EVAL_TAG):
            points.setdefault(scalar.step, scalar.value)
    return points


def rebuild(run_dir: Path) -> tuple[int, int]:
    """Merge recovered evaluations into metrics.jsonl, return (kept, added)."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return (0, 0)

    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    have_eval = {int(r["step"]) for r in records if "eval_return_mean" in r}
    wanted = {s: v for s, v in eval_points(run_dir).items() if s not in have_eval}
    if not wanted:
        return (len(records), 0)

    for step, value in wanted.items():
        records.append({"step": step, "eval_return_mean": value})

    # Training and evaluation records may share a step value, so they are
    # ordered independently rather than deduplicated against each other.
    records.sort(key=lambda r: (int(r["step"]), "eval_return_mean" in r))
    metrics_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return (len(records) - len(wanted), len(wanted))


def main() -> None:
    root = Path("results")
    names = sys.argv[1:] or sorted(
        path.name
        for path in root.iterdir()
        if (path / "tb").is_dir() and (path / "metrics.jsonl").exists()
    )
    for name in names:
        kept, added = rebuild(root / name)
        print(f"{name:<32} kept={kept:<5} recovered={added}")


if __name__ == "__main__":
    main()
