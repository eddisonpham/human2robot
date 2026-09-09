"""Plot learning curves from recorded runs.

Example:
  uv run dynhand-plot --runs "SAC=results/run_a, SAC+demo=results/run_b" \
    --metric eval_return_mean --out results/plots/tier_a.png
"""

import argparse
from pathlib import Path

from dynhand.evaluation.plots import plot_learning_curves


def parse_runs(spec: str) -> dict[str, list[Path]]:
    """Parse a label=path,label=path spec into the runs mapping."""
    runs: dict[str, list[Path]] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        label, _, path = item.partition("=")
        runs[label.strip()] = [Path(path.strip()) / "metrics.jsonl"]
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curves")
    parser.add_argument(
        "--runs",
        required=True,
        help="Comma-separated label=run_dir pairs (run_dir contains metrics.jsonl)",
    )
    parser.add_argument("--metric", default="eval_return_mean")
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--title", default="")
    parser.add_argument("--out", default="results/plots/learning_curves.png")
    args = parser.parse_args()
    runs = parse_runs(args.runs)
    out = plot_learning_curves(
        runs,
        metric=args.metric,
        smooth=args.smooth,
        title=args.title,
        out_path=args.out,
    )
    print(out)


if __name__ == "__main__":
    main()
