"""Summarize the Tier B 5-condition ablation matrix from run metrics.

Reads every ``results/tier_b_pickup_cond_<letter>_s<seed>`` run directory and
prints the trajectory statistics used in docs/RL_RESULTS.md, so the writeup can
be regenerated from disk rather than transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CONDITIONS = ["A", "B", "C", "D", "E"]
DESCRIPTION = {
    "A": "from-scratch SAC",
    "B": "BC-init + demo replay",
    "C": "blackbox dynamics aug",
    "D": "residual dynamics aug",
    "E": "demos + residual dynamics aug",
}


def _load(run_dir: Path) -> list[dict]:
    """Return every metrics row for one run directory."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _status(run_dir: Path) -> str:
    """Return the recorded run status, or unknown when absent."""
    path = run_dir / "run_status.json"
    if not path.exists():
        return "unknown"
    with path.open(encoding="utf-8") as handle:
        return str(json.load(handle).get("status", "unknown"))


def summarize(run_dir: Path) -> dict:
    """Compute trajectory and critic statistics for a single run."""
    rows = _load(run_dir)
    evals = [r["eval_return_mean"] for r in rows if "eval_return_mean" in r]
    losses = [r["qf_loss"] for r in rows if "qf_loss" in r]
    return {
        "name": run_dir.name,
        "status": _status(run_dir),
        "final_step": max((r["step"] for r in rows), default=0),
        "evals": len(evals),
        "mean": statistics.fmean(evals) if evals else float("nan"),
        "median": statistics.median(evals) if evals else float("nan"),
        "best": max(evals) if evals else float("nan"),
        "worst": min(evals) if evals else float("nan"),
        "last": evals[-1] if evals else float("nan"),
        "qf_median": statistics.median(losses) if losses else float("nan"),
        "qf_max": max(losses) if losses else float("nan"),
        "spikes_1e3": sum(1 for x in losses if x > 1e3),
        "spikes_1e6": sum(1 for x in losses if x > 1e6),
    }


def find_runs(results_dir: Path) -> dict[str, list[Path]]:
    """Group run directories by condition letter, ordered by seed."""
    found: dict[str, list[Path]] = {c: [] for c in CONDITIONS}
    for letter in CONDITIONS:
        for path in sorted(results_dir.glob(f"tier_b_pickup_cond_{letter}_s*")):
            if path.is_dir():
                found[letter].append(path)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--markdown", action="store_true", help="Emit GitHub-flavored tables"
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    runs = find_runs(results_dir)

    summaries: dict[str, list[dict]] = {}
    for letter, paths in runs.items():
        summaries[letter] = [summarize(p) for p in paths]

    if args.markdown:
        print("| Condition | Seeds | Mean eval | Median | Best | Worst | Final |")
        print("|-----------|-------|-----------|--------|------|-------|-------|")
        for letter in CONDITIONS:
            stats = [s for s in summaries[letter] if s["evals"]]
            if not stats:
                continue
            means = statistics.fmean(s["mean"] for s in stats)
            print(
                f"| {letter} ({DESCRIPTION[letter]}) | {len(stats)} | "
                f"{means:.1f} | "
                f"{statistics.median(s['median'] for s in stats):.1f} | "
                f"{max(s['best'] for s in stats):.1f} | "
                f"{min(s['worst'] for s in stats):.1f} | "
                f"{statistics.fmean(s['last'] for s in stats):.1f} |"
            )
        print()
        print(
            "| Condition | Seed | Status | Step | Evals | Mean |"
            " qf median | qf max | >1e3 | >1e6 |"
        )
        print(
            "|-----------|------|--------|------|-------|------|"
            "-----------|--------|------|------|"
        )
        for letter in CONDITIONS:
            for s in summaries[letter]:
                print(
                    f"| {letter} | {s['name'].split('_')[-1]} | {s['status']} | "
                    f"{s['final_step']} | {s['evals']} | {s['mean']:.1f} | "
                    f"{s['qf_median']:.4g} | {s['qf_max']:.4g} | "
                    f"{s['spikes_1e3']} | {s['spikes_1e6']} |"
                )
        return

    for letter in CONDITIONS:
        for s in summaries[letter]:
            print(
                f"{letter} {s['name'][-3:]:>4} {s['status']:>9} "
                f"step={s['final_step']:>8} evals={s['evals']:>4} "
                f"mean={s['mean']:>9.1f} best={s['best']:>8.1f} "
                f"qf_med={s['qf_median']:>9.4g} qf_max={s['qf_max']:>10.4g} "
                f"spikes={s['spikes_1e3']}/{s['spikes_1e6']}"
            )


if __name__ == "__main__":
    main()
