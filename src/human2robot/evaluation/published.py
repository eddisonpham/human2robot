"""The published figures, read from the artifacts they were measured on.

Every headline number in `README.md` and `docs/RESULTS.md` is produced by one
of the experiment scripts in `scripts/` and written to
`results/trajectory_optimization/*.json`. The documentation used to restate
those numbers by hand, and did so correctly for a while and then did not: the
kinematic table reported a descent rate of 99/100 while the recorded artifact
said 100/100, and the split-granularity table quoted three-seed error bars for
an experiment that had only ever been run at one seed.

The defect is that the documents are a second, unversioned copy of data that
already exists on disk in a machine-readable form. This module is the
definition of which figure comes from which artifact, so there is one place to
look and one place to change.

`docs/published_figures.json` is a committed snapshot of the figures, produced
by `refresh_published_figures()` and checked by
`tests/python/test_published_results.py` against both this module and the live
artifacts. The snapshot exists because `results/` is gitignored: a test that
only ran where a local experiment directory happened to exist would pass
vacuously on a fresh clone, which is the failure mode this is meant to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "RESULTS_DIR",
    "FIGURE_NAMES",
    "load_snapshot",
    "load_figures",
    "refresh_published_figures",
]

#: Where the experiment scripts write. Gitignored, so absent on a fresh clone.
RESULTS_DIR = Path("results/trajectory_optimization")

#: The committed snapshot the documentation is checked against.
SNAPSHOT_PATH = Path("docs/published_figures.json")

#: Every figure the documentation quotes, with the artifact it comes from. The
#: mapping is explicit rather than discovered so that a renamed or missing
#: artifact fails here with a clear message instead of silently dropping a
#: figure from the snapshot.
SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "jerk_reduction_s1": (
        "real_vs_synthetic_s1.json",
        ("dexycb", "max_jerk_reduction_pct"),
    ),
    "jerk_reduction_s2": (
        "real_vs_synthetic_s2.json",
        ("dexycb", "max_jerk_reduction_pct"),
    ),
    "jerk_reduction_synthetic": (
        "real_vs_synthetic_s1.json",
        ("synthetic", "max_jerk_reduction_pct"),
    ),
    "smoothness_reduction_s1": (
        "real_vs_synthetic_s1.json",
        ("dexycb", "smoothness_reduction_pct"),
    ),
    "smoothness_reduction_s2": (
        "real_vs_synthetic_s2.json",
        ("dexycb", "smoothness_reduction_pct"),
    ),
    "velocity_reduction_s1": (
        "real_vs_synthetic_s1.json",
        ("dexycb", "max_velocity_reduction_pct"),
    ),
    "velocity_reduction_s2": (
        "real_vs_synthetic_s2.json",
        ("dexycb", "max_velocity_reduction_pct"),
    ),
    "acceleration_reduction_s1": (
        "real_vs_synthetic_s1.json",
        ("dexycb", "max_acceleration_reduction_pct"),
    ),
    "acceleration_reduction_s2": (
        "real_vs_synthetic_s2.json",
        ("dexycb", "max_acceleration_reduction_pct"),
    ),
    "convergence_s1": ("real_vs_synthetic_s1.json", ("dexycb", "converged")),
    "convergence_s2": ("real_vs_synthetic_s2.json", ("dexycb", "converged")),
    "convergence_synthetic": (
        "real_vs_synthetic_s1.json",
        ("synthetic", "converged"),
    ),
    "descent_holdout_s1": ("optimizer_split.json", ("holdout_dexycb", "improved")),
    "descent_cross_subject": (
        "optimizer_split.json",
        ("cross_subject_dexycb_s2", "improved"),
    ),
    "median_cost_reduction_holdout_s1": (
        "optimizer_split.json",
        ("holdout_dexycb", "median_improvement_pct"),
    ),
    "median_cost_reduction_cross_subject": (
        "optimizer_split.json",
        ("cross_subject_dexycb_s2", "median_improvement_pct"),
    ),
    "selected_step_size": ("optimizer_split.json", ("selected_step_size",)),
    "noise_scale": ("optimizer_split.json", ("noise_scale",)),
    "bc_controlled_s1": ("bc_downstream_dexycb.json", ("__controlled_pct__",)),
    "bc_controlled_s2": ("bc_downstream_dexycb_s2.json", ("__controlled_pct__",)),
    "bc_controlled_synthetic": ("bc_downstream.json", ("__naive_pct__",)),
    "bc_max_err_raw_s1": ("bc_downstream_dexycb.json", ("raw", "max_err")),
    "bc_max_err_control_s1": (
        "bc_downstream_dexycb.json",
        ("resampled_control", "max_err"),
    ),
    "bc_max_err_opt_s1": ("bc_downstream_dexycb.json", ("optimized", "max_err")),
    "bc_max_err_raw_s2": ("bc_downstream_dexycb_s2.json", ("raw", "max_err")),
    "bc_max_err_control_s2": (
        "bc_downstream_dexycb_s2.json",
        ("resampled_control", "max_err"),
    ),
    "bc_max_err_opt_s2": ("bc_downstream_dexycb_s2.json", ("optimized", "max_err")),
    "granularity_transition_s1": (
        "bc_split_granularity_dexycb.json",
        ("results", "transition", "advantage_pct", "mean"),
    ),
    "granularity_trajectory_s1": (
        "bc_split_granularity_dexycb.json",
        ("results", "trajectory", "advantage_pct", "mean"),
    ),
    "granularity_prefix_s1": (
        "bc_split_granularity_dexycb.json",
        ("results", "prefix", "advantage_pct", "mean"),
    ),
    "granularity_transition_s2": (
        "bc_split_granularity_dexycb_s2.json",
        ("results", "transition", "advantage_pct", "mean"),
    ),
    "granularity_trajectory_s2": (
        "bc_split_granularity_dexycb_s2.json",
        ("results", "trajectory", "advantage_pct", "mean"),
    ),
    "granularity_prefix_s2": (
        "bc_split_granularity_dexycb_s2.json",
        ("results", "prefix", "advantage_pct", "mean"),
    ),
    "granularity_prefix_synthetic": (
        "bc_split_granularity_synthetic.json",
        ("results", "prefix", "advantage_pct", "mean"),
    ),
    "executable_raw_s1": ("feasibility_dexycb.json", ("arms", "raw", "feasible")),
    "executable_control_s1": (
        "feasibility_dexycb.json",
        ("arms", "resampled", "feasible"),
    ),
    "executable_opt_s1": ("feasibility_dexycb.json", ("arms", "optimized", "feasible")),
    "executable_raw_s2": ("feasibility_dexycb_s2.json", ("arms", "raw", "feasible")),
    "executable_control_s2": (
        "feasibility_dexycb_s2.json",
        ("arms", "resampled", "feasible"),
    ),
    "executable_opt_s2": (
        "feasibility_dexycb_s2.json",
        ("arms", "optimized", "feasible"),
    ),
    "executable_raw_synthetic": (
        "feasibility_synthetic.json",
        ("arms", "raw", "feasible"),
    ),
    "executable_opt_synthetic": (
        "feasibility_synthetic.json",
        ("arms", "optimized", "feasible"),
    ),
    "prefix_trivial_baseline_s1": (
        "bc_split_granularity_dexycb.json",
        ("results", "prefix", "trivial_baseline_mse", "resampled_control"),
    ),
    "prefix_control_score_s1": (
        "bc_split_granularity_dexycb.json",
        ("results", "prefix", "resampled_control", "mean"),
    ),
    "prefix_opt_score_s1": (
        "bc_split_granularity_dexycb.json",
        ("results", "prefix", "optimized", "mean"),
    ),
    "prefix_trivial_baseline_synthetic": (
        "bc_split_granularity_synthetic.json",
        ("results", "prefix", "trivial_baseline_mse", "raw"),
    ),
    "prefix_control_score_synthetic": (
        "bc_split_granularity_synthetic.json",
        ("results", "prefix", "raw", "mean"),
    ),
    "tail_speed_ratio_control_s1": (
        "tail_extrapolation_dexycb.json",
        ("arms", "resampled_control", "terminal_speed", "final_over_mid_median"),
    ),
    "tail_speed_ratio_optimized_s1": (
        "tail_extrapolation_dexycb.json",
        ("arms", "optimized", "terminal_speed", "final_over_mid_median"),
    ),
    "tail_speed_ratio_raw_synthetic": (
        "tail_extrapolation_synthetic.json",
        ("arms", "raw", "terminal_speed", "final_over_mid_median"),
    ),
}

#: Sorted names, for tests that iterate rather than naming each one.
FIGURE_NAMES = tuple(sorted(SOURCES))


class MissingArtifact(FileNotFoundError):
    """Raised when a recorded result a published figure depends on is absent."""


def _dig(payload: dict, path: tuple[str, ...]) -> float:
    """Walk a nested key path, failing loudly on anything unexpected.

    A figure that cannot be located must not resolve to a default, because the
    whole point of this module is that a documentation number is traceable to
    the artifact that produced it.
    """
    node: object = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise MissingArtifact(
                f"key {key!r} not found while resolving {'.'.join(path)}"
            )
        node = node[key]
    if isinstance(node, (dict, list)):
        raise MissingArtifact(f"{'.'.join(path)} resolved to a {type(node).__name__}")
    return float(node)


def _require(payload: dict, *keys: str) -> float:
    """Read a numeric leaf, reporting absence as `MissingArtifact`.

    The derived figures below index into the payload directly, which raises a
    bare `KeyError` on an artifact that lost an arm. That is inconsistent with
    every other resolution path here and gives the caller a much less
    actionable message, so they go through the same guard.
    """
    for key in keys:
        if key not in payload:
            raise MissingArtifact(f"key {key!r} not found in {sorted(payload)}")
        payload = payload[key]
    if not isinstance(payload, (int, float)):
        raise MissingArtifact(f"{'.'.join(keys)} is not numeric: {payload!r}")
    return float(payload)


def _controlled_pct(payload: dict) -> float:
    """The optimizer's advantage over the resampled control arm.

    The published controlled figure is derived from two recorded arms rather
    than stored, so it is computed here from the artifact to keep one
    definition of the arithmetic.
    """
    optimized = _require(payload, "optimized", "mse")
    control = _require(payload, "resampled_control", "mse")
    return 100.0 * (1.0 - optimized / control)


def _naive_pct(payload: dict) -> float:
    return 100.0 * (
        1.0 - _require(payload, "optimized", "mse") / _require(payload, "raw", "mse")
    )


_DERIVED = {"__controlled_pct__": _controlled_pct, "__naive_pct__": _naive_pct}


def load_figures(results_dir: Path | str = RESULTS_DIR) -> dict[str, float]:
    """Read every published figure from the recorded artifacts.

    Raises `MissingArtifact` if any artifact or key is absent. It does not
    return partial results, because a snapshot built from a partial read is
    exactly how a figure goes missing from the documentation unnoticed.
    """
    root = Path(results_dir)
    cache: dict[str, dict] = {}
    out: dict[str, float] = {}
    for name in FIGURE_NAMES:
        filename, path = SOURCES[name]
        if filename not in cache:
            target = root / filename
            if not target.exists():
                raise MissingArtifact(f"{target} does not exist")
            cache[filename] = json.loads(target.read_text())
        payload = cache[filename]
        derive = _DERIVED.get(path[0])
        if derive is not None:
            out[name] = derive(payload)
        else:
            out[name] = _dig(payload, path)
    return out


def load_snapshot(path: Path | str = SNAPSHOT_PATH) -> dict[str, float]:
    """Read the committed snapshot of the published figures."""
    target = Path(path)
    if not target.exists():
        raise MissingArtifact(f"{target} does not exist")
    return {k: float(v) for k, v in json.loads(target.read_text())["figures"].items()}


def refresh_published_figures(
    results_dir: Path | str = RESULTS_DIR,
    path: Path | str = SNAPSHOT_PATH,
) -> dict[str, float]:
    """Regenerate the committed snapshot from the live artifacts.

    Run this after re-running the experiment scripts, and commit the result
    alongside the documentation change. `test_published_results.py` fails if the
    snapshot and the artifacts disagree, so forgetting to refresh it is a test
    failure rather than a stale number in the README.
    """
    figures = load_figures(results_dir)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "sources": {k: list(v) for k, v in sorted(SOURCES.items())},
                "figures": {k: figures[k] for k in FIGURE_NAMES},
            },
            indent=2,
        )
        + "\n"
    )
    return figures
