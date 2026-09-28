"""Why does the optimizer hurt on the tail-extrapolation split but help elsewhere?

`scripts/check_bc_split_granularity.py` reports that training on optimized
trajectories and testing on the held-out tail beats the resampled control on
the transition and whole-trajectory splits, and loses on the prefix split. On
real DexYCB retargets the prefix advantage is negative on every seed, while on
synthetic data it is positive on every seed. This script asks whether that is
because the optimizer changes the *statistical distribution* of the tail, or
because it changes the *difficulty* of it.

Four measurements, per arm:

1. `tail_delta_rms`: root-mean-square of the state change across the held-out
   tail. If the optimizer damps the tail, this falls, and a damped tail is a
   different regression problem from a lively one.
2. `terminal_error`: distance between the arm's final configuration and the
   resampled control's. The optimizer is given the whole trajectory including
   its tail, so a nonzero terminal error means it moved the endpoint.
3. `tail_vs_body_delta_rms`: the same statistic over the training portion, so
   the tail can be compared against the trajectory it was cut from.
4. `tracking_rms`: how far the arm departs from the trajectory it was
   optimizing toward. A large value means the optimizer was pulling away from
   the input, which would be visible to a model trained on the input's tail.

The interesting comparison is the *ratio* of tail to body motion. If the
optimizer shrinks the tail much more than the body, the tail stops looking like
the rest of the trajectory and a model trained on optimized prefixes is being
asked to predict a distribution it never saw.

Usage:

    uv run python scripts/diagnose_tail_extrapolation.py [--set dexycb]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_downstream_bc import DEMO_SETS, load_positions  # noqa: E402

#: The holdout fraction the prefix split uses. Matched here so the tail
#: statistics describe the same slice the split tests.
PREFIX_FRACTION = 0.1


def split_lengths(trajectories: list[np.ndarray], fraction: float) -> list[int]:
    """The tail length each trajectory contributes, as `split_prefix` cuts it."""
    return [max(2, int(fraction * len(t))) for t in trajectories]


def delta_rms(trajectories: list[np.ndarray], lengths: list[int] | None) -> dict:
    """RMS state change per step over a whole set or over its tails only.

    The base coordinates are pinned at zero in every arm, so including them
    would divide the signal by a constant without adding information. The
    measurement is over the 16 finger joints.
    """
    deltas = [np.diff(t[:, 6:], axis=0) for t in trajectories]
    if lengths is not None:
        deltas = [d[len(d) - n :] for d, n in zip(deltas, lengths, strict=True)]
    stacked = np.concatenate(deltas)
    if not np.isfinite(stacked).all():
        return {"rms": float("inf"), "n_frames": int(len(stacked))}
    return {
        "rms": float(np.sqrt(np.mean(stacked**2))),
        "n_frames": int(len(stacked)),
    }


def deviation_profile(arms: dict[str, list[np.ndarray]], baseline: str) -> dict:
    """How far each arm departs from the baseline as a function of position.

    Normalized position, 0 at the start and 1 at the end, so trajectories of
    different lengths pool into one profile. The prefix split holds out the
    final 10 percent, so if deviation peaks there the held-out region is
    precisely the region the optimizer changed most, and a model trained on the
    optimized body is being asked to predict the part of the trajectory that
    received the least input information.
    """
    n_bins = 20
    if baseline not in arms:
        return {}
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for traj, ref in zip(arms[baseline], arms["optimized"], strict=True):
        n = min(len(traj), len(ref))
        err = np.linalg.norm(traj[:n, 6:] - ref[:n, 6:], axis=1)
        idx = np.minimum((np.arange(n) / n * n_bins).astype(int), n_bins - 1)
        np.add.at(sums, idx, err)
        np.add.at(counts, idx, 1)
    per_bin = np.divide(sums, counts, out=np.zeros(n_bins), where=counts > 0)
    return {
        "bins": n_bins,
        "mean_rad": per_bin.tolist(),
        "last_decile_mean_rad": float(per_bin[-2:].mean()),
        "first_decile_mean_rad": float(per_bin[:2].mean()),
        "peak_position": float(np.argmax(per_bin) / n_bins),
        "peak_rad": float(per_bin.max()),
    }


def terminal_speed(trajectories: list[np.ndarray]) -> dict:
    """How much motion is left in the final frames.

    A hand reaching the end of a reach decelerates to a stop, so its last
    frames carry very little state change. If that is what the real data does
    and the synthetic data does not, the prefix split is a regime change on one
    set and an ordinary interpolation on the other, which is a different
    experiment rather than a harder version of the same one.
    """
    final = np.array(
        [
            np.linalg.norm(np.diff(t[-3:, 6:], axis=0), axis=1).mean()
            for t in trajectories
        ]
    )
    middle = np.array(
        [
            np.linalg.norm(
                np.diff(t[len(t) // 2 - 2 : len(t) // 2 + 1, 6:], axis=0), axis=1
            ).mean()
            for t in trajectories
        ]
    )
    ratio = final / np.maximum(middle, 1e-12)
    return {
        "final_step_rad": float(final.mean()),
        "mid_trajectory_step_rad": float(middle.mean()),
        "final_over_mid_mean": float(ratio.mean()),
        "final_over_mid_median": float(np.median(ratio)),
        "fraction_below_0.5": float((ratio < 0.5).mean()),
    }


def zero_prediction_mse(trajectories: list[np.ndarray], fraction: float) -> dict:
    """MSE of predicting no state change at all, over the prefix-split tail.

    This is the floor a regressor cannot beat by predicting motion: it is the
    score of the model that says "the hand has stopped". If one arm sits at
    that floor, its tail is trivially predictable and its split score measures
    how quiet the tail is rather than how well the motion extrapolates.
    """
    tail = [t[int((1.0 - fraction) * len(t)) :] for t in trajectories]
    errors = [np.diff(t[:, 6:], axis=0) for t in tail]
    usable = [e for e in errors if len(e) > 0]
    if not usable:
        return {"mse": float("nan"), "transitions": 0}
    stacked = np.concatenate(usable)
    return {
        "mse": float(np.mean(stacked**2)),
        "transitions": int(len(stacked)),
        "rms": float(np.sqrt(np.mean(stacked**2))),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def cut_alignment(trajectories: list[np.ndarray], lengths: list[int]) -> dict:
    """How well the first tail step continues the last body step.

    A smooth trajectory enters its tail moving the way it left its body, so the
    cosine between the last body delta and the first tail delta is near 1. A
    value near 0 or below means the tail starts a new motion rather than
    continuing one, which is exactly the regime where a model trained on the
    body cannot predict the tail.
    """
    align, step_ratio = [], []
    for traj, n in zip(trajectories, lengths, strict=True):
        d = np.diff(traj[:, 6:], axis=0)
        last_body = d[-n - 1]
        first_tail = d[-n]
        align.append(_cosine(last_body, first_tail))
        nb, nt = np.linalg.norm(last_body), np.linalg.norm(first_tail)
        if nb > 1e-12:
            step_ratio.append(float(nt / nb))
    align = np.array(align, dtype=float)
    align = align[np.isfinite(align)]
    return {
        "mean": float(align.mean()) if align.size else float("nan"),
        "median": float(np.median(align)) if align.size else float("nan"),
        "fraction_below_0.5": float((align < 0.5).mean())
        if align.size
        else float("nan"),
        "first_tail_step_over_last_body_step": float(np.mean(step_ratio))
        if step_ratio
        else float("nan"),
    }


def accel_rms(trajectories: list[np.ndarray], lengths: list[int] | None) -> float:
    """RMS second difference, over the whole set or over its tails only."""
    seconds = []
    for i, traj in enumerate(trajectories):
        d2 = np.diff(traj[:, 6:], n=2, axis=0)
        if lengths is not None:
            d2 = d2[len(d2) - lengths[i] :]
        seconds.append(d2)
    stacked = np.concatenate(seconds)
    if not np.isfinite(stacked).all():
        return float("inf")
    return float(np.sqrt(np.mean(stacked**2)))


def terminal_errors(arms: dict[str, list[np.ndarray]], baseline: str) -> dict:
    """Distance from each arm's final configuration to the baseline's."""
    ref = arms[baseline]
    out = {}
    for name, trajs in arms.items():
        if name == baseline or len(trajs) != len(ref):
            out[name] = {"rms": float("nan"), "n": len(trajs)}
            continue
        end = np.stack([t[-1, 6:] for t in trajs])
        ref_end = np.stack([t[-1, 6:] for t in ref])
        out[name] = {
            "rms": float(np.sqrt(np.mean((end - ref_end) ** 2))),
            "max": float(np.abs(end - ref_end).max()),
            "n": len(trajs),
        }
    return out


def diagnose(name: str) -> dict:
    """Measure tail and body statistics for every arm of one data set."""
    raw_dir, opt_dir, _ = DEMO_SETS[name]
    arms = {
        "raw": load_positions(raw_dir),
        "optimized": load_positions(opt_dir, suffix="_opt"),
    }
    # Synthetic data is already at the control rate, so the pipeline that
    # produces it writes no resampled control arm. The comparison is against
    # raw there, and pretending otherwise would invent an arm.
    if sorted(opt_dir.glob("*_resampled.npz")):
        arms = {
            "raw": arms["raw"],
            "resampled_control": load_positions(opt_dir, suffix="_resampled"),
            "optimized": arms["optimized"],
        }
    lengths = split_lengths(arms["raw"], PREFIX_FRACTION)

    stats: dict[str, dict] = {}
    for arm, trajs in arms.items():
        body = delta_rms(trajs, None)
        tail = delta_rms(trajs, lengths)
        stats[arm] = {
            "body_delta_rms": body["rms"],
            "tail_delta_rms": tail["rms"],
            "tail_over_body": tail["rms"] / body["rms"]
            if body["rms"]
            else float("inf"),
            "body_frames": body["n_frames"],
            "tail_frames": tail["n_frames"],
            "cut_alignment": cut_alignment(trajs, lengths),
            "tail_accel_rms": accel_rms(trajs, lengths),
            "body_accel_rms": accel_rms(trajs, None),
            "terminal_speed": terminal_speed(trajs),
            "zero_prediction_mse": zero_prediction_mse(trajs, PREFIX_FRACTION),
        }

    baseline = "resampled_control" if "resampled_control" in arms else "raw"
    return {
        "set": name,
        "prefix_fraction": PREFIX_FRACTION,
        "baseline": baseline,
        "tail_lengths": {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": float(np.mean(lengths)),
        },
        "trajectories": len(arms["raw"]),
        "arms": stats,
        "terminal_error_vs_" + baseline: terminal_errors(arms, baseline),
        "deviation_profile_optimized_vs_" + baseline: deviation_profile(arms, baseline),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    name = "dexycb"
    if "--set" in argv:
        name = argv[argv.index("--set") + 1]
    if name not in DEMO_SETS:
        print(
            f"unknown demo set {name!r}; choose from {sorted(DEMO_SETS)}",
            file=sys.stderr,
        )
        return 1

    report = diagnose(name)
    print(f"tail lengths (frames): {report['tail_lengths']}")
    print(f"baseline for terminal error: {report['baseline']}")
    ends = report["terminal_error_vs_" + report["baseline"]]
    print(
        f"\n{'arm':<20}{'cut align':>11}{'align<0.5':>11}"
        f"{'tail accel':>12}{'body accel':>12}{'terminal rms':>14}"
    )
    for arm, s in report["arms"].items():
        end = ends[arm]
        ca = s["cut_alignment"]
        term = "n/a" if np.isnan(end["rms"]) else f"{end['rms']:.4f}"
        print(
            f"{arm:<20}{ca['mean']:>11.3f}{ca['fraction_below_0.5']:>11.3f}"
            f"{s['tail_accel_rms']:>12.5f}{s['body_accel_rms']:>12.5f}{term:>14}"
        )
    print(
        f"\n{'arm':<20}{'final step':>12}{'mid step':>11}{'final/mid':>11}{'<0.5':>8}"
    )
    for arm, s in report["arms"].items():
        ts = s["terminal_speed"]
        print(
            f"{arm:<20}{ts['final_step_rad']:>12.5f}{ts['mid_trajectory_step_rad']:>11.5f}"
            f"{ts['final_over_mid_median']:>11.3f}{ts['fraction_below_0.5']:>8.2f}"
        )
    print("\nprefix-tail predict-nothing baseline (the score of 'the hand stopped'):")
    for arm, s in report["arms"].items():
        z = s["zero_prediction_mse"]
        print(
            f"  {arm:<20}mse {z['mse']:.3e}  rms {z['rms']:.5f}  "
            f"({z['transitions']} test transitions)"
        )
    profile = report["deviation_profile_optimized_vs_" + report["baseline"]]
    if profile:
        print(
            f"\noptimized deviation from {report['baseline']} by position: "
            f"first decile {profile['first_decile_mean_rad']:.4f} rad, "
            f"last decile {profile['last_decile_mean_rad']:.4f} rad, "
            f"peak {profile['peak_rad']:.4f} at {profile['peak_position']:.0%}"
        )
    print("\nmotion scale (delta rms, body and tail):")
    print(f"{'arm':<20}{'body rms':>11}{'tail rms':>11}{'tail/body':>11}")
    for arm, s in report["arms"].items():
        print(
            f"{arm:<20}{s['body_delta_rms']:>11.5f}{s['tail_delta_rms']:>11.5f}"
            f"{s['tail_over_body']:>11.3f}"
        )

    out = (
        Path("results/trajectory_optimization")
        / f"tail_extrapolation_{name.replace('-', '_')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
