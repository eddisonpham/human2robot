"""Domain-randomization robustness helpers (agents/10 §3)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np

#: Scales applied to the model's dynamics when the caller does not choose.
DEFAULT_SCALES: tuple[float, ...] = (0.8, 1.0, 1.2)

#: The model arrays scaled together. Motor strength and joint damping are
#: grouped with mass and friction because they are the parameters a real robot
#: differs by between units, and scaling only the inertias would test a
#: perturbation no hardware exhibits.
_SCALED_FIELDS = ("body_mass", "geom_friction", "dof_damping", "actuator_gainprm")


@contextmanager
def domain_randomization(model, scale: float) -> Iterator[None]:
    """Temporarily scale the model's dynamics by `scale`, then restore them.

    The restore is unconditional. A caller that leaves the model scaled would
    silently carry the perturbation into the next condition, which turns a
    robustness sweep into an ordering effect.

    Raises `AttributeError` if `model` is not a MuJoCo model, so a non-MuJoCo
    environment fails loudly instead of reporting a robustness number nothing
    was perturbed to produce.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    for field in _SCALED_FIELDS:
        if not hasattr(model, field):
            raise AttributeError(
                f"model has no {field!r}: it is not a MuJoCo model to randomize"
            )
    original = {
        field: np.array(getattr(model, field), copy=True) for field in _SCALED_FIELDS
    }
    try:
        for field, values in original.items():
            getattr(model, field)[:] = values * scale
        yield
    finally:
        for field, values in original.items():
            getattr(model, field)[:] = values


def domain_randomized_eval(
    sac,
    env_id: str,
    episodes: int = 20,
    seed: int = 0,
    scales: Sequence[float] = DEFAULT_SCALES,
) -> dict[str, float]:
    """Evaluate one policy under several dynamics perturbations.

    Every condition is a scaling of the *same* model and every condition is
    evaluated from the same initial states, so the spread across scales is
    attributable to the perturbation alone. Seeding each condition differently
    would confound the two and is easy to do by accident, because the natural
    way to avoid reusing a seed is to derive one from the scale.

    A policy that is genuinely robust returns similar values across scales; one
    that has memorized the nominal dynamics does not.

    `env_id` must resolve to an environment exposing a MuJoCo `model`. There is
    nothing to randomize in a black-box environment, and pretending otherwise is
    what this function used to do.
    """
    from human2robot.evaluation.evaluate import build_single_env, evaluate_env

    if not scales:
        raise ValueError("scales must be non-empty")
    env = build_single_env(env_id)
    try:
        model = getattr(env, "model", None)
        if model is None:
            raise ValueError(f"{env_id} exposes no MuJoCo model to randomize")
        results: dict[str, float] = {}
        for scale in scales:
            with domain_randomization(model, scale):
                metrics = evaluate_env(sac, env, episodes=episodes, seed=seed)
            results[f"scale_{scale:.1f}"] = metrics["eval_return_mean"]
    finally:
        env.close()

    values = list(results.values())
    spread = float(max(values) - min(values))
    return {
        "robustness_mean": float(np.mean(values)),
        "robustness_spread": spread,
        **results,
    }


def generalization_split(
    train_ids: list[str], test_ids: list[str]
) -> dict[str, object]:
    """Validate object-level held-out split (agents/08 §6)."""
    train_set, test_set = set(train_ids), set(test_ids)
    if train_set & test_set:
        raise ValueError("train/test object leakage")
    return {"train_size": len(train_set), "test_size": len(test_set), "leakage": False}
