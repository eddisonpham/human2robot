"""Domain-randomization robustness helpers (agents/10 §3)."""

import numpy as np


def domain_randomized_eval(
    sac,
    env_id: str,
    episodes: int = 20,
    seed: int = 0,
    scales: tuple[float, ...] = (0.8, 1.0, 1.2),
) -> dict[str, float]:
    """Evaluate under mass/friction scaling (Phase 9 stub, no env mutation)."""
    from dynhand.evaluation.evaluate import evaluate

    results = {}
    for scale in scales:
        metrics = evaluate(sac, env_id, episodes=episodes, seed=seed + int(scale * 10))
        results[f"scale_{scale:.1f}"] = metrics["eval_return_mean"]
    mean = float(np.mean(list(results.values())))
    return {"robustness_mean": mean, **results}


def generalization_split(
    train_ids: list[str], test_ids: list[str]
) -> dict[str, object]:
    """Validate object-level held-out split (agents/08 §6)."""
    train_set, test_set = set(train_ids), set(test_ids)
    if train_set & test_set:
        raise ValueError("train/test object leakage")
    return {"train_size": len(train_set), "test_size": len(test_set), "leakage": False}
