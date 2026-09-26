"""Phase 6 acceptance: dynamics ensemble quality and multi-step rollout.

Per agents/09 Phase 6:
1. One-step prediction beats a zero-residual (mean-prediction) baseline.
2. Multi-step rollout errors at k in {1,5,10,20,50} are reported and don't
   diverge catastrophically at k=50.
3. The residual variant's error is lower than the black-box variant's under
   domain-randomized evaluation (this is the first quantitative signal that
   "physics structure" is doing something).
"""

import numpy as np

from human2robot.dynamics.ensemble import DynamicsEnsemble, DynamicsMetrics


def _make_transition_batch(
    state_dim: int,
    action_dim: int,
    n: int,
    seed: int,
    physics_delta_scale: float = 0.01,
) -> dict:
    """Generate synthetic transitions with a known dynamics structure.

    The "true" dynamics: next_state = state + f(s,a) + physics_delta
    where physics_delta is a known function of state and physics_delta
    is a small deterministic term (gravity-like bias on some dims).
    The black-box model must learn f(s,a), the residual model gets physics_delta
    for free and only learns the residual f(s,a).
    """
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, state_dim)).astype(np.float32) * 0.1
    actions = rng.uniform(-1, 1, size=(n, action_dim)).astype(np.float32)
    # True dynamics: small nonlinear function of state + action + physics bias
    physics_deltas = np.zeros_like(states)
    physics_deltas[:, 0] = 0.001 * states[:, 1] ** 2  # gravity-like on dim 0
    physics_deltas[:, 1] = -0.001 * states[:, 0]  # coupling on dim 1
    f_sa = (
        np.tanh(states * 0.5) * 0.01 + np.mean(actions, axis=1, keepdims=True) * 0.005
    )
    next_states = states + f_sa + physics_deltas
    rewards = (
        np.sum(actions * states[:, :action_dim], axis=1, keepdims=True).astype(
            np.float32
        )
        * 0.01
    )
    return {
        "states": states,
        "actions": actions,
        "next_states": next_states,
        "physics_deltas": physics_deltas,
        "rewards": rewards,
    }


def test_one_step_beats_zero_residual_baseline():
    """One-step MSE of ensemble < MSE of predicting next_state = state.

    Uses the zero-residual baseline (predicting no change) as the
    comparison point. The ensemble must beat this trivial baseline.
    """
    state_dim, action_dim, n = 10, 4, 500
    data = _make_transition_batch(state_dim, action_dim, n, seed=0)
    model = DynamicsEnsemble(state_dim, action_dim, ensemble_size=3, mode="blackbox")
    model.fit(
        data["states"],
        data["actions"],
        data["next_states"],
        epochs=20,
        batch_size=64,
        seed=0,
    )
    metrics = model.evaluate(
        data["states"],
        data["actions"],
        data["next_states"],
        horizons=(1,),
    )
    # Zero-residual baseline: predict next = state (delta = 0)
    zero_mse = float(np.mean((data["next_states"] - data["states"]) ** 2))
    assert metrics.one_step_mse < zero_mse, (
        f"ensemble one-step MSE {metrics.one_step_mse:.6f} >= "
        f"zero-residual baseline {zero_mse:.6f}"
    )


def test_multi_step_rollout_does_not_diverge_catastrophically():
    """Rollout errors at k=1,5,10,20,50 are reported and k=50 isn't NaN/inf."""
    state_dim, action_dim, n = 8, 3, 300
    data = _make_transition_batch(state_dim, action_dim, n, seed=1)
    model = DynamicsEnsemble(state_dim, action_dim, ensemble_size=2, mode="blackbox")
    model.fit(
        data["states"],
        data["actions"],
        data["next_states"],
        epochs=15,
        batch_size=64,
        seed=0,
    )
    horizons = (1, 5, 10, 20, 50)
    metrics = model.evaluate(
        data["states"],
        data["actions"],
        data["next_states"],
        horizons=horizons,
    )
    assert isinstance(metrics, DynamicsMetrics)
    assert len(metrics.rollout_errors) == len(horizons)
    for k, err in metrics.rollout_errors.items():
        assert np.isfinite(err), f"rollout error at k={k} is not finite: {err}"
        assert err >= 0, f"rollout error at k={k} is negative: {err}"
    # k=50 error should be larger than k=1 (rollout accumulates error) but
    # not catastrophically large (e.g. not 10000x the one-step error)
    e1 = metrics.rollout_errors[1]
    e50 = metrics.rollout_errors[50]
    assert e50 > e1, f"k=50 error {e50} should be > k=1 error {e1}"
    assert e50 < 10000 * e1, (
        f"k=50 error {e50:.6f} is catastrophically large vs k=1 {e1:.6f}"
    )


def test_residual_mode_beats_blackbox_with_physics_deltas():
    """Residual variant error < blackbox variant error when physics_deltas provided.

    This is the key signal that physics structure helps: the residual model
    gets the known physics term for free and only needs to learn the small
    residual, while the blackbox model must learn everything from scratch.
    """
    state_dim, action_dim, n = 8, 3, 500
    data = _make_transition_batch(state_dim, action_dim, n, seed=2)
    horizons = (1, 5, 10)

    # Blackbox model: learns everything from scratch
    bb = DynamicsEnsemble(state_dim, action_dim, ensemble_size=3, mode="blackbox")
    bb.fit(
        data["states"],
        data["actions"],
        data["next_states"],
        epochs=20,
        batch_size=64,
        seed=0,
    )
    bb_metrics = bb.evaluate(
        data["states"],
        data["actions"],
        data["next_states"],
        horizons=horizons,
    )

    # Residual model: given physics_deltas, learns only the residual
    res = DynamicsEnsemble(state_dim, action_dim, ensemble_size=3, mode="residual")
    res.fit(
        data["states"],
        data["actions"],
        data["next_states"],
        epochs=20,
        batch_size=64,
        physics_deltas=data["physics_deltas"],
        rewards=data["rewards"],
        seed=0,
    )
    res_metrics = res.evaluate(
        data["states"],
        data["actions"],
        data["next_states"],
        horizons=horizons,
        physics_deltas=data["physics_deltas"],
    )

    for k in horizons:
        bb_err = bb_metrics.rollout_errors[k]
        res_err = res_metrics.rollout_errors[k]
        assert res_err < bb_err, (
            f"residual mode error {res_err:.6f} >= blackbox error {bb_err:.6f} "
            f"at k={k}; physics structure is not helping"
        )


def test_ensemble_predict_rewards_when_fitted():
    """Reward prediction works after fitting with rewards."""
    state_dim, action_dim, n = 6, 2, 200
    data = _make_transition_batch(state_dim, action_dim, n, seed=3)
    model = DynamicsEnsemble(state_dim, action_dim, ensemble_size=2, mode="blackbox")
    model.fit(
        data["states"],
        data["actions"],
        data["next_states"],
        epochs=10,
        batch_size=64,
        rewards=data["rewards"],
        seed=0,
    )
    pred_rewards = model.predict_rewards(data["states"][:10], data["actions"][:10])
    assert pred_rewards.shape == (10,)
    assert np.isfinite(pred_rewards).all()


def test_predict_requires_fit():
    """Predict raises RuntimeError if called before fit."""
    model = DynamicsEnsemble(4, 2, ensemble_size=1, mode="blackbox")
    try:
        model.predict(np.zeros((1, 4)), np.zeros((1, 2)))
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "fit" in str(e).lower()


def test_residual_mode_requires_physics_deltas_at_fit():
    """Residual mode raises at fit time if physics_deltas not provided."""
    model = DynamicsEnsemble(4, 2, ensemble_size=1, mode="residual")
    try:
        model.fit(
            np.zeros((10, 4)),
            np.zeros((10, 2)),
            np.zeros((10, 4)),
            epochs=1,
        )
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "physics_deltas" in str(e)
