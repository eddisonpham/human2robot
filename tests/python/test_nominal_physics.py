"""Tests for the nominal physics delta provider.

The central guarantee is that the delta the residual ensemble is trained
against actually describes a real environment step.  That equivalence is what
was silently broken, so it is pinned here against the live env.
"""

import mujoco
import numpy as np
import pytest

from human2robot.dynamics.nominal_physics import (
    _BASE_VEL_SCALE,
    _action_to_ctrl,
    _apply_action,
    _initial_qpos_from_obs,
    _initial_qvel_from_obs,
    _physics_delta_to_obs_delta,
    _quat_delta_to_rotvec_delta,
    _rotvec_to_quat,
    _set_state_from_obs,
    compute_obs_delta,
    compute_physics_deltas,
    physics_delta_from_ensemble,
)
from human2robot.envs.allegro import AllegroPickupEnv
from human2robot.utils.rotation import rotation_vector


@pytest.fixture(scope="module")
def env():
    instance = AllegroPickupEnv()
    yield instance
    instance.close()


def _rollout(env, steps: int, seed: int = 0):
    """Yield (obs, action, next_obs) triples from random real env steps."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        action = rng.uniform(-1.0, 1.0, 22).astype(np.float32)
        next_obs, _, _, _, _ = env.step(action)
        yield obs, action, next_obs
        obs = next_obs


def test_compute_obs_delta_matches_real_env_step(env) -> None:
    """The nominal delta must reproduce the env's own observation change."""
    model = env.model
    errors = [
        np.abs(compute_obs_delta(model, obs, action) - (nxt - obs)).max()
        for obs, action, nxt in _rollout(env, 25)
    ]
    assert max(errors) < 1e-4


def test_compute_obs_delta_is_finite_and_correctly_shaped(env) -> None:
    obs, _ = env.reset(seed=3)
    action = np.zeros(22, dtype=np.float32)
    delta = compute_obs_delta(env.model, obs, action)
    assert delta.shape == (64,)
    assert delta.dtype == np.float32
    assert np.all(np.isfinite(delta))


def test_compute_obs_delta_accepts_batched_action(env) -> None:
    obs, _ = env.reset(seed=4)
    batch = np.zeros((1, 22), dtype=np.float32)
    assert compute_obs_delta(env.model, obs, batch).shape == (64,)


def test_compute_obs_delta_passes_ctrl_through_unchanged(env) -> None:
    """With ctrl_from_action disabled the raw action is the ctrl vector."""
    obs, _ = env.reset(seed=5)
    action = np.zeros((1, 16), dtype=np.float64)
    delta = compute_obs_delta(env.model, obs, action, ctrl_from_action=False)
    assert np.all(np.isfinite(delta))


def test_physics_deltas_round_trip_to_near_exact_obs_delta(env) -> None:
    """The compat qpos/qvel path must stay close to the direct obs delta."""
    model = env.model
    errors = []
    for obs, action, nxt in _rollout(env, 15, seed=6):
        converted = _physics_delta_to_obs_delta(
            model, obs, compute_physics_deltas(model, obs, action)
        )
        errors.append(np.abs(converted - (nxt - obs)).max())
    assert max(errors) < 5e-2


def test_physics_deltas_shape_matches_model(env) -> None:
    obs, _ = env.reset(seed=7)
    action = np.zeros(22, dtype=np.float32)
    delta = compute_physics_deltas(env.model, obs, action)
    assert delta.shape == (env.model.nq + env.model.nv,)
    assert delta.dtype == np.float32


def test_physics_deltas_accepts_batched_action_and_raw_ctrl(env) -> None:
    obs, _ = env.reset(seed=8)
    batched = compute_physics_deltas(env.model, obs, np.zeros((1, 22), np.float32))
    raw = compute_physics_deltas(
        env.model, obs, np.zeros((1, 16), np.float64), ctrl_from_action=False
    )
    assert batched.shape == raw.shape


def test_set_state_from_obs_restores_env_state(env) -> None:
    """Reconstruction from obs must match the env's own qpos/qvel."""
    obs, _ = env.reset(seed=9)
    data = mujoco.MjData(env.model)
    _set_state_from_obs(env.model, data, obs)
    assert np.abs(data.qpos - env.data.qpos).max() < 1e-6
    assert np.abs(data.qvel - env.data.qvel).max() < 1e-6
    assert data.time == 0.0


def test_set_state_from_obs_accepts_batched_obs(env) -> None:
    obs, _ = env.reset(seed=10)
    data = mujoco.MjData(env.model)
    _set_state_from_obs(env.model, data, obs.reshape(1, -1))
    assert np.abs(data.qpos - env.data.qpos).max() < 1e-6


def test_apply_action_overrides_base_velocity(env) -> None:
    data = mujoco.MjData(env.model)
    data.qvel[0:6] = 99.0
    action = np.ones((1, 22), dtype=np.float64)
    _apply_action(env.model, data, action, np.zeros((1, env.model.nu)))
    assert np.allclose(data.qvel[0:6], _BASE_VEL_SCALE)


def test_action_to_ctrl_maps_into_ctrlrange(env) -> None:
    model = env.model
    ctrl = _action_to_ctrl(model, np.ones((1, 22)))
    assert ctrl.shape == (1, model.nu)
    assert np.allclose(ctrl[0], model.actuator_ctrlrange[:, 1])
    low = _action_to_ctrl(model, -np.ones((1, 22)))
    assert np.allclose(low[0], model.actuator_ctrlrange[:, 0])


def test_action_to_ctrl_honors_explicit_bounds(env) -> None:
    ctrl = _action_to_ctrl(
        env.model, np.zeros((1, 22)), np.full(16, -0.5), np.full(16, 0.5)
    )
    assert np.allclose(ctrl[0], 0.0)
    assert _action_to_ctrl(env.model, np.zeros(22), None, None).shape == (1, 16)


def test_action_to_ctrl_clips_out_of_range_actions(env) -> None:
    ctrl = _action_to_ctrl(env.model, np.full((1, 22), 5.0))
    assert np.all(ctrl <= env.model.actuator_ctrlrange[:, 1] + 1e-12)


def test_rotvec_to_quat_uses_mujoco_w_first_order() -> None:
    """MuJoCo stores quaternions as [w,x,y,z]; identity must be [1,0,0,0]."""
    assert np.allclose(_rotvec_to_quat(np.zeros(3)), [1.0, 0.0, 0.0, 0.0])
    quarter_turn = _rotvec_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    expected_w = np.cos(np.pi / 4)
    assert np.isclose(quarter_turn[0], expected_w)
    assert np.isclose(np.linalg.norm(quarter_turn[1:4]), np.sin(np.pi / 4))


def test_rotvec_to_quat_round_trips_through_rotation_matrix() -> None:
    rotvec = np.array([0.3, -0.2, 0.5])
    quat = _rotvec_to_quat(rotvec)
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    assert np.allclose(rotation_vector(mat.reshape(3, 3)), rotvec, atol=1e-9)


def test_rotvec_to_quat_accepts_batched_input() -> None:
    batched = _rotvec_to_quat(np.zeros((2, 3)))
    assert batched.shape == (4,)


def test_quat_delta_to_rotvec_delta_uses_vector_components() -> None:
    """A pure-x quaternion delta must produce a pure-x rotvec delta."""
    quat_delta = np.array([[0.0, 0.1, 0.0, 0.0]])
    assert np.allclose(_quat_delta_to_rotvec_delta(quat_delta, None), [0.2, 0.0, 0.0])


def test_quat_delta_to_rotvec_delta_keeps_batches_two_dimensional() -> None:
    quat_delta = np.zeros((2, 4))
    quat_delta[:, 1] = 0.05
    out = _quat_delta_to_rotvec_delta(quat_delta, None)
    assert out.shape == (2, 3)
    assert np.allclose(out[:, 0], 0.1)


def test_initial_qpos_and_qvel_match_set_state(env) -> None:
    obs, _ = env.reset(seed=11)
    data = mujoco.MjData(env.model)
    _set_state_from_obs(env.model, data, obs)
    assert np.allclose(_initial_qpos_from_obs(env.model, obs), data.qpos)
    assert np.allclose(_initial_qvel_from_obs(env.model, obs), data.qvel)


def test_initial_qpos_and_qvel_accept_batched_obs(env) -> None:
    obs, _ = env.reset(seed=12)
    assert _initial_qpos_from_obs(env.model, obs.reshape(1, -1)).shape == (
        env.model.nq,
    )
    assert _initial_qvel_from_obs(env.model, obs.reshape(1, -1)).shape == (
        env.model.nv,
    )


def test_physics_delta_to_obs_delta_leaves_contact_and_target_untouched(env) -> None:
    """Contact and target entries are not derivable, so their delta stays zero."""
    obs, _ = env.reset(seed=13)
    obs_delta = _physics_delta_to_obs_delta(
        env.model, obs, np.zeros(env.model.nq + env.model.nv)
    )
    assert np.allclose(obs_delta[56:64], 0.0)


def test_physics_delta_to_obs_delta_preserves_batch(env) -> None:
    obs, _ = env.reset(seed=14)
    size = env.model.nq + env.model.nv
    batched = _physics_delta_to_obs_delta(
        env.model, obs.reshape(1, -1), np.zeros((1, size))
    )
    assert batched.shape == (64,)


def test_physics_delta_from_ensemble_is_not_wired() -> None:
    with pytest.raises(NotImplementedError, match="not wired yet"):
        physics_delta_from_ensemble(None, None, np.zeros((1, 64)), np.zeros((1, 22)))
