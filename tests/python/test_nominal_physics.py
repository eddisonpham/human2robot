"""Tests for the nominal physics delta provider.

The central guarantee is that the delta the residual ensemble is trained
against actually describes a real environment step. That equivalence is what
was silently broken, so it is pinned here against the live env.
"""

import mujoco
import numpy as np
import pytest

from human2robot.dynamics.nominal_physics import (
    _BASE_VEL_SCALE,
    _action_to_ctrl,
    _apply_action,
    _rotvec_to_quat,
    _set_state_from_obs,
    compute_obs_delta,
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
    errors = [
        np.abs(compute_obs_delta(env.model, obs, action) - (nxt - obs)).max()
        for obs, action, nxt in _rollout(env, 25)
    ]
    assert max(errors) < 1e-4


def test_compute_obs_delta_is_finite_and_correctly_shaped(env) -> None:
    obs, _ = env.reset(seed=3)
    delta = compute_obs_delta(env.model, obs, np.zeros(22, dtype=np.float32))
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
    delta = compute_obs_delta(
        env.model, obs, np.zeros((1, 16), np.float64), ctrl_from_action=False
    )
    assert np.all(np.isfinite(delta))


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
    _apply_action(env.model, data, np.ones((1, 22)), np.zeros((1, env.model.nu)))
    assert np.allclose(data.qvel[0:6], _BASE_VEL_SCALE)


def test_action_to_ctrl_maps_into_ctrlrange(env) -> None:
    model = env.model
    high = _action_to_ctrl(model, np.ones((1, 22)))
    assert high.shape == (1, model.nu)
    assert np.allclose(high[0], model.actuator_ctrlrange[:, 1])
    low = _action_to_ctrl(model, -np.ones((1, 22)))
    assert np.allclose(low[0], model.actuator_ctrlrange[:, 0])


def test_action_to_ctrl_honors_explicit_bounds(env) -> None:
    ctrl = _action_to_ctrl(
        env.model, np.zeros((1, 22)), np.full(16, -0.5), np.full(16, 0.5)
    )
    assert np.allclose(ctrl[0], 0.0)
    assert _action_to_ctrl(env.model, np.zeros(22)).shape == (1, 16)


def test_action_to_ctrl_clips_out_of_range_actions(env) -> None:
    ctrl = _action_to_ctrl(env.model, np.full((1, 22), 5.0))
    assert np.all(ctrl <= env.model.actuator_ctrlrange[:, 1] + 1e-12)


def test_rotvec_to_quat_uses_mujoco_w_first_order() -> None:
    """MuJoCo stores quaternions as [w,x,y,z]; identity must be [1,0,0,0]."""
    assert np.allclose(_rotvec_to_quat(np.zeros(3)), [1.0, 0.0, 0.0, 0.0])
    quarter_turn = _rotvec_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    assert np.isclose(quarter_turn[0], np.cos(np.pi / 4))
    assert np.isclose(np.linalg.norm(quarter_turn[1:4]), np.sin(np.pi / 4))


def test_rotvec_to_quat_round_trips_through_rotation_matrix() -> None:
    rotvec = np.array([0.3, -0.2, 0.5])
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, _rotvec_to_quat(rotvec))
    assert np.allclose(rotation_vector(mat.reshape(3, 3)), rotvec, atol=1e-9)


def test_rotvec_to_quat_accepts_batched_input() -> None:
    assert _rotvec_to_quat(np.zeros((2, 3))).shape == (4,)
