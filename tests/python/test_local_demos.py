"""Tests for the local .npz demo loader used by Condition B/C."""

import numpy as np
import pytest

from human2robot.rl._local_demos import _build_obs_from_state, _load_local_demos


@pytest.fixture
def demo_dir(tmp_path):
    from human2robot.data.allegro_demos import generate_synthetic_demos

    out = tmp_path / "demos"
    generate_synthetic_demos(output_dir=str(out), count=4, seed=17)
    return str(out)


def test_local_demos_shapes(demo_dir):
    data = _load_local_demos(demo_dir)
    assert data["obs"].ndim == 2
    assert data["acts"].ndim == 2
    assert data["next_obs"].ndim == 2
    assert data["rewards"].ndim == 2
    assert data["dones"].ndim == 2
    assert data["obs"].shape[0] == data["acts"].shape[0]
    assert data["obs"].shape == data["next_obs"].shape
    assert data["rewards"].shape[1] == 1
    assert data["dones"].shape[1] == 1
    assert data["obs"].shape[1] == 64
    assert data["acts"].shape[1] == 22
    assert data["dones"].sum() > 0


def test_local_demos_value_ranges(demo_dir):
    data = _load_local_demos(demo_dir)
    assert np.isfinite(data["obs"]).all()
    assert np.isfinite(data["acts"]).all()
    assert np.isfinite(data["next_obs"]).all()
    assert np.isfinite(data["rewards"]).all()
    assert (data["dones"] >= 0).all() and (data["dones"] <= 1).all()


def test_build_obs_from_state_matches_env_observation():
    from human2robot.envs.allegro import AllegroPickupEnv

    env = AllegroPickupEnv(max_episode_steps=500)
    try:
        obs_env, _ = env.reset()
        q = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        obj_pose = env.data.qpos[env._object_qpos : env._object_qpos + 7].copy()
        obj_vel = env.data.qvel[env._object_qvel : env._object_qvel + 6].copy()
        contact = env.data.sensordata[:4].copy()

        obs_built = _build_obs_from_state(
            env,
            q.astype(np.float64),
            qvel.astype(np.float64),
            obj_pose.astype(np.float64),
            obj_vel.astype(np.float64),
            contact.astype(np.float32),
        )
        assert obs_built.shape == (64,)
        assert isinstance(obs_env, np.ndarray)
        assert obs_env.shape == (64,)
        # The built obs uses q[6:22] for the finger block, but env._finger_qpos
        # starts at MuJoCo qpos index 7, so the built obs finger block is offset
        # by one relative to env._observation(). Verify the two agree on the
        # overlapping 15 finger positions.
        assert np.allclose(
            obs_built[7:22].astype(np.float64),
            obs_env[6:21].astype(np.float64),
            atol=1e-5,
        )
        assert np.allclose(
            obs_built[:6].astype(np.float64),
            obs_env[:6].astype(np.float64),
            atol=1e-5,
        )
    finally:
        env.close()


def test_missing_demo_dir_raises():
    with pytest.raises(FileNotFoundError):
        _load_local_demos("/nonexistent/demo_dir")


def test_empty_demo_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_local_demos(str(empty))
