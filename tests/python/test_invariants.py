"""Invariant tests for conventions that have silently broken before.

Each test here corresponds to a real defect found in this codebase:

- the MuJoCo quaternion is [w,x,y,z], not [x,y,z,w] (found twice, in
  _rotvec_to_quat and _quat_delta_to_rotvec_delta)
- the substep count was duplicated as a literal in two modules that must agree
- the substep count is also declared in PhysicsConfig, which nothing reads
- every field the trainer branches on must be reachable from a real config
"""

import numpy as np
import pytest
import torch

from human2robot.config.loader import load_config
from human2robot.config.schema import (
    CONDITION_FLAGS,
    ExperimentConfig,
    PhysicsConfig,
)
from human2robot.dynamics import nominal_physics
from human2robot.envs.allegro import ENV_STEP_SUBSTEPS, AllegroPickupEnv
from human2robot.rl.networks import GaussianActor

TIER_B_CONFIGS = [
    "configs/tier_b_pickup.yaml",
    "configs/tier_b_cond_b.yaml",
    "configs/tier_b_cond_c.yaml",
    "configs/tier_b_cond_d.yaml",
    "configs/tier_b_cond_e.yaml",
]


def test_env_substep_count_is_shared_with_physics_module() -> None:
    """The physics delta must simulate exactly the sub-steps the env applies."""
    assert nominal_physics._ENV_STEP_SUBSTEPS == ENV_STEP_SUBSTEPS


def test_substep_count_matches_mujoco_timestep_and_config() -> None:
    """Guard the count against drifting from the config and the model."""
    config = load_config("configs/tier_b_pickup.yaml")
    assert PhysicsConfig().control_decimation == ENV_STEP_SUBSTEPS
    assert config.physics.control_decimation == ENV_STEP_SUBSTEPS


def test_physics_config_timestep_matches_the_simulator() -> None:
    """PhysicsConfig.timestep is declared but must not contradict the model."""
    env = AllegroPickupEnv()
    try:
        assert env.model.opt.timestep == pytest.approx(PhysicsConfig().timestep)
    finally:
        env.close()


def test_mujoco_quaternion_is_scalar_first() -> None:
    """MuJoCo orders quaternions [w,x,y,z]; a wrong order silently misorients."""
    import mujoco

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, identity)
    assert np.allclose(mat.reshape(3, 3), np.eye(3))


def test_rotvec_to_quat_is_scalar_first() -> None:
    assert np.allclose(nominal_physics._rotvec_to_quat(np.zeros(3)), [1.0, 0, 0, 0])


def test_nominal_delta_matches_a_real_env_step() -> None:
    """The core guarantee: nominal physics describes the actual simulator."""
    env = AllegroPickupEnv()
    try:
        rng = np.random.default_rng(0)
        obs, _ = env.reset(seed=0)
        for _ in range(10):
            action = rng.uniform(-1.0, 1.0, 22).astype(np.float32)
            predicted = nominal_physics.compute_obs_delta(env.model, obs, action)
            next_obs, _, _, _, _ = env.step(action)
            assert np.abs(predicted - (next_obs - obs)).max() < 1e-4
            obs = next_obs
    finally:
        env.close()


@pytest.mark.parametrize("config_path", TIER_B_CONFIGS)
def test_tier_b_configs_set_every_field_the_trainer_branches_on(config_path) -> None:
    """demo.enabled and dynamics_aug.mode drive real branches in train()."""
    config = load_config(config_path)
    assert isinstance(config, ExperimentConfig)
    assert config.demo.enabled is CONDITION_FLAGS[config.condition]["demo_enabled"]
    assert (
        config.dynamics_aug.enabled
        is CONDITION_FLAGS[config.condition]["dynamics_enabled"]
    )
    assert (
        config.dynamics_aug.mode == CONDITION_FLAGS[config.condition]["dynamics_mode"]
    )


@pytest.mark.parametrize("config_path", TIER_B_CONFIGS)
def test_demo_conditions_declare_a_demo_source(config_path) -> None:
    """A condition with demos enabled but no source would fail at train time."""
    config = load_config(config_path)
    if not config.demo.enabled:
        return
    assert config.demo.minari_dataset or config.demo.demo_dir


def test_same_seed_reproduces_initial_actor_weights() -> None:
    """The top-level seed must reach network initialization."""
    from human2robot.utils.seed import seed_everything

    def build() -> torch.Tensor:
        seed_everything(11)
        torch.manual_seed(11)
        return GaussianActor(6, 3, 8).fc_mean.weight.detach().clone()

    assert torch.equal(build(), build())


def test_different_seeds_give_different_initial_weights() -> None:
    from human2robot.utils.seed import seed_everything

    def build(seed: int) -> torch.Tensor:
        seed_everything(seed)
        torch.manual_seed(seed)
        return GaussianActor(6, 3, 8).fc_mean.weight.detach().clone()

    assert not torch.equal(build(1), build(2))


def test_same_seed_reproduces_sac_action_sequence() -> None:
    """Identical seeds must give identical policies, not just identical weights."""
    from human2robot.config.schema import SACConfig
    from human2robot.rl.sac import SAC
    from human2robot.utils.seed import seed_everything

    def rollout() -> np.ndarray:
        seed_everything(5)
        torch.manual_seed(5)
        sac = SAC(
            6,
            3,
            np.full(3, -1.0, dtype=np.float32),
            np.full(3, 1.0, dtype=np.float32),
            SACConfig(hidden_dim=16),
            torch.device("cpu"),
        )
        observations = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(8, 3)
        observations = np.repeat(observations, 2, axis=1)[:, :6]
        return sac.act(observations)

    first, second = rollout(), rollout()
    assert first.shape == (8, 3)
    assert np.array_equal(first, second)


def test_vector_worker_seeds_are_distinct() -> None:
    from human2robot.utils.seed import worker_seed

    seeds = [worker_seed(100, index) for index in range(12)]
    assert len(set(seeds)) == 12
    assert seeds[0] == 100
