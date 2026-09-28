"""Invariant tests for conventions that have silently broken before.

Each test here corresponds to a real defect found in this codebase:

- the MuJoCo quaternion is [w,x,y,z], not [x,y,z,w] (found twice, in
  _rotvec_to_quat and _quat_delta_to_rotvec_delta)
- the substep count was duplicated as a literal in two modules that must agree
- the substep count is also declared in PhysicsConfig, which nothing reads
- every field the trainer branches on must be reachable from a real config
"""

import sys
from pathlib import Path
from types import SimpleNamespace

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

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


# --- retargeting dimensionality ------------------------------------------------
#
# A published result in this repository measured a 16-DoF hand while the data
# path under test carried 90 percent of its variance in one dimension. The
# retargeter summed MANO joint magnitudes into a single per-finger curl scalar
# and broadcast it to all four joints, so it was not a retargeting at all. Every
# number derived from it described that scalar, and the smoothness numbers in
# particular were inflated because there was no structure to preserve.

DEMO_SETS = {
    "dexycb_ik": "data/demonstrations_dexycb_ik",
    "dexycb_ik_s2": "data/demonstrations_dexycb_ik_s2",
    "synthetic": "data/demonstrations",
}
DEGENERATE_SETS = {"dexycb": "data/demonstrations_dexycb"}


def _stack_finger_positions(directory: str) -> np.ndarray:
    from human2robot.data.allegro_demos import load_demo_npz

    paths = sorted(Path(directory).glob("*.npz"))
    assert paths, f"no demos in {directory}"
    return np.concatenate([load_demo_npz(p).q[:, 6:] for p in paths])


def _dims_for_90_percent_variance(q: np.ndarray) -> int:
    centered = q - q.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    share = singular**2 / (singular**2).sum()
    return int(np.searchsorted(np.cumsum(share), 0.90) + 1)


@pytest.mark.parametrize("name", sorted(DEMO_SETS))
def test_retargeted_demos_are_not_degenerate(name: str) -> None:
    """A real retargeting must not collapse onto a single degree of freedom.

    The 1-DOF retargeter this pins put 90 percent of the variance in one
    dimension of 16. A genuine hand retargeting uses most of them.
    """
    directory = DEMO_SETS[name]
    if not Path(directory).is_dir():
        pytest.skip(f"{name} demos not generated")
    assert _dims_for_90_percent_variance(_stack_finger_positions(directory)) >= 4


@pytest.mark.parametrize("name", sorted(DEGENERATE_SETS))
def test_known_degenerate_set_is_still_degenerate(name: str) -> None:
    """Pins the historical behavior so the deletion cannot be a silent fix.

    If the basis retargeter is ever replaced rather than removed, this fails
    loudly instead of the published numbers quietly changing meaning.
    """
    directory = DEGENERATE_SETS[name]
    if not Path(directory).is_dir():
        pytest.skip(f"{name} demos not generated")
    assert _dims_for_90_percent_variance(_stack_finger_positions(directory)) == 1


@pytest.mark.parametrize("name", sorted(DEMO_SETS))
def test_finger_joints_articulate_independently(name: str) -> None:
    """Adjacent joints of a finger must move by a meaningful amount.

    The 1-DOF retargeter drove all four joints of a finger with one scalar plus
    a 0.15 spread, so adjacent joints differed by ~0.05 rad. Real retargeting
    articulates the chain. Note the Allegro's distal joints move *less* than
    its proximal ones under normal flexion, so the test is on magnitude, not on
    the direction of the gradient.
    """
    directory = DEMO_SETS[name]
    if not Path(directory).is_dir():
        pytest.skip(f"{name} demos not generated")
    q = _stack_finger_positions(directory)
    adjacent = np.mean(
        [
            np.abs(q[:, f * 4 + 2] - q[:, f * 4 + 1]).mean()
            + np.abs(q[:, f * 4 + 3] - q[:, f * 4 + 2]).mean()
            for f in range(4)
        ]
    )
    assert adjacent > 0.2


def test_retargeting_script_refuses_to_overwrite_the_default_subject() -> None:
    from retarget_dexycb_ik import _resolve_output_dir

    args = SimpleNamespace(output_dir=None, subject="20200709-subject-01")
    assert _resolve_output_dir(args).name == "demonstrations_dexycb_ik"
    second = SimpleNamespace(output_dir=None, subject="20200813-subject-02")
    with pytest.raises(SystemExit, match="--output-dir is required"):
        _resolve_output_dir(second)
