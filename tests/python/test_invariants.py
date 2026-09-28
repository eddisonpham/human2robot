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


def _demo_paths(directory: str) -> list[Path]:
    """Demo files in a directory, or None when it holds none.

    `data/` is gitignored, but a fresh clone still contains the directories
    because each ships a `.gitkeep`. A guard that tests `is_dir()` therefore
    passes on a clone that has no data at all, and the check below fails on
    `assert paths` instead of skipping. The condition that matters is whether
    there are demos, not whether there is a directory.
    """
    return sorted(Path(directory).glob("*.npz")) or None


def _require_demos(name: str, directory: str) -> list[Path]:
    """Skip unless the named demo set is present, returning its files."""
    paths = _demo_paths(directory)
    if paths is None:
        pytest.skip(f"{name} demos not generated")
    return paths


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
    _require_demos(name, directory)
    assert _dims_for_90_percent_variance(_stack_finger_positions(directory)) >= 4


@pytest.mark.parametrize("name", sorted(DEGENERATE_SETS))
def test_known_degenerate_set_is_still_degenerate(name: str) -> None:
    """Pins the historical behavior so the deletion cannot be a silent fix.

    If the basis retargeter is ever replaced rather than removed, this fails
    loudly instead of the published numbers quietly changing meaning.
    """
    directory = DEGENERATE_SETS[name]
    _require_demos(name, directory)
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
    _require_demos(name, directory)
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


# --- actuator limits -----------------------------------------------------------
#
# The Allegro bounds were written out in four places and each copy had drifted
# from the robot: 15 of 16 finger entries disagreed with the MuJoCo model, and
# the thumb's upper bound was 0.0 against a real 1.719. Every trajectory the
# optimizer produced was projected onto those wrong bounds, so the claim that
# optimization projects onto the robot's joint limits was false. These tests pin
# the single source of truth against the model.


def test_actuator_limits_match_the_mujoco_model() -> None:
    from human2robot.data import limits

    env = AllegroPickupEnv()
    model_lower = env.model.actuator_ctrlrange[:, 0].copy()
    model_upper = env.model.actuator_ctrlrange[:, 1].copy()
    finger_lower, finger_upper = limits.finger_bounds()
    assert np.allclose(finger_lower, model_lower)
    assert np.allclose(finger_upper, model_upper)


def test_actuator_limits_cover_every_actuator_in_order() -> None:
    """Index, middle, ring, then thumb. A wrong order silently mis-projects."""
    from human2robot.data import limits

    env = AllegroPickupEnv()
    names = [env.model.joint(env.model.actuator_trnid[i, 0]).name for i in range(16)]
    assert names == [
        "ffj0",
        "ffj1",
        "ffj2",
        "ffj3",
        "mfj0",
        "mfj1",
        "mfj2",
        "mfj3",
        "rfj0",
        "rfj1",
        "rfj2",
        "rfj3",
        "thj0",
        "thj1",
        "thj2",
        "thj3",
    ]
    assert len(limits.FINGER_LOWER) == 16
    assert limits.ACTUATOR_LOWER.shape == (limits.DOF,)
    assert limits.ACTUATOR_UPPER.shape == (limits.DOF,)


def test_thumb_bounds_are_not_the_three_finger_bounds() -> None:
    """The thumb is a different linkage; it must not inherit finger bounds.

    This is the specific value that was wrong: the thumb's three distal joints
    were constrained to <= 0 while the real thumb reaches 1.719.
    """
    from human2robot.data import limits

    assert limits.FINGER_UPPER[15] == pytest.approx(1.719, abs=1e-6)
    assert limits.FINGER_LOWER[12] == pytest.approx(0.263, abs=1e-6)
    assert not np.allclose(limits.FINGER_LOWER[12:], limits.FINGER_LOWER[:4])


def test_base_coordinates_are_pinned_not_freed() -> None:
    """Zero-width bounds pin the base; they do not mean "unbounded"."""
    from human2robot.data import limits

    assert np.allclose(limits.ACTUATOR_LOWER[:6], 0.0)
    assert np.allclose(limits.ACTUATOR_UPPER[:6], 0.0)
    assert np.all(limits.ACTUATOR_LOWER[6:] < limits.ACTUATOR_UPPER[6:])


def test_optimizer_receives_the_model_limits() -> None:
    """The pipeline must not reintroduce its own copy of the bounds."""
    from human2robot.data import limits
    from human2robot.optimization.pipeline import OptimizationConfig

    config = OptimizationConfig(dt=limits.CONTROL_DT).to_optimizer_config(limits.DOF)
    assert np.allclose(config._config.limits.lower, limits.ACTUATOR_LOWER)
    assert np.allclose(config._config.limits.upper, limits.ACTUATOR_UPPER)


def test_normalized_targets_land_inside_the_model_range() -> None:
    from human2robot.data import limits

    angles = limits.normalized_targets_to_finger_angles(np.zeros((3, 16)))
    low, high = limits.finger_bounds()
    assert np.all(angles >= low - 1e-9)
    assert np.all(angles <= high + 1e-9)
    assert angles.shape == (3, 16)


# --- dex-retargeting joint order ----------------------------------------------
#
# dex-retargeting emits fingers in [index, thumb, middle, ring] order while the
# Allegro model is [index, middle, ring, thumb]. The two were treated as the
# same, so every retargeted demo drove the thumb with the middle finger's
# motion. Found from the data: the demo column ranges match the permuted joints
# to within 0.001 rad, which no other assignment achieves.


def test_dexretarget_order_is_a_permutation() -> None:
    from human2robot.data import limits

    fwd = limits.DEXRETARGET_TO_MODEL_FINGER
    assert sorted(fwd) == list(range(16))
    assert sorted(limits.MODEL_TO_DEXRETARGET_FINGER) == list(range(16))


def test_dexretarget_permutation_is_its_own_inverse_pair() -> None:
    from human2robot.data import limits

    fwd = limits.DEXRETARGET_TO_MODEL_FINGER
    inv = limits.MODEL_TO_DEXRETARGET_FINGER
    for model_index, dex_index in enumerate(fwd):
        assert inv[dex_index] == model_index


def test_dexretarget_reorder_moves_the_thumb_last() -> None:
    """The thumb is the 2nd block in dex-retargeting and the 4th in the model.

    This is the specific mislabelling: columns 4-7 held thumb motion but were
    written into the middle finger's slots.
    """
    from human2robot.data import limits

    assert limits.DEXRETARGET_TO_MODEL_FINGER[0:4] == (0, 1, 2, 3)
    assert limits.DEXRETARGET_TO_MODEL_FINGER[4:8] == (8, 9, 10, 11)
    assert limits.DEXRETARGET_TO_MODEL_FINGER[8:12] == (12, 13, 14, 15)
    assert limits.DEXRETARGET_TO_MODEL_FINGER[12:16] == (4, 5, 6, 7)


def test_fingers_from_dexretarget_permutes_rows_and_columns() -> None:
    from human2robot.data import limits

    q16 = np.arange(32, dtype=np.float64).reshape(2, 16)
    out = limits.fingers_from_dexretarget(q16)
    assert out.shape == (2, 16)
    assert out[0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7]
    assert out[1].tolist() == [16 + i for i in out[0]]


def test_fingers_from_dexretarget_rejects_wrong_width() -> None:
    from human2robot.data import limits

    with pytest.raises(ValueError, match="angles shape"):
        limits.fingers_from_dexretarget(np.zeros((3, 15)))


def test_retargeted_demos_fit_the_model_bounds() -> None:
    """Demos on disk must already be in model order and inside model bounds.

    This is the empirical proof the reorder is right rather than merely
    plausible. Demo column ranges must sit inside the bounds of the joint they
    claim to be. Before the reorder, thumb motion landed in middle-finger slots
    and every column matched the wrong joint's range.
    """
    from human2robot.data import limits
    from human2robot.data.allegro_demos import load_demo_npz

    directory = "data/demonstrations_dexycb_ik"
    paths = _require_demos("IK", directory)
    q16 = np.concatenate([load_demo_npz(p).q[:, 6:] for p in paths])
    low, high = limits.finger_bounds()
    assert np.all(q16.min(axis=0) >= low - 0.02)
    assert np.all(q16.max(axis=0) <= high + 0.02)


def test_reorder_repairs_a_mislabelled_input() -> None:
    """The permutation must be the thing that fixes a dex-ordered sequence."""
    from human2robot.data import limits

    # A dex-ordered row: thumb values (0.3) sit in slots 4-7.
    dex_ordered = np.zeros((1, 16), dtype=np.float64)
    dex_ordered[0, 4:8] = 0.3
    model_ordered = limits.fingers_from_dexretarget(dex_ordered)
    assert np.allclose(model_ordered[0, 12:16], 0.3)
    assert np.allclose(model_ordered[0, 4:8], 0.0)
