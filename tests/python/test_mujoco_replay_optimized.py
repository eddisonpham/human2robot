"""MuJoCo replay acceptance: optimized demos replay without divergence.

Spec section 18 item 9: the existing MuJoCo simulator can replay the
optimized trajectory.
"""

import numpy as np
import pytest

from human2robot.data.allegro_demos import generate_synthetic_demos
from human2robot.envs.allegro import AllegroPickupEnv
from human2robot.optimization import (
    OptimizationConfig,
    optimize_demo_directory,
)


@pytest.fixture(scope="module", name="replay_paths")
def replay_paths_fixture(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    opt = tmp_path_factory.mktemp("opt")
    generate_synthetic_demos(output_dir=raw, count=2, seed=3)
    config = OptimizationConfig(seed=0, max_iterations=80)
    optimize_demo_directory(raw, opt, config)
    return sorted(raw.glob("*.npz")), sorted(opt.glob("*_opt.npz"))


def _replay_drift(env: AllegroPickupEnv, q_seq: np.ndarray) -> float:
    """Open-loop position-target replay; returns max joint-space drift."""
    env.reset(seed=0)
    actuator_low = env._actuator_low
    actuator_high = env._actuator_high
    center = 0.5 * (actuator_low + actuator_high)
    span = 0.5 * (actuator_high - actuator_low)
    max_drift = 0.0
    for target in q_seq:
        action = np.zeros(22, dtype=np.float32)
        action[6:] = np.clip(
            (target[6:22] - center) / np.maximum(span, 1e-6), -1.0, 1.0
        )
        env.step(action)
        achieved = env.data.qpos[env._finger_qpos]
        max_drift = max(max_drift, float(np.abs(achieved - target[6:22]).max()))
    return max_drift


def test_optimized_replay_stays_bounded(replay_paths, tmp_path):
    raw_paths, opt_paths = replay_paths
    env = AllegroPickupEnv()
    try:
        for path in opt_paths:
            data = np.load(path)
            drift = _replay_drift(env, data["q"].astype(float))
            assert np.isfinite(drift)
            assert drift < 1.0
    finally:
        env.close()


def test_optimized_demos_are_schema_valid(replay_paths):
    from human2robot.data.allegro_demos import load_demo_npz

    _, opt_paths = replay_paths
    for path in opt_paths:
        demo = load_demo_npz(path)
        assert demo.q.shape[1] == 22
        assert np.isfinite(demo.q).all()
