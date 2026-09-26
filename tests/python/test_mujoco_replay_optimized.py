"""MuJoCo replay acceptance: optimized demos replay without divergence.

Spec section 18 item 9: the existing MuJoCo simulator can replay the
optimized trajectory.
"""

import numpy as np
import pytest

from human2robot.data.allegro_demos import (
    generate_synthetic_demos,
    load_demo_npz,
    validate_open_loop_replay,
)
from human2robot.envs.allegro import AllegroPickupEnv
from human2robot.optimization import (
    OptimizationConfig,
    optimize_demo_directory,
)


@pytest.fixture(scope="module", name="replay_paths")
def replay_paths_fixture(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    opt = tmp_path_factory.mktemp("opt")
    generate_synthetic_demos(output_dir=raw, count=5, seed=3)
    config = OptimizationConfig(seed=0, max_iterations=80)
    optimize_demo_directory(raw, opt, config)
    return sorted(raw.glob("*.npz")), sorted(opt.glob("*_opt.npz"))


def _replay_drift(env: AllegroPickupEnv, q_seq: np.ndarray) -> float:
    """Open-loop position-target replay; returns max joint-space drift."""
    env.reset(seed=0)
    actuator_low = env._actuator_low
    actuator_high = env._actuator_high
    max_drift = 0.0
    for target in q_seq:
        action = np.zeros(22, dtype=np.float32)
        n_finger = min(target.shape[0] - 6, 16)
        if n_finger <= 0:
            env.step(np.zeros(22, dtype=np.float32))
            continue
        ft = target[6 : 6 + n_finger]
        action[6 : 6 + n_finger] = np.clip(
            (ft - actuator_low[:n_finger])
            / np.maximum(actuator_high[:n_finger] - actuator_low[:n_finger], 1e-6)
            * 2.0
            - 1.0,
            -1.0,
            1.0,
        )
        env.step(action)
        if not np.isfinite(env.data.qpos).all():
            return float("inf")
        achieved = env.data.qpos[env._finger_qpos][:n_finger]
        drift = float(np.abs(achieved - ft).max())
        max_drift = max(max_drift, drift)
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


def test_open_loop_replay_acceptance(replay_paths):
    """Phase 5 acceptance: optimized demos replay open-loop without divergence.

    Per agents/09 Phase 5: replaying a demo's a_demo sequence open-loop
    through the Phase 2 environment reproduces the retargeted q trajectory
    within a small tracking error.
    """
    _, opt_paths = replay_paths
    report = validate_open_loop_replay(opt_paths, tolerance=0.5, sample=5)
    assert report["pass"] is True, (
        f"open-loop replay failed: mean_error={report['mean_error']:.4f}"
    )


def test_raw_demos_replay_without_divergence(replay_paths):
    """Phase 4 acceptance: raw synthetic demos replay open-loop without exploding.

    Per agents/09 Phase 4: retargeted trajectories replay open-loop in the
    Phase 2 environment without joint-limit violations or divergence for
    at least 90% of processed clips.
    """
    raw_paths, _ = replay_paths
    env = AllegroPickupEnv(max_episode_steps=200)
    try:
        drift_ok = 0
        drift_bad = 0
        for path in raw_paths:
            demo = load_demo_npz(path)
            q22 = demo.q  # 22-D: base(6)+fingers(16)
            env.reset(seed=0)
            drift = _replay_drift(env, q22)
            if np.isfinite(drift) and drift < 1.0:
                drift_ok += 1
            else:
                drift_bad += 1
        total = drift_ok + drift_bad
        assert total > 0, "no demos replayed"
        assert drift_ok / total >= 0.9, (
            f"only {drift_ok}/{total} demos replayed without divergence"
        )
    finally:
        env.close()


def test_optimized_demos_are_schema_valid(replay_paths):
    _, opt_paths = replay_paths
    for path in opt_paths:
        demo = load_demo_npz(path)
        assert demo.q.shape[1] == 22
        assert np.isfinite(demo.q).all()
