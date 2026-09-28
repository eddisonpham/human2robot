"""MuJoCo replay acceptance: optimized demos replay without divergence.

Spec section 18 item 9: the existing MuJoCo simulator can replay the
optimized trajectory.

This uses `evaluation.feasibility`, which replays open-loop against the real
actuator limits. It replaced a `validate_open_loop_replay` helper that was named
for this check but compared `a_demo` arithmetic against stub bounds of
`zeros(22)` and `ones(22)` and never opened the simulator, so the acceptance
criterion was never actually exercised. The numbers asserted here are measured,
not assumed: on the 5-seed synthetic set the raw demos drift 0.25 to 0.30 rad
and the optimized ones 0.11, so the bounds below carry roughly a factor of two
of headroom.
"""

import numpy as np
import pytest

from human2robot.data.allegro_demos import generate_synthetic_demos, load_demo_npz
from human2robot.envs.allegro import AllegroPickupEnv
from human2robot.evaluation.feasibility import (
    DEFAULT_MAX_DRIFT,
    score_trajectory,
    summarize,
)
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


@pytest.fixture(scope="module", name="env")
def env_fixture():
    env = AllegroPickupEnv(max_episode_steps=200)
    yield env
    env.close()


def _score(env, paths):
    return [
        score_trajectory(np.load(path)["q"].astype(float), env, trajectory_id=path.stem)
        for path in paths
    ]


def test_optimized_replay_stays_bounded(replay_paths, env):
    _, opt_paths = replay_paths
    reports = _score(env, opt_paths)
    for report in reports:
        assert not report.diverged, f"{report.trajectory_id} diverged"
        assert np.isfinite(report.tracking_drift_max)
        assert report.tracking_drift_max < 0.20, (
            f"{report.trajectory_id} drift {report.tracking_drift_max:.4f}"
        )


def test_optimized_demos_track_better_than_the_raw_demos(replay_paths, env):
    """The optimizer must make the trajectory more executable, not just smoother.

    This is the Phase 5 acceptance criterion in the form that can fail: replay
    the optimized trajectory and compare against the unoptimized one it came
    from. Asserting only that the optimized replay terminates would pass for a
    trajectory the optimizer had made worse.
    """
    raw_paths, opt_paths = replay_paths
    raw_reports = {r.trajectory_id: r for r in _score(env, raw_paths)}
    for report in _score(env, opt_paths):
        source = report.trajectory_id.removesuffix("_opt")
        assert source in raw_reports, f"no raw counterpart for {report.trajectory_id}"
        before = raw_reports[source]
        assert report.tracking_drift_max < before.tracking_drift_max, (
            f"{source}: optimization raised drift "
            f"{before.tracking_drift_max:.4f} -> {report.tracking_drift_max:.4f}"
        )


def test_every_replayed_demo_is_feasible(replay_paths, env):
    raw_paths, opt_paths = replay_paths
    for label, paths in (("raw", raw_paths), ("opt", opt_paths)):
        summary = summarize(_score(env, paths))
        assert summary["diverged"] == 0, label
        assert summary["out_of_bounds_frames"] == 0, label
        assert summary["feasible"] == summary["count"], label
        assert summary["tracking_drift_max_worst"] < DEFAULT_MAX_DRIFT, label


def test_raw_demos_replay_without_divergence(replay_paths, env):
    """Phase 4 acceptance: retargeted trajectories replay open-loop.

    Per agents/09 Phase 4, at least 90 percent of processed clips must replay
    without joint-limit violations or divergence. Measured on this set all 5 do,
    so the threshold is not being passed by a margin of one clip.
    """
    raw_paths, _ = replay_paths
    reports = _score(env, raw_paths)
    assert reports, "no demos replayed"
    ok = sum(1 for r in reports if not r.diverged and r.tracking_drift_max < 1.0)
    assert ok / len(reports) >= 0.9, f"only {ok}/{len(reports)} replayed cleanly"


def test_optimized_demos_are_schema_valid(replay_paths):
    _, opt_paths = replay_paths
    for path in opt_paths:
        demo = load_demo_npz(path)
        assert demo.q.shape[1] == 22
        assert np.isfinite(demo.q).all()
