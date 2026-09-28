"""The paths that were reachable but had no test.

Most of the suite covers behaviour. These cases cover the branches that only
fire when something is wrong: a misconfigured limit array, a singular rotation,
an observation space that is a dict rather than a box, a metrics stream that was
corrupted by a resume.

They are grouped by the failure they prevent rather than by module. The first
group is the important one. A singular rotation and a limit array of the wrong
length are both silent corruptions rather than crashes, and both have shipped in
this project before: the first produced observations up to 6.35e8, the second
let the optimizer record a `dof` it never enforced.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from human2robot.data.allegro_demos import load_demo_npz
from human2robot.data.limits import ACTUATOR_LOWER, ACTUATOR_UPPER, DOF
from human2robot.data.processing import smooth
from human2robot.evaluation import benchmark
from human2robot.evaluation.evaluate import evaluate_env
from human2robot.utils import rotation


class _ScalarPolicy:
    """Minimal policy so `evaluate_env` can be driven without a SAC object."""

    def __init__(self) -> None:
        self.action_dim = 2

    def predict(self, obs, deterministic: bool = True):
        return np.zeros(self.action_dim, dtype=np.float32), None


# --- silent-corruption guards ---------------------------------------------


def test_a_limit_array_of_the_wrong_length_is_rejected():
    """The optimizer must not accept limits it will silently mis-index.

    This is the check added after the facade was found storing a `dof` it never
    enforced. Without it a shorter bounds array is accepted and the C++ side
    reads past its own allocation.
    """
    from human2robot.cpp_bindings import OptimizerConfig

    for name, values in (
        ("lower", np.zeros(DOF - 1)),
        ("upper", np.zeros(DOF + 1)),
        ("max_velocity", np.zeros(3)),
        ("max_acceleration", np.zeros(7)),
    ):
        kwargs = {
            "lower": ACTUATOR_LOWER,
            "upper": ACTUATOR_UPPER,
            "max_velocity": np.full(DOF, 2.0),
            "max_acceleration": np.full(DOF, 100.0),
        }
        kwargs[name] = values
        with pytest.raises(ValueError, match=name):
            OptimizerConfig(dof=DOF, **kwargs)


def test_inverted_limit_bounds_are_rejected():
    """A lower bound above its upper bound is an empty, unmeetable interval.

    Equal bounds stay legal, because the six pinned base coordinates depend on
    it, so this asserts strictly-greater only.
    """
    from human2robot.cpp_bindings import OptimizerConfig

    lower = np.array(ACTUATOR_LOWER, dtype=float)
    upper = np.array(ACTUATOR_UPPER, dtype=float)
    lower[3], upper[3] = 1.0, -1.0
    with pytest.raises(ValueError, match="lower must not exceed upper"):
        OptimizerConfig(
            dof=DOF,
            lower=lower,
            upper=upper,
            max_velocity=np.full(DOF, 2.0),
            max_acceleration=np.full(DOF, 100.0),
        )

    pinned = np.zeros(DOF)
    OptimizerConfig(
        dof=DOF,
        lower=pinned,
        upper=pinned,
        max_velocity=np.full(DOF, 2.0),
        max_acceleration=np.full(DOF, 100.0),
    )


def test_a_near_degenerate_rotation_matrix_returns_identity_not_nan():
    """The guard that caused observations of 6.35e8 in the RL environment.

    Every branch of the matrix-to-quaternion conversion picks a denominator
    that can approach zero for a matrix on a boundary case. When they all do,
    the assembled quaternion has near-zero norm, and normalizing it would
    amplify numerical noise into an arbitrarily large or non-finite value.
    Returning identity is a real answer for a rotation this close to singular.
    """
    for scale in (0.0, 1e-14, 1e-9):
        m = np.array(
            [
                [1.0 + scale, 0.0, 0.0],
                [0.0, 1.0 - scale, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        q = rotation._matrix_to_quaternion(m)
        assert np.isfinite(q).all(), f"scale {scale} produced {q}"
        assert np.allclose(np.abs(q), [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_rotation_vector_is_bounded_at_180_degrees():
    """The public conversion stays finite where a skew division would not.

    The reported bug divided by `2 * sin(angle)`, which is singular at 180
    degrees. `rotation_vector` uses the largest-diagonal branch for exactly
    this reason, and the 180 degree case is the one that has to hold.
    """
    for angle in (0.0, np.pi / 2, np.pi - 1e-9, np.pi):
        axis = np.array([0.0, 0.0, 1.0])
        m = _axis_angle(axis, angle)
        v = rotation.rotation_vector(m)
        assert np.isfinite(v).all(), f"angle {angle} produced {v}"
        assert np.linalg.norm(v) <= np.pi + 1e-6


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix, built here to avoid a project dependency."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


# --- observation shape -----------------------------------------------------


def _dict_obs_env_id() -> str:
    """Register a minimal Dict-observation env and return its id.

    CartPole has a Box space, so it never reaches the flattening branch. A Dict
    space is what the Adroit environments used, and what made the missing
    wrapper fail deep inside the forward pass rather than at construction.
    """
    import gymnasium
    import gymnasium.envs.registration

    class _DictObsEnv(gymnasium.Env):
        observation_space = gymnasium.spaces.Dict(
            {
                "a": gymnasium.spaces.Box(-1, 1, (2,), dtype=np.float32),
                "b": gymnasium.spaces.Box(-1, 1, (3,), dtype=np.float32),
            }
        )
        action_space = gymnasium.spaces.Box(-1, 1, (1,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    env_id = "TestDictObs-v0"
    if env_id not in gymnasium.registry:
        gymnasium.register(id=env_id, entry_point=lambda: _DictObsEnv())
    return env_id


def test_the_vector_worker_flattens_a_dict_space():
    """A dict observation space must be flattened before the policy sees it.

    Both `build_single_env` and the vector-env worker do this independently.
    Skipping it feeds the network a dict where it expects a flat vector, which
    fails deep inside the forward pass rather than at construction. The worker
    had its own copy of the check, so it gets its own test.
    """
    import gymnasium

    from human2robot.envs.vec import _worker

    init = _worker(_dict_obs_env_id(), seed=3)
    env = init()
    try:
        assert not isinstance(env.observation_space, gymnasium.spaces.Dict)
        assert env.observation_space.shape == (5,)
    finally:
        env.close()


def test_build_single_env_flattens_a_dict_space():
    """The trainer's own construction path, separate from the worker's copy."""
    import gymnasium

    from human2robot.evaluation.evaluate import build_single_env

    env = build_single_env(_dict_obs_env_id())
    try:
        assert not isinstance(env.observation_space, gymnasium.spaces.Dict)
        assert env.observation_space.shape == (5,)
    finally:
        env.close()


# --- metrics stream integrity ---------------------------------------------


def test_a_blank_line_in_a_metrics_stream_is_skipped(tmp_path):
    """Blank lines occur in real metrics files and must not abort the read.

    `metrics.jsonl` is appended to by the trainer and by resume, so a partially
    written line or a stray newline is expected rather than exceptional.
    """
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "eval_return_mean": 0.5}),
                "",
                json.dumps({"step": 2, "eval_return_mean": 1.5}),
                "",
            ]
        )
    )
    steps, values = benchmark._series(path, "eval_return_mean")
    assert list(steps) == [1, 2]
    assert list(values) == [0.5, 1.5]


def test_a_missing_metric_is_reported_rather_than_returned_empty(tmp_path):
    """A metric that was never recorded must raise, not read as no data.

    Returning an empty series would let a benchmark silently compare conditions
    on nothing, which is the class of quiet failure this suite exists to catch.
    """
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"step": 1, "other": 1.0}))
    with pytest.raises(KeyError, match="missing"):
        benchmark._series(path, "eval_return_mean")


def test_benchmark_requires_at_least_one_run():
    with pytest.raises(ValueError, match="at least one run"):
        benchmark.summarize_runs([])


def test_plotting_refuses_an_unhealthy_metrics_stream(tmp_path):
    """A stream that failed the integrity audit must not be plotted.

    Resume used to write duplicate records, silently corrupting runs. The audit
    catches it, and this asserts the refusal actually happens rather than the
    check being advisory.
    """
    from human2robot.evaluation import audit, plots

    path = tmp_path / "metrics.jsonl"
    records = [{"step": i, "eval_return_mean": 1.0} for i in range(1, 11)]
    path.write_text("\n".join(json.dumps(r) for r in records))
    assert audit.audit_metrics(path).healthy

    duplicated = tmp_path / "dup.jsonl"
    duplicated.write_text("\n".join(json.dumps(r) for r in records + records))
    assert not audit.audit_metrics(duplicated).healthy
    with pytest.raises(ValueError, match="unhealthy"):
        plots.plot_learning_curves({"A": [duplicated]}, out_path=tmp_path / "p.png")


# --- input validation ------------------------------------------------------


def test_evaluate_env_rejects_a_non_positive_episode_count():
    """`episodes=0` would return an empty mean and look like a valid result."""
    import gymnasium

    env = gymnasium.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="episodes must be >= 1"):
            evaluate_env(_ScalarPolicy(), env, episodes=0, seed=0)
    finally:
        env.close()


def test_audit_summarize_reads_a_run_directory(tmp_path):
    """`summarize` is the run-directory entry point the audit CLI uses."""
    from human2robot.evaluation import audit

    records = [{"step": i, "eval_return_mean": float(i)} for i in range(1, 6)]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    report = audit.summarize(tmp_path)
    assert report.healthy
    assert report.eval_lines == 5
    assert report.malformed_lines == 0


def test_comparing_directories_with_no_matching_pairs_is_reported(tmp_path):
    """A typo in a directory name must raise, not compare nothing.

    Matching is by filename stem, so two unrelated directories produce zero
    pairs. Returning an empty comparison would read as "no difference".
    """
    from human2robot.evaluation import traj_compare

    (tmp_path / "raw").mkdir()
    (tmp_path / "opt").mkdir()
    (tmp_path / "raw" / "alpha.npz").write_bytes(b"")
    (tmp_path / "opt" / "unrelated_opt.npz").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="no raw/optimized pairs"):
        traj_compare.compare_directory(tmp_path / "raw", tmp_path / "opt")

    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="both raw and optimized"):
        traj_compare.compare_directory(tmp_path / "empty", tmp_path / "opt")


def test_a_batch_with_demonstrations_draws_from_both_sources():
    """The demonstration branch only runs when a demo buffer is supplied.

    With `demo_buffer=None` the whole batch comes from online, so the mixture
    arithmetic is never exercised. Asserting both shapes here is what keeps the
    ratios honest when a run is configured without demonstrations.
    """
    from human2robot.rl.schedules import sample_three

    class _Buffer:
        def __init__(self, tag: int) -> None:
            self.tag = tag

        def sample(self, n: int) -> dict[str, np.ndarray]:
            return {"obs": np.full((n, 2), self.tag, dtype=np.float32)}

    mixed = sample_three(
        _Buffer(0),
        _Buffer(1),
        None,
        batch_size=8,
        demo_ratio_value=0.5,
        synthetic_ratio=0.0,
    )
    assert mixed["obs"].shape[0] == 8
    assert set(mixed["obs"][:, 0].tolist()) == {0, 1}
    assert int((mixed["obs"][:, 0] == 1).sum()) == 4

    online_only = sample_three(
        _Buffer(0), None, None, batch_size=8, demo_ratio_value=0.5, synthetic_ratio=0.0
    )
    assert set(online_only["obs"][:, 0].tolist()) == {0}


def test_an_unknown_minari_id_falls_back_to_local_demos(monkeypatch):
    """A bad dataset id must fall back to the local demos, not raise.

    The fallback is silent by design so a misconfigured id still yields data
    for a run, which is exactly why it needs a test: without one, a run can
    quietly train on a different data source than its config names.
    """
    import minari

    from human2robot.rl import demo as demo_mod

    calls = {}

    def _boom(dataset_id, download=True):
        calls["id"] = dataset_id
        raise RuntimeError("no such dataset")

    monkeypatch.setattr(minari, "load_dataset", _boom)
    monkeypatch.setattr(demo_mod, "_load_local_demos", lambda d: {"from": d, "n": 0})
    result = demo_mod.load_minari_transitions("not-a-dataset")
    assert calls["id"] == "not-a-dataset"
    assert result == {"from": "not-a-dataset", "n": 0}


def test_missing_bindings_report_how_to_build_them(monkeypatch, tmp_path):
    """A missing C++ extension must say how to build it, not fail obscurely.

    This is the first error a new user hits, and a bare ImportError from a
    dynamic loader gives no hint that a cmake step is what is missing.
    """
    from human2robot import cpp_bindings

    monkeypatch.setattr(cpp_bindings, "_BUILD_DIRS", (tmp_path,))
    with pytest.raises(ImportError, match="cmake -S cpp -B cpp/build"):
        cpp_bindings._load_module()


def test_an_episode_that_ends_early_stops_being_recorded(tmp_path):
    """Recording must stop when the episode ends, not run past the horizon.

    The synthetic generator drives random actions for a sampled horizon, and
    the environment's own step limit can fire first. Padding that with
    simulator states that were never reached would fabricate trajectory data.
    """
    from human2robot.data import allegro_demos
    from human2robot.envs import allegro as allegro_env

    class _OneStepEnv(allegro_env.AllegroPickupEnv):
        def __init__(self) -> None:
            super().__init__(max_episode_steps=3)

    # The generator imports the class inside the function, so the patch has to
    # be on the defining module rather than on the one holding the function.
    monkey = allegro_env.AllegroPickupEnv
    allegro_env.AllegroPickupEnv = _OneStepEnv  # type: ignore[attr-defined]
    try:
        paths = allegro_demos.generate_synthetic_demos(
            output_dir=tmp_path, count=1, seed=0
        )
    finally:
        allegro_env.AllegroPickupEnv = monkey  # type: ignore[attr-defined]

    demo = load_demo_npz(paths[0])
    assert demo.q.shape[0] < 40, "the short episode was not truncated"


def test_the_residual_model_accepts_an_unbatched_control():
    """A single control with no batch axis is reshaped, not rejected.

    Uses the project's real model because this path reads force sensors and
    cable sites by name; a stand-in with the right joint count but none of
    that instrumentation would test the stand-in instead of the code.
    """
    from human2robot.dynamics import nominal_physics
    from human2robot.envs.allegro import AllegroPickupEnv

    env = AllegroPickupEnv()
    try:
        obs, _ = env.reset(seed=0)
        flat = np.asarray(obs).reshape(-1)
        ctrl = np.zeros(int(env.model.nu), dtype=np.float64)
        unbatched = nominal_physics.compute_obs_delta(
            env.model, flat, ctrl, ctrl_from_action=False
        )
        batched = nominal_physics.compute_obs_delta(
            env.model, flat, ctrl.reshape(1, -1), ctrl_from_action=False
        )
        assert np.isfinite(unbatched).all()
        assert np.allclose(unbatched.ravel(), batched.ravel()[: unbatched.size])
    finally:
        env.close()


# --- joint layout ----------------------------------------------------------


def test_shape_conversions_reject_the_wrong_width():
    """Every (T, 16) helper refuses anything that is not (T, 16).

    A 15-wide or 17-wide array would be indexed silently and put a finger value
    in the wrong slot, which is the joint-ordering defect this layout exists to
    prevent.
    """
    from human2robot.data import limits

    for bad in (np.zeros((4, 15)), np.zeros((4, 17)), np.zeros(16)):
        with pytest.raises(ValueError):
            limits.fingers_from_dexretarget(bad)
        with pytest.raises(ValueError):
            limits.normalized_targets_to_finger_angles(bad)
        with pytest.raises(ValueError):
            limits.to_q22(bad)


def test_normalized_targets_map_onto_the_actuator_range():
    """-1 and +1 land exactly on the lower and upper actuator bounds.

    The mapping is centre-plus-scaled-span, so the ends have to be exact or a
    target of 1.0 produces a demand outside the range the simulator accepts.
    """
    from human2robot.data import limits

    low, high = limits.finger_bounds()
    lo = limits.normalized_targets_to_finger_angles(-np.ones((1, 16)))
    hi = limits.normalized_targets_to_finger_angles(np.ones((1, 16)))
    assert np.allclose(lo[0], low)
    assert np.allclose(hi[0], high)
    mid = limits.normalized_targets_to_finger_angles(np.zeros((1, 16)))
    assert np.allclose(mid[0], 0.5 * (low + high))

    # Values beyond the normalized range are clipped, not extrapolated.
    assert np.allclose(
        limits.normalized_targets_to_finger_angles(np.full((1, 16), 5.0)), hi
    )


def test_to_q22_embeds_fingers_after_a_zero_base():
    """The 22-DoF configuration is six pinned base coordinates then 16 fingers."""
    from human2robot.data import limits

    fingers = np.arange(16, dtype=float).reshape(1, 16)
    q = limits.to_q22(fingers)
    assert q.shape == (1, 22)
    assert np.all(q[0, :6] == 0.0)
    assert np.array_equal(q[0, 6:], fingers[0])


def test_clamping_lands_inside_the_model_bounds():
    """An out-of-range trajectory is projected onto the actuator bounds.

    Both ends are checked, because the thumb's lower bound is positive and a
    one-sided test would miss a value driven below it.
    """
    from human2robot.data import limits

    low, high = limits.finger_bounds()
    clamped = limits.clamp_finger_angles(np.full((2, 16), -99.0))
    assert np.allclose(clamped, np.broadcast_to(low, (2, 16)))
    assert np.all(clamped >= low - 1e-12)

    clamped = limits.clamp_finger_angles(np.full((2, 16), 99.0))
    assert np.all(clamped <= high + 1e-12)


# --- optimization pipeline -------------------------------------------------


def _write_demo(path, length: int = 12) -> None:
    from human2robot.data.schema import DEMO_SCHEMA_VERSION, DemoTrajectory

    t = np.linspace(0, 1, length)
    q = np.zeros((length, 22), dtype=np.float32)
    q[:, 6:] = (0.3 * t)[:, None] * np.linspace(0.1, 0.5, 16)[None, :]
    traj = DemoTrajectory(
        q=q,
        qdot=np.zeros((length, 22), dtype=np.float32),
        a_demo=np.zeros((length, 22), dtype=np.float32),
        object_pose=np.zeros((length, 7), dtype=np.float32),
        object_vel=np.zeros((length, 6), dtype=np.float32),
        contact=np.zeros((length, 5), dtype=np.float32),
        trajectory_id=path.stem,
        task_id="allegro_pickup",
        source="unit-test",
    )
    traj.validate()
    np.savez_compressed(
        path,
        q=traj.q,
        qdot=traj.qdot,
        a_demo=traj.a_demo,
        object_pose=traj.object_pose,
        object_vel=traj.object_vel,
        contact=traj.contact,
        trajectory_id=traj.trajectory_id,
        task_id=traj.task_id,
        source=traj.source,
        schema_version=np.array(DEMO_SCHEMA_VERSION),
    )


def test_optimizing_a_file_writes_beside_the_input_by_default(tmp_path):
    """No output path means `<stem>_opt.npz` next to the source.

    The default is what makes a one-off call work, and it is the branch a
    caller hits by omitting the argument rather than by passing one.
    """
    from human2robot.optimization.pipeline import (
        OptimizationConfig,
        optimize_demo_file,
    )

    source = tmp_path / "demo.npz"
    _write_demo(source)
    metrics = optimize_demo_file(source, config=OptimizationConfig())
    expected = tmp_path / "demo_opt.npz"
    assert expected.exists()
    assert "final_cost" in metrics
    assert load_demo_npz(expected).trajectory_id == "demo_opt"


def test_optimizing_a_directory_builds_its_own_config(tmp_path):
    """Omitting the config must still produce a valid run, not a None error."""
    from human2robot.optimization.pipeline import optimize_demo_directory

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_demo(raw / "a.npz")
    _write_demo(raw / "b.npz")
    manifest = optimize_demo_directory(raw, tmp_path / "out")
    assert manifest["count"] == 2
    assert set(manifest["per_file"]) == {"a.npz", "b.npz"}
    assert manifest["config"]["dt"] > 0.0
    assert (tmp_path / "out" / "a_opt.npz").exists()
    assert (tmp_path / "out" / "optimization_manifest.json").exists()


def test_optimizing_an_empty_directory_is_reported(tmp_path):
    """An empty input directory must raise rather than write a empty report."""
    from human2robot.optimization.pipeline import optimize_demo_directory

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no NPZ demos"):
        optimize_demo_directory(empty, tmp_path / "out")


# --- feasibility divergence ------------------------------------------------


class _DivergingEnv:
    """An environment whose state goes non-finite partway through a replay.

    Stand-in for a simulator that blows up under a commanded trajectory. It
    exposes only what `score_trajectory` touches, and it is the natural way to
    test the divergence check because provoking a real divergence would need a
    model that is genuinely unstable.
    """

    def __init__(self, fail_after: int) -> None:
        self._fail_after = fail_after
        self._steps = 0
        self._finger_qpos = np.arange(16)
        self.data = type("Data", (), {})()
        self.data.qpos = np.zeros(22)

    def reset(self, seed=None):
        self._steps = 0
        self.data.qpos = np.zeros(22)
        return np.zeros(4), {}

    def step(self, action):
        self._steps += 1
        if self._steps > self._fail_after:
            self.data.qpos = np.full(22, np.nan)
        return np.zeros(4), 0.0, False, False, {}

    def close(self) -> None:
        pass


def test_a_replay_that_diverges_is_reported_not_scored():
    """Non-finite simulator state ends the replay and marks the run diverged.

    Without the check the drift of the frames already recorded would be reported
    as if the trajectory had been performed, which is the one thing this metric
    exists to rule out.
    """
    from human2robot.data.limits import BASE_DOF, DOF
    from human2robot.evaluation import feasibility as F

    q = np.zeros((20, DOF))
    q[:, BASE_DOF:] = 0.2
    report = F.score_trajectory(q, _DivergingEnv(fail_after=5), trajectory_id="div")
    assert report.diverged
    assert not report.is_feasible
    assert report.tracking_drift_max == float("inf")


def test_smoothing_leaves_a_short_sequence_untouched():
    """Sequences shorter than the window are returned as given.

    A Hann window cannot be centred on a short sequence, and inventing padding
    for a two-step trajectory would fabricate a midpoint. Returning the input is
    the honest result, and trajectories this short do occur: `to_transitions`
    keeps any trajectory of two or more configurations.
    """
    short = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(smooth(short), short)
    assert np.array_equal(smooth(short, window=2), short)
    assert np.array_equal(smooth(short, window=99), short)
