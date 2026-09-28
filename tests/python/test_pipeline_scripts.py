"""The trajectory-pipeline experiment scripts.

These produce the project's primary measured result and had no tests, so a
change to either could silently alter a published number.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from human2robot.data.allegro_demos import load_demo_npz
from human2robot.data.schema import DEMO_SCHEMA_VERSION, DemoTrajectory

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    """Import a script by path, since scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", name="bc_script")
def bc_script_fixture():
    return _load("run_downstream_bc")


@pytest.fixture(scope="module", name="compare_script")
def compare_script_fixture():
    return _load("compare_dexycb_synthetic")


def write_demo(path: Path, horizon: int = 30, dof: int = 22) -> None:
    """Write a schema-valid demo NPZ with a smooth ramp.

    The schema pins every joint array to 22 columns, so *dof* exists only to
    make a wrong value an obvious test failure.
    """
    q = np.linspace(0.0, 0.4, horizon, dtype=np.float32)[:, None] * np.ones(
        (1, dof), dtype=np.float32
    )
    np.savez_compressed(
        path,
        q=q.astype(np.float32),
        qdot=np.zeros((horizon, dof), dtype=np.float32),
        a_demo=np.zeros((horizon, dof), dtype=np.float32),
        object_pose=np.zeros((horizon, 7), dtype=np.float32),
        object_vel=np.zeros((horizon, 6), dtype=np.float32),
        contact=np.zeros((horizon, 5), dtype=np.float32),
        trajectory_id=np.array(path.stem),
        task_id=np.array("allegro_pickup"),
        source=np.array("unit-test"),
        schema_version=np.array(DEMO_SCHEMA_VERSION),
    )


# --- run_downstream_bc --------------------------------------------------------


def test_bc_demo_sets_are_disjoint_and_named(bc_script):
    assert set(bc_script.DEMO_SETS) == {"synthetic", "dexycb", "dexycb-s2"}
    raw_dirs = [v[0] for v in bc_script.DEMO_SETS.values()]
    opt_dirs = [v[1] for v in bc_script.DEMO_SETS.values()]
    out_paths = [v[2] for v in bc_script.DEMO_SETS.values()]
    assert len(set(raw_dirs)) == len(raw_dirs)
    assert len(set(opt_dirs)) == len(opt_dirs)
    assert len(set(out_paths)) == len(out_paths)
    for raw_dir, opt_dir, out_path in bc_script.DEMO_SETS.values():
        assert raw_dir != opt_dir
        assert out_path.suffix == ".json"


def test_bc_rejects_unknown_demo_set(bc_script, capsys):
    assert bc_script.main(["--set", "nonexistent"]) == 1
    assert "unknown demo set" in capsys.readouterr().err


def test_bc_load_positions_requires_demos(bc_script, tmp_path):
    with pytest.raises(FileNotFoundError, match="no demos"):
        bc_script.load_positions(tmp_path)


def test_bc_load_positions_filters_by_suffix(bc_script, tmp_path):
    write_demo(tmp_path / "a.npz")
    write_demo(tmp_path / "b_opt.npz")
    assert len(bc_script.load_positions(tmp_path)) == 2
    assert len(bc_script.load_positions(tmp_path, suffix="_opt")) == 1


def test_bc_transition_dataset_splits_and_aligns(bc_script):
    trajs = [
        np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32)
    ]
    rng = np.random.default_rng(0)
    (train, holdout) = bc_script.make_transition_dataset(trajs, rng)
    assert train[0].shape[0] == train[1].shape[0]
    assert holdout[0].shape[0] == holdout[1].shape[0]
    assert train[0].shape[0] + holdout[0].shape[0] == 9
    np.testing.assert_allclose(train[1], 1.0, atol=1e-6)


def test_bc_transition_dataset_is_deterministic(bc_script):
    trajs = [
        np.linspace(0, 1, 20, dtype=np.float32)[:, None]
        * np.ones((1, 2), dtype=np.float32)
    ]
    first = bc_script.make_transition_dataset(trajs, np.random.default_rng(7))
    second = bc_script.make_transition_dataset(trajs, np.random.default_rng(7))
    np.testing.assert_allclose(first[0][0], second[0][0])
    np.testing.assert_allclose(first[1][0], second[1][0])


def test_bc_train_and_evaluate_shapes(bc_script):
    rng = np.random.default_rng(0)
    ramp = np.linspace(0, 1, 40, dtype=np.float32)[:, None] * np.ones(
        (1, 4), dtype=np.float32
    )
    train, holdout = bc_script.make_transition_dataset([ramp], rng)
    model, best = bc_script.train_bc(train, holdout, seed=0, epochs=3, batch=8)
    metrics = bc_script.evaluate(model, holdout)
    assert best >= 0.0
    assert set(metrics) == {"mse", "mae", "max_err"}
    assert metrics["mse"] >= 0.0
    assert metrics["max_err"] >= metrics["mae"] - 1e-9


def test_bc_optimized_beats_raw_end_to_end(bc_script, tmp_path, monkeypatch):
    """A jittered raw set must imitate worse than its smooth counterpart.

    This is the property the whole experiment rests on, so it is asserted end
    to end rather than trusting the recorded artifact.
    """
    rng = np.random.default_rng(1)
    raw_dir = tmp_path / "raw"
    opt_dir = tmp_path / "opt"
    raw_dir.mkdir()
    opt_dir.mkdir()
    horizon = 40
    ramp = np.linspace(0, 0.5, horizon, dtype=np.float32)[:, None] * np.ones(
        (1, 22), dtype=np.float32
    )
    for i in range(6):
        write_demo(raw_dir / f"d{i}.npz", horizon=horizon)
        np.savez_compressed(
            raw_dir / f"d{i}.npz",
            q=(ramp + rng.normal(0, 0.05, ramp.shape)).astype(np.float32),
            qdot=np.zeros((horizon, 22), dtype=np.float32),
            a_demo=np.zeros((horizon, 22), dtype=np.float32),
            object_pose=np.zeros((horizon, 7), dtype=np.float32),
            object_vel=np.zeros((horizon, 6), dtype=np.float32),
            contact=np.zeros((horizon, 5), dtype=np.float32),
            trajectory_id=np.array(f"d{i}"),
            task_id=np.array("allegro_pickup"),
            source=np.array("unit-test"),
            schema_version=np.array(DEMO_SCHEMA_VERSION),
        )
        write_demo(opt_dir / f"d{i}_opt.npz", horizon=horizon)

    out = tmp_path / "out.json"
    monkeypatch.setitem(bc_script.DEMO_SETS, "unit", (raw_dir, opt_dir, out))
    assert bc_script.main(["--set", "unit"]) == 0
    results = json.loads(out.read_text())
    assert set(results) == {"raw", "optimized", "mixed"}
    assert results["optimized"]["mse"] < results["raw"]["mse"]


# --- compare_dexycb_synthetic -------------------------------------------------


def test_compare_aggregate_reports_mean_std_max(compare_script):
    keys = ("max_velocity", "max_acceleration", "max_jerk", "smoothness")
    rows = [{k: float(i + 1) for k in keys} for i in range(3)]
    agg = compare_script.aggregate(rows)
    assert set(agg) == set(keys)
    for k in keys:
        assert agg[k]["mean"] == pytest.approx(2.0)
        assert agg[k]["max"] == pytest.approx(3.0)
        assert agg[k]["std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))


def test_compare_reductions_are_positive_when_optimizer_helps(compare_script):
    report = {
        "raw": {
            k: {"mean": 10.0, "std": 0.0, "max": 10.0}
            for k in ("max_velocity", "max_acceleration", "max_jerk", "smoothness")
        },
        "optimized": {
            k: {"mean": 5.0, "std": 0.0, "max": 5.0}
            for k in ("max_velocity", "max_acceleration", "max_jerk", "smoothness")
        },
    }
    out = compare_script.reductions(report)
    assert out["max_jerk_reduction_pct"] == pytest.approx(50.0)
    assert out["smoothness_reduction_pct"] == pytest.approx(50.0)


def test_compare_reductions_survive_zero_raw(compare_script):
    """The max(raw, 1e-12) guard avoids a division by zero.

    It reports a full 100 percent reduction when the raw mean is exactly zero,
    which is nonsense as a measurement but is the documented guard behaviour.
    """
    keys = ("max_velocity", "max_acceleration", "max_jerk", "smoothness")
    report = {
        "raw": {k: {"mean": 0.0, "std": 0.0, "max": 0.0} for k in keys},
        "optimized": {k: {"mean": 0.0, "std": 0.0, "max": 0.0} for k in keys},
    }
    out = compare_script.reductions(report)
    assert out["max_jerk_reduction_pct"] == pytest.approx(100.0)


def test_compare_written_demo_is_schema_valid(compare_script, tmp_path):
    traj = DemoTrajectory(
        q=np.zeros((12, 22), dtype=np.float32),
        qdot=np.zeros((12, 22), dtype=np.float32),
        a_demo=np.zeros((12, 22), dtype=np.float32),
        object_pose=np.zeros((12, 7), dtype=np.float32),
        object_vel=np.zeros((12, 6), dtype=np.float32),
        contact=np.zeros((12, 5), dtype=np.float32),
        trajectory_id="seq0",
        task_id="allegro_pickup",
        source="dexycb",
    )
    q_opt = np.linspace(0.0, 0.3, 12)[:, None] * np.ones((1, 22))
    out = tmp_path / "seq0_opt.npz"
    compare_script.write_optimized_demo(out, q_opt, traj)
    loaded = load_demo_npz(out)
    loaded.validate()
    assert loaded.trajectory_id == "seq0_opt"
    assert loaded.source == "dexycb_opt"
    # Derived fields must match the written trajectory length, not the raw demo.
    assert loaded.qdot.shape == (12, 22)
    assert loaded.a_demo.shape == (12, 22)
    assert loaded.object_pose.shape == (12, 7)
    assert loaded.q.shape == (12, 22)


def test_compare_control_tag_is_recorded(compare_script, tmp_path):
    traj = DemoTrajectory(
        q=np.zeros((6, 22), dtype=np.float32),
        qdot=np.zeros((6, 22), dtype=np.float32),
        a_demo=np.zeros((6, 22), dtype=np.float32),
        object_pose=np.zeros((6, 7), dtype=np.float32),
        object_vel=np.zeros((6, 6), dtype=np.float32),
        contact=np.zeros((6, 5), dtype=np.float32),
        trajectory_id="seq1",
        task_id="allegro_pickup",
        source="dexycb",
    )
    out = tmp_path / "seq1_resampled.npz"
    compare_script.write_optimized_demo(out, np.zeros((6, 22)), traj, tag="resampled")
    loaded = load_demo_npz(out)
    assert loaded.trajectory_id == "seq1_resampled"
    assert loaded.source == "dexycb_resampled"


def test_compare_missing_dexycb_dir_raises_with_a_command(
    compare_script, tmp_path, monkeypatch
):
    absent = (tmp_path / "absent", tmp_path / "absent_opt", tmp_path / "absent.json")
    monkeypatch.setattr(
        compare_script,
        "SUBJECT_SETS",
        {**compare_script.SUBJECT_SETS, "subject-01": absent},
    )
    monkeypatch.setattr(compare_script, "SYNTH_DIR", tmp_path / "absent2")
    with pytest.raises(FileNotFoundError, match="human2robot.data.dexycb"):
        compare_script.main([])


def test_compare_rejects_an_unknown_subject(compare_script, capsys):
    assert compare_script.main(["--subject", "subject-99"]) == 1
    assert "unknown subject" in capsys.readouterr().err


def test_compare_subjects_write_to_separate_outputs(compare_script, tmp_path):
    """Two subjects must not overwrite each other's report."""
    outs = {v[2] for v in compare_script.SUBJECT_SETS.values()}
    assert len(outs) == len(compare_script.SUBJECT_SETS)
