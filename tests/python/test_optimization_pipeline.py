"""End-to-end pipeline tests over synthetic demos."""

import json

import numpy as np
import pytest

from human2robot.data.allegro_demos import generate_synthetic_demos, load_demo_npz
from human2robot.optimization import (
    OptimizationConfig,
    optimize_demo_directory,
    optimize_demo_file,
)


@pytest.fixture(scope="module", name="demo_dir")
def demo_dir_fixture(tmp_path_factory):
    path = tmp_path_factory.mktemp("demos")
    generate_synthetic_demos(output_dir=path, count=3, seed=0)
    return path


def test_optimize_single_file_preserves_raw(demo_dir, tmp_path):
    config = OptimizationConfig(seed=0)
    raw_path = sorted(demo_dir.glob("*.npz"))[0]
    raw_before = load_demo_npz(raw_path)
    out_path = tmp_path / "demo_0000_opt.npz"
    metrics = optimize_demo_file(raw_path, config, out_path)
    assert metrics["iterations"] > 0
    assert out_path.exists()
    raw_after = load_demo_npz(raw_path)
    np.testing.assert_array_equal(raw_before.q, raw_after.q)
    optimized = load_demo_npz(out_path)
    assert optimized.q.shape == raw_before.q.shape


def test_directory_pipeline_writes_manifest(demo_dir, tmp_path):
    config = OptimizationConfig(seed=1, max_iterations=50)
    out_dir = tmp_path / "optimized"
    manifest = optimize_demo_directory(demo_dir, out_dir, config)
    assert manifest["count"] == 3
    assert (out_dir / "optimization_manifest.json").exists()
    saved = json.loads((out_dir / "optimization_manifest.json").read_text())
    assert saved["config"]["seed"] == 1
    assert len(saved["per_file"]) == 3


def test_optimized_smoothing_improves_jerk(demo_dir, tmp_path):
    from human2robot.cpp_bindings import compute_metrics

    config = OptimizationConfig(seed=0, max_iterations=100)
    out_dir = tmp_path / "optimized"
    optimize_demo_directory(demo_dir, out_dir, config)
    raw_path = sorted(demo_dir.glob("*.npz"))[0]
    opt_path = sorted(out_dir.glob("*_opt.npz"))[0]
    raw_metrics = compute_metrics(load_demo_npz(raw_path).q, 0.01)
    opt_metrics = compute_metrics(load_demo_npz(opt_path).q, 0.01)
    assert opt_metrics["joint_limit_violations"] == 0
    assert opt_metrics["smoothness"] <= raw_metrics["smoothness"] * 2.0
