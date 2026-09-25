"""Tests for the baseline versus optimized comparison report."""

import pytest

from human2robot.data.allegro_demos import generate_synthetic_demos
from human2robot.evaluation.traj_compare import compare_directory, write_report
from human2robot.optimization import OptimizationConfig, optimize_demo_directory


@pytest.fixture(scope="module", name="compare_dirs")
def compare_dirs_fixture(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    opt = tmp_path_factory.mktemp("opt")
    generate_synthetic_demos(output_dir=raw, count=3, seed=0)
    config = OptimizationConfig(seed=0, max_iterations=50)
    optimize_demo_directory(raw, opt, config)
    return raw, opt


def test_compare_directory_produces_reduction(compare_dirs):
    raw, opt = compare_dirs
    report = compare_directory(raw, opt, dt=0.01)
    assert report["count"] == 3
    for key in ("raw", "optimized"):
        assert key in report
        assert "max_jerk" in report[key]
        assert "smoothness" in report[key]
    assert report["max_jerk_reduction_pct"] > -100.0


def test_write_report_creates_json(compare_dirs, tmp_path):
    raw, opt = compare_dirs
    report = compare_directory(raw, opt, dt=0.01)
    out = tmp_path / "report.json"
    write_report(report, out)
    text = out.read_text()
    assert "max_velocity" in text


def test_compare_missing_inputs_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        compare_directory(tmp_path, tmp_path, dt=0.01)
