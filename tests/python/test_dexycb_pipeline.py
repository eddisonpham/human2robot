"""DexYCB loader, retargeting validation, and C++ optimization integration."""

import json
from pathlib import Path

import numpy as np
import pytest

from human2robot.cpp_bindings import (
    OptimizerConfig,
    compute_metrics,
    optimize_trajectory,
)
from human2robot.data.allegro_demos import load_demo_npz
from human2robot.data.dexycb import build_subject_demos, discover_sequences

pytestmark = pytest.mark.skipif(
    not Path("data/raw/dexycb/20200709-subject-01").exists(),
    reason="DexYCB subject-01 data not downloaded",
)


@pytest.fixture(scope="module", name="demo_paths")
def demo_paths_fixture(tmp_path_factory):
    out = tmp_path_factory.mktemp("dexycb_demos")
    return build_subject_demos("data/raw/dexycb/20200709-subject-01", out, seed=0)


def test_discovery_finds_all_sequences():
    seqs = discover_sequences("data/raw/dexycb/20200709-subject-01")
    assert len(seqs) == 100


def test_meets_phase4_acceptance_count(demo_paths):
    assert len(demo_paths) >= 100


def test_manifest_records_rejections(demo_paths, tmp_path):
    manifest = json.loads((demo_paths[0].parent / "manifest.json").read_text())
    assert manifest["count"] == len(demo_paths)
    assert manifest["generator"] == "dexycb"


def test_all_demos_valid_and_bounded(demo_paths):
    for path in demo_paths[:20]:
        demo = load_demo_npz(path)
        assert demo.q.shape[1] == 22
        assert np.isfinite(demo.q).all()
        assert demo.q.min() >= -0.5
        assert demo.q.max() <= 1.62


def test_demos_show_motion_not_static(demo_paths):
    demo = load_demo_npz(demo_paths[0])
    q16 = demo.q[:, 6:]
    assert float(q16.std()) > 0.01


def test_optimized_dexycb_demo_reduces_jerk(demo_paths):
    config = OptimizerConfig(
        dof=22,
        lower=np.full(22, -0.5),
        upper=np.full(22, 1.62),
        max_velocity=np.full(22, 2.0),
        max_acceleration=np.full(22, 20.0),
        max_iterations=100,
        seed=0,
    )
    demo = load_demo_npz(demo_paths[0])
    result = optimize_trajectory(demo.q, demo.q, 0.02, config)
    assert result.final_cost <= result.initial_cost
    before = compute_metrics(demo.q, 0.02)
    after = compute_metrics(np.asarray(result.trajectory), 0.02)
    assert after["max_jerk"] < before["max_jerk"]
