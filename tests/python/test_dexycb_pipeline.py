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
from human2robot.data.dexycb import discover_sequences

IK_DEMO_DIR = Path("data/demonstrations_dexycb_ik")

pytestmark = pytest.mark.skipif(
    not Path("data/raw/dexycb/20200709-subject-01").exists(),
    reason="DexYCB subject-01 data not downloaded",
)


@pytest.fixture(scope="module", name="demo_paths")
def demo_paths_fixture():
    """The DexPilot IK retargets, which are what every result is measured on.

    The basis retargeter in ``data/dexycb.py`` is disabled because it collapses
    16 joint dimensions into 1; see ``docs/FINDINGS_retargeting.md``.
    """
    paths = sorted(IK_DEMO_DIR.glob("*.npz"))
    if not paths:
        pytest.skip("run scripts/retarget_dexycb_ik.py first")
    return paths


def test_discovery_finds_all_sequences():
    seqs = discover_sequences("data/raw/dexycb/20200709-subject-01")
    assert len(seqs) == 100


def test_meets_phase4_acceptance_count(demo_paths):
    assert len(demo_paths) >= 100


def test_manifest_records_rejections(demo_paths):
    manifest = json.loads((demo_paths[0].parent / "manifest.json").read_text())
    assert manifest["count"] == len(demo_paths)


def test_all_demos_valid_and_bounded(demo_paths):
    for path in demo_paths[:20]:
        demo = load_demo_npz(path)
        assert demo.q.shape[1] == 22
        assert np.isfinite(demo.q).all()
        assert demo.q.min() >= -0.5
        assert demo.q.max() <= 1.75


def test_the_basis_retargeter_is_disabled():
    """It is not a retargeting, so it must not silently produce demos again.

    This function drove all four joints of a finger from one curl scalar, which
    put 90 percent of the demo variance in a single dimension. Every published
    trajectory number came from it before that was found.
    """
    from human2robot.data.dexycb import _mano_to_joint_targets

    with pytest.raises(NotImplementedError, match="not a retargeting"):
        _mano_to_joint_targets(np.zeros((4, 51)))


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
