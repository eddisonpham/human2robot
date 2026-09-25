"""IK-retargeted DexYCB demo validation and basis-mapping comparison."""

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

pytestmark = pytest.mark.skipif(
    not Path("data/demonstrations_dexycb_ik/manifest.json").exists(),
    reason="IK retargeting has not been run",
)

BASIS_DIR = Path("data/demonstrations_dexycb")
IK_DIR = Path("data/demonstrations_dexycb_ik")


@pytest.fixture(scope="module", name="demo_pairs")
def demo_pairs_fixture():
    basis_paths = sorted(BASIS_DIR.glob("dexycb_*.npz"))
    ik_paths = sorted(IK_DIR.glob("ik_*.npz"))
    pairs = []
    for basis_path, ik_path in zip(basis_paths, ik_paths, strict=False):
        ik_seq = ik_path.stem.removeprefix("ik_")
        traj_id = str(np.load(basis_path)["trajectory_id"])
        if traj_id == f"dexycb_{ik_seq}":
            pairs.append((basis_path, ik_path))
    return pairs


def test_manifest_counts():
    manifest = json.loads((IK_DIR / "manifest.json").read_text())
    assert manifest["count"] >= 100
    assert manifest["retargeter"] == "dexpilot-ik"


def test_ik_demos_valid_and_motion_rich(demo_pairs):
    assert len(demo_pairs) >= 10
    for _, ik_path in demo_pairs:
        demo = load_demo_npz(ik_path)
        q16 = demo.q[:, 6:]
        assert np.isfinite(q16).all()
        assert q16.min() >= -0.5
        assert q16.max() <= 1.72
        assert float(q16.std()) > 0.02


def test_ik_demos_have_active_joints_comparable_to_basis():
    basis_stds, ik_stds = [], []
    for basis_path, ik_path in sorted(
        zip(
            sorted(BASIS_DIR.glob("dexycb_*.npz")),
            sorted(IK_DIR.glob("ik_*.npz")),
            strict=False,
        )
    )[:20]:
        if not basis_path.exists():
            continue
        basis_q = load_demo_npz(basis_path).q[:, 6:]
        ik_q = load_demo_npz(ik_path).q[:, 6:]
        basis_stds.append(float(basis_q.std()))
        ik_stds.append(float(ik_q.std()))
    assert np.mean(ik_stds) > 0.1
    assert np.mean(ik_stds) > 0.5 * np.mean(basis_stds)


def test_cpp_optimizer_improves_ik_demo(demo_pairs):
    _, ik_path = demo_pairs[0]
    demo = load_demo_npz(ik_path)
    config = OptimizerConfig(
        dof=22,
        lower=np.full(22, -0.5),
        upper=np.full(22, 1.72),
        max_velocity=np.full(22, 2.0),
        max_acceleration=np.full(22, 20.0),
        max_iterations=100,
        seed=0,
    )
    result = optimize_trajectory(demo.q, demo.q, 0.02, config)
    before = compute_metrics(demo.q, 0.02)
    after = compute_metrics(np.asarray(result.trajectory), 0.02)
    assert after["max_jerk"] < before["max_jerk"]
