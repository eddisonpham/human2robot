"""DexYCB sequence discovery and MANO loading.

The retargeting that used to live here has been removed. It drove all four
joints of a finger from a single curl scalar, so the demo data carried 90
percent of its variance in one dimension of 16 and every published trajectory
result described that scalar rather than a hand. See
``docs/FINDINGS_retargeting.md``.

Retargeting now lives in ``scripts/retarget_dexycb_ik.py``, which runs DexPilot
vector retargeting against MANO keypoints. That path needs a MuJoCo-free
environment, which is why it is a script rather than a module here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def discover_sequences(subject_dir: str | Path) -> list[Path]:
    """Return sorted pose.npz paths for every sequence under a subject."""
    root = Path(subject_dir)
    return sorted(root.glob("*/pose.npz"))


def load_sequence(path: str | Path) -> dict:
    """Load one DexYCB sequence's MANO parameters and metadata."""
    path = Path(path)
    meta_path = path.parent / "meta.yml"
    num_frames = None
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip().startswith("num_frames:"):
                num_frames = int(line.split(":")[1].strip())
                break
    data = np.load(path)
    pose_m = np.asarray(data["pose_m"])[:, 0, :]
    if num_frames is not None:
        pose_m = pose_m[:num_frames]
    return {
        "pose_m": pose_m,
        "pose_y": np.asarray(data["pose_y"]),
        "sequence_id": path.parent.name,
        "path": path,
    }


def _mano_to_joint_targets(pose_m: np.ndarray) -> np.ndarray:
    """Removed: the basis retargeter was not a retargeting.

    It summed MANO joint magnitudes into one curl scalar per finger and
    broadcast that to all four joints with a 0.15 spread, which put 90 percent
    of the demo variance in a single dimension of 16 and made every published
    trajectory result describe that scalar. Use
    ``scripts/retarget_dexycb_ik.py`` (DexPilot IK against MANO keypoints).

    The stub is retained so that any surviving caller fails loudly rather than
    silently regenerating degenerate demos.
    """
    raise NotImplementedError(
        "the basis retargeter was removed: it is not a retargeting, see "
        "docs/FINDINGS_retargeting.md. Use scripts/retarget_dexycb_ik.py"
    )
