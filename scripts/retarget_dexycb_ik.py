"""DexPilot IK retargeting of DexYCB MANO parameters to Allegro joints.

Runs inside WSL where dex-retargeting is installed (~/h2r-wsl, see
scripts/wsl_setup_retargeting.sh). Performs approximate MANO keypoint
forward kinematics from pose.npz parameters, then optimizes Allegro joint
angles with DexPilot (vector retargeting, scaling 1.6, low-pass 0.2).

Output NPZs follow the standard demo schema (22-dim q, base zeros) so the
C++ optimizer and downstream BC consume them unchanged.

MANO parent tree (wrist + 15 joints), fingertip joints 3/6/9/12 per
finger chain, and DexPilot's 20-keypoint MANO layout follow the published
MANO model and dex-retargeting conventions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_MANO_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)

_FINGERTIP_JOINTS = (3, 6, 9, 12)

_FINGER_KEYPOINT_BASES = (1, 5, 9, 13)

_PROJECT_ROOT = Path("/mnt/c/Users/phamt/Projects/human2robot/human2robot")

_ALLEGRO_LOW = np.array(
    [0.0] * 6 + [-0.47] + [0.196] * 3 + [-0.175] + [0.0] * 8 + [-0.8] * 3
)
_ALLEGRO_HIGH = np.array(
    [0.0] * 6 + [0.47] + [1.61] * 3 + [1.72] + [1.57] * 8 + [0.0] * 3
)


def axis_angle_to_matrix(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1e-9:
        return np.eye(3)
    k = v / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def mano_keypoints(pose_m: np.ndarray, link_length: float = 0.032) -> np.ndarray:
    """Compute the 20-keypoint layout DexPilot expects, per frame.

    Only the wrist (0) and the four fingertip keypoints (4, 8, 12, 16)
    influence DexPilot's objective; they come from MANO joints 0 (wrist)
    and 3, 6, 9, 12 (index, middle, ring, pinky tips). MANO joint
    positions are built with the published parent tree and a fixed
    segment length, then scaled by the DexPilot factor (1.6).

    pose_m: (T, 51) MANO parameters (global rot 0:3, hand joints 3:48).
    Returns (T, 20, 3) in meters.
    """
    T = len(pose_m)
    joints = np.zeros((T, 16, 3))
    for t in range(T):
        rotations = [None] * 16
        rotations[0] = axis_angle_to_matrix(pose_m[t, 0:3])
        for i in range(1, 16):
            parent = _MANO_PARENTS[i]
            local = axis_angle_to_matrix(pose_m[t, 3 + 3 * (i - 1) : 6 + 3 * (i - 1)])
            rotations[i] = rotations[parent] @ local
            joints[t, i] = joints[t, parent] + rotations[parent] @ np.array(
                [0.0, -link_length, 0.0]
            )
    keypoints = np.zeros((T, 20, 3))
    keypoints[:, 0] = joints[:, 0]
    keypoints[:, 4] = joints[:, 3]
    keypoints[:, 8] = joints[:, 6]
    keypoints[:, 12] = joints[:, 9]
    keypoints[:, 16] = joints[:, 12]
    return keypoints * 1.6


def build_dexpilot():
    import dex_retargeting.constants as C
    from dex_retargeting.retargeting_config import RetargetingConfig

    config_path = C.get_default_config_path(
        C.RobotName.allegro, C.RetargetingType.dexpilot, C.HandType.right
    )
    RetargetingConfig.set_default_urdf_dir(str(Path.home() / "h2r-assets"))
    config = RetargetingConfig.load_from_file(config_path)
    return config.build()


def to_allegro_q22(q16: np.ndarray) -> np.ndarray:
    q22 = np.zeros((len(q16), 22))
    q22[:, 6:] = q16
    return q22


def retarget_sequence(seq: dict, retargeting, link_length: float) -> np.ndarray:
    pose_m = seq["pose_m"]
    tracked = pose_m[np.linalg.norm(pose_m, axis=1) > 1e-9]
    if len(tracked) < 8:
        raise ValueError(f"too few tracked frames: {len(tracked)}")
    keypoints = mano_keypoints(tracked, link_length=link_length)
    human_indices = retargeting.optimizer.target_link_human_indices
    q16 = np.zeros((len(tracked), 16))
    for t in range(len(tracked)):
        ref_vec = keypoints[t][human_indices[1]] - keypoints[t][human_indices[0]]
        q16[t] = retargeting.retarget(ref_vec)
    return q16


# Robot URDF frame to MANO keypoint frame rotation, verified against
# dex-retargeting's DexPilot anchor geometry (wrist origin, z along fingers,
# y toward the thumb) and MANO's wrist frame (y down the finger chain).
_ROBOT_TO_MANO = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def main(argv: list[str]) -> int:
    subject_dir = _PROJECT_ROOT / "data/raw/dexycb/20200709-subject-01"
    out_dir = _PROJECT_ROOT / "data/demonstrations_dexycb_ik"
    out_dir.mkdir(parents=True, exist_ok=True)
    link_length = float(argv[1]) if len(argv) > 1 else 0.032

    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    import importlib.util

    processing_spec = importlib.util.spec_from_file_location(
        "h2r_processing",
        _PROJECT_ROOT / "src/human2robot/data/processing.py",
    )
    processing = importlib.util.module_from_spec(processing_spec)
    sys.modules["h2r_processing"] = processing
    processing_spec.loader.exec_module(processing)

    schema_spec = importlib.util.spec_from_file_location(
        "h2r_schema", _PROJECT_ROOT / "src/human2robot/data/schema.py"
    )
    schema = importlib.util.module_from_spec(schema_spec)
    sys.modules["h2r_schema"] = schema
    schema_spec.loader.exec_module(schema)

    demo_actions = processing.demo_actions
    smooth = processing.smooth
    differentiate = processing.differentiate
    DemoTrajectory = schema.DemoTrajectory
    DEMO_SCHEMA_VERSION = schema.DEMO_SCHEMA_VERSION

    retargeting = build_dexpilot()

    pose_paths = sorted(subject_dir.glob("*/pose.npz"))
    print(f"retargeting {len(pose_paths)} sequences, link_length={link_length}")
    started = time.perf_counter()
    written, rejected = [], 0
    for pose_path in pose_paths:
        data = np.load(pose_path)
        pose_m = np.asarray(data["pose_m"])[:, 0, :]

        try:
            q16 = retarget_sequence({"pose_m": pose_m}, retargeting, link_length)
        except ValueError:
            rejected += 1
            continue
        q16 = smooth(q16, window=5)
        q22 = to_allegro_q22(q16)
        qdot, _ = differentiate(q22)
        a_demo = demo_actions(q22, _ALLEGRO_LOW, _ALLEGRO_HIGH)
        n = len(q22)
        object_pose = np.zeros((n, 7))
        object_pose[:, 6] = 1.0
        object_vel = np.zeros((n, 6))
        contact = np.zeros((n, 5))
        traj = DemoTrajectory(
            q=q22.astype(np.float32),
            qdot=qdot.astype(np.float32),
            a_demo=a_demo.astype(np.float32),
            object_pose=object_pose.astype(np.float32),
            object_vel=object_vel.astype(np.float32),
            contact=contact.astype(np.float32),
            trajectory_id=f"dexycb_ik_{pose_path.parent.name}",
            task_id="allegro_pickup",
            source="dexycb_ik",
        )
        try:
            traj.validate()
        except Exception:
            rejected += 1
            continue
        out = out_dir / f"ik_{pose_path.parent.name}.npz"
        np.savez_compressed(
            out,
            q=traj.q,
            qdot=traj.qdot,
            a_demo=traj.a_demo,
            object_pose=traj.object_pose,
            object_vel=traj.object_vel,
            contact=traj.contact,
            trajectory_id=np.array(traj.trajectory_id),
            task_id=np.array(traj.task_id),
            source=np.array(traj.source),
            schema_version=np.array(DEMO_SCHEMA_VERSION),
        )
        written.append(out)
    wall = time.perf_counter() - started
    manifest = {
        "count": len(written),
        "rejected": rejected,
        "retargeter": "dexpilot-ik",
        "link_length": link_length,
        "wall_seconds": wall,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"written {len(written)}, rejected {rejected}, wall {wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
