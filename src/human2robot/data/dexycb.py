"""DexYCB demonstration loading and retargeting to the Allegro hand.

Reads subject sequences (pose.npz MANO parameters, meta.yml), converts
MANO joint parameters into 16 Allegro finger joint angles via a
weighted-linear-basis approximation, applies the smoothing pipeline, and
stores demos in the standard schema. Frames before MANO tracking starts
(zero parameters) are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from human2robot.data.processing import demo_actions, differentiate, smooth
from human2robot.data.schema import DEMO_SCHEMA_VERSION, DemoTrajectory

_ACTUATOR_LOW = np.array(
    [0.0] * 6 + [-0.47] + [0.196] * 3 + [-0.175] + [0.0] * 8 + [-0.8] * 3
)
_ACTUATOR_HIGH = np.array(
    [0.0] * 6 + [0.47] + [1.61] * 3 + [1.72] + [1.57] * 8 + [0.0] * 3
)

_JOINT_LOW = _ACTUATOR_LOW[6:]
_JOINT_HIGH = _ACTUATOR_HIGH[6:]
_JOINT_CENTER = 0.5 * (_JOINT_LOW + _JOINT_HIGH)
_JOINT_SPAN = 0.5 * (_JOINT_HIGH - _JOINT_LOW)

_MANO_BASIS = np.tile(np.array([0.7, 0.4, 0.6, 0.5, 0.3, 0.6, 0.5, 0.4]), (4, 1))


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
    """Map 45 MANO joint angles to 16 Allegro finger targets in [-1, 1]."""
    joints = pose_m[:, 3:48].reshape(len(pose_m), 15, 3)
    magnitudes = np.linalg.norm(joints, axis=2)
    signs = np.tanh(joints.sum(axis=2))
    curls = np.clip(magnitudes.sum(axis=1) / 6.0, 0.0, 2.0) - 0.5
    curls = np.clip(curls, -1.0, 1.0)
    targets = np.empty((len(pose_m), 16))
    for finger in range(4):
        start = finger * 4
        spread = signs[:, 3 * finger : 3 * finger + 3]
        targets[:, start] = 0.3 * curls + 0.2 * spread[:, 0]
        targets[:, start + 1 : start + 4] = curls[:, None] + 0.15 * spread
    return np.clip(targets, -1.0, 1.0)


def sequence_to_demo(
    path: str | Path,
    sequence_id: str | None = None,
    smooth_window: int = 5,
) -> DemoTrajectory:
    """Convert one DexYCB sequence into a validated demo trajectory."""
    seq = load_sequence(path)
    pose_m = seq["pose_m"]
    tracked = pose_m[np.linalg.norm(pose_m, axis=1) > 1e-9]
    if len(tracked) < 8:
        raise ValueError(f"sequence {seq['sequence_id']} has too few tracked frames")
    targets = _mano_to_joint_targets(tracked)
    q16 = _JOINT_CENTER + targets * _JOINT_SPAN
    q16 = smooth(q16, window=smooth_window)
    q22 = np.zeros((len(q16), 22))
    q22[:, 6:] = q16
    qdot, _ = differentiate(q22)
    a_demo = demo_actions(q22, _ACTUATOR_LOW, _ACTUATOR_HIGH)
    pose_y = seq["pose_y"]
    object_pose = np.zeros((len(q22), 7))
    object_vel = np.zeros((len(q22), 6))
    object_pose[:, 6] = 1.0
    if len(pose_y) >= len(q22):
        for i in range(len(q22)):
            y = pose_y[min(i, len(pose_y) - 1)]
            if np.any(y[0][:3] != 0):
                object_pose[i, :3] = y[0][4:7]
                object_pose[i, 3:7] = [0.0, 0.0, 0.0, 1.0]
                object_vel[i, :] = 0.0
    contact = np.zeros((len(q22), 5))
    sid = sequence_id or f"dexycb_{seq['sequence_id']}"
    traj = DemoTrajectory(
        q=q22.astype(np.float32),
        qdot=qdot.astype(np.float32),
        a_demo=a_demo.astype(np.float32),
        object_pose=object_pose.astype(np.float32),
        object_vel=object_vel.astype(np.float32),
        contact=contact.astype(np.float32),
        trajectory_id=sid,
        task_id="allegro_pickup",
        source="dexycb",
    )
    traj.validate()
    return traj


def build_subject_demos(
    subject_dir: str | Path,
    output_dir: str | Path,
    seed: int = 0,
    max_count: int | None = None,
) -> list[Path]:
    """Convert every usable sequence of a subject into demo NPZs."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rejected = 0
    for pose_path in discover_sequences(subject_dir):
        try:
            traj = sequence_to_demo(pose_path)
        except (ValueError, KeyError):
            rejected += 1
            continue
        idx = len(written)
        out = output_dir / f"dexycb_{idx:04d}.npz"
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
        if max_count is not None and len(written) >= max_count:
            break
    manifest = {
        "subject_dir": str(subject_dir),
        "count": len(written),
        "rejected": rejected,
        "seed_used": bool(max_count is None),
        "rng_seed": int(seed),
        "generator": "dexycb",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / ".rng_check").write_text(str(rng.integers(0, 1000)))
    return written


def main(argv: list[str] | None = None) -> int:
    """Retarget every DexYCB sequence of a subject into demo NPZs.

    Usage:

        uv run python -m human2robot.data.dexycb \\
            --subject-dir data/raw/dexycb/20200709-subject-01 \\
            --output-dir data/demonstrations_dexycb
    """
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Retarget DexYCB sequences into Allegro demo NPZs"
    )
    parser.add_argument(
        "--subject-dir",
        default=None,
        help="Directory holding one subdirectory per sequence with pose.npz",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination for the demo NPZs and manifest",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the manifest")
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Stop after writing this many demos",
    )
    parser.add_argument(
        "--subject",
        default="20200709-subject-01",
        help="Subject name, used to derive paths under the project root",
    )
    args = parser.parse_args(argv)

    subject_dir = (
        Path(args.subject_dir)
        if args.subject_dir is not None
        else project_root / "data" / "raw" / "dexycb" / args.subject
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else project_root / "data" / "demonstrations_dexycb"
    )

    if not subject_dir.is_dir():
        print(
            f"subject dir not found: {subject_dir}\n"
            "run scripts/download_dexycb.sh first, or pass --subject-dir",
            file=sys.stderr,
        )
        return 1
    if not discover_sequences(subject_dir):
        print(
            f"no pose.npz sequences under {subject_dir}\n"
            "pass --subject-dir pointing at an extracted subject",
            file=sys.stderr,
        )
        return 1

    written = build_subject_demos(
        subject_dir,
        output_dir,
        seed=args.seed,
        max_count=args.max_count,
    )
    print(f"wrote {len(written)} demos to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
