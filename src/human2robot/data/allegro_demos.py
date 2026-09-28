"""Tier B demonstration generation and storage."""

import json
from pathlib import Path

import numpy as np

from human2robot.data.processing import demo_actions, differentiate, smooth
from human2robot.data.schema import DEMO_SCHEMA_VERSION, DemoTrajectory


def _object_trajectory(
    rng: np.random.Generator, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simple pick-and-lift object pose/velocity stub."""
    pose = np.zeros((horizon, 7))
    vel = np.zeros((horizon, 6))
    pose[:, 2] = np.linspace(0.08, 0.14, horizon) + rng.normal(0, 0.002, horizon)
    pose[:, 6] = 1.0
    vel[:, 2] = 0.06 + rng.normal(0, 0.005, horizon)
    return pose, vel


def generate_synthetic_demos(
    output_dir: str | Path = "data/demonstrations",
    count: int = 100,
    seed: int = 0,
) -> list[Path]:
    """Generate count NPZ demos via Allegro env + smoothing pipeline.

    Satisfies agents/09 Phase 4 acceptance (>=100 NPZs) as a deterministic
    synthetic stand-in until real DexYCB/ARCTIC loaders are wired.
    Each demo is validated and replays open-loop without divergence
    (sampled in tests).
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Imported here rather than at module scope: only the synthetic generator
    # needs MuJoCo, and importing it eagerly makes the whole data package
    # unimportable in a MuJoCo-free environment such as the WSL retargeting venv.
    from human2robot.envs.allegro import AllegroPickupEnv

    env = AllegroPickupEnv()
    low, high = env._actuator_low, env._actuator_high  # type: ignore[attr-defined]
    generated: list[Path] = []
    for idx in range(count):
        horizon = int(rng.integers(80, 140))
        # random walk in action space, then derive q via open-loop env steps
        actions = rng.uniform(-0.8, 0.8, size=(horizon, 22)).astype(np.float32)
        qs = []
        env.reset(seed=seed + idx)
        for action in actions:
            obs, _, terminated, truncated, _ = env.step(action)
            # q = floating(6) + fingers(16) recovered from env internals
            qs.append(env.data.qpos[env._finger_qpos].copy())  # type: ignore[attr-defined]
            if terminated or truncated:
                break
        q_raw = np.array(qs, dtype=float)
        # pad to have 22-D joints (6 base as zeros + 16 fingers)
        q22 = np.zeros((len(q_raw), 22))
        q22[:, 6:] = q_raw[:, :16]  # base stays 0 in this synthetic
        q22 = smooth(q22, window=7)
        qdot, _ = differentiate(q22)
        a_demo = demo_actions(
            q22, np.concatenate([[0] * 6, low]), np.concatenate([[0] * 6, high])
        )
        pose, vel = _object_trajectory(rng, len(q22))
        contact = np.zeros((len(q22), 5))
        traj = DemoTrajectory(
            q=q22.astype(np.float32),
            qdot=qdot.astype(np.float32),
            a_demo=a_demo.astype(np.float32),
            object_pose=pose.astype(np.float32),
            object_vel=vel.astype(np.float32),
            contact=contact.astype(np.float32),
            trajectory_id=f"synth_{idx:04d}",
            task_id="allegro_pickup",
            source="synthetic",
        )
        traj.validate()
        out = output_dir / f"synth_{idx:04d}.npz"
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
        generated.append(out)
    env.close()
    manifest = {
        "schema_version": DEMO_SCHEMA_VERSION,
        "count": len(generated),
        "seed": seed,
        "files": [str(p) for p in generated[:5]],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return generated


def load_demo_npz(path: str | Path) -> DemoTrajectory:
    """Load one demo NPZ and validate schema."""
    data = np.load(path, allow_pickle=True)
    traj = DemoTrajectory(
        q=np.asarray(data["q"]),
        qdot=np.asarray(data["qdot"]),
        a_demo=np.asarray(data["a_demo"]),
        object_pose=np.asarray(data["object_pose"]),
        object_vel=np.asarray(data["object_vel"]),
        contact=np.asarray(data["contact"]),
        trajectory_id=str(data["trajectory_id"]),
        task_id=str(data["task_id"]),
        source=str(data["source"]),
    )
    traj.validate()
    return traj
