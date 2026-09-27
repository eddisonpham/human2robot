"""Demonstration loading for local .npz demo files and Minari fallback.

Used by Condition B/C for demo-seeded replay buffers.  The local loader
replays stored q/qdot trajectories through the AllegroPickupEnv internals
so the resulting transitions use the same 64-d observation the environment
produces at runtime.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from human2robot.data.allegro_demos import load_demo_npz
from human2robot.envs.allegro import AllegroPickupEnv
from human2robot.utils.rotation import rotation_vector as _rotation_vector


def _build_obs_from_state(
    env: AllegroPickupEnv,
    q: np.ndarray,
    qvel: np.ndarray,
    obj_pose: np.ndarray,
    obj_vel: np.ndarray,
    contact: np.ndarray,
) -> np.ndarray:
    """Write q/qvel/object/contact into MuJoCo and rebuild the 64-d obs."""
    env.data.qpos[env._base_qpos : env._base_qpos + 7] = q[:7]
    env.data.qpos[env._finger_qpos] = q[6:22]
    env.data.qpos[env._object_qpos : env._object_qpos + 7] = obj_pose[:7]
    env.data.qvel[env._base_qvel : env._base_qvel + 6] = qvel[:6]
    env.data.qvel[env._finger_qvel] = qvel[6:22]
    env.data.qvel[env._object_qvel : env._object_qvel + 6] = obj_vel[:6]
    env.data.sensordata[:4] = contact[:4]
    mujoco.mj_forward(env.model, env.data)

    base_qpos_arr = env.data.qpos[env._base_qpos : env._base_qpos + 7]
    base_rot = np.zeros((3, 3), dtype=np.float64)
    mujoco.mju_quat2Mat(base_rot.ravel(), base_qpos_arr[3:])
    base_q = np.concatenate(
        (
            base_qpos_arr[:3],
            _rotation_vector(base_rot),
            env.data.qpos[env._finger_qpos].astype(np.float32),
        )
    )
    base_qvel = np.concatenate(
        (
            env.data.qvel[env._base_qvel : env._base_qvel + 6],
            env.data.qvel[env._finger_qvel],
        )
    )
    object_qpos_arr = env.data.qpos[env._object_qpos : env._object_qpos + 7]
    object_rot = np.zeros((3, 3), dtype=np.float64)
    mujoco.mju_quat2Mat(object_rot.ravel(), object_qpos_arr[3:])
    object_qvel = env.data.qvel[env._object_qvel : env._object_qvel + 6]
    contact_arr = np.array(
        [env.data.sensordata[i] > 1e-6 for i in range(4)] + [False],
        dtype=np.float32,
    )
    target = np.array([0.0, 0.0, 0.18], dtype=np.float32)
    obs = np.concatenate(
        (
            base_q,
            base_qvel,
            object_qpos_arr[:3],
            _rotation_vector(object_rot),
            object_qvel,
            contact_arr,
            target,
        )
    )
    return obs.astype(np.float32)


def _load_local_demos(demo_dir: str) -> dict[str, np.ndarray]:
    """Load every .npz demo in *demo_dir* and flatten to transitions.

    Each stored trajectory is replayed through the environment's MuJoCo data
    so the returned observations match what the live env would produce.
    """
    demo_path = Path(demo_dir)
    if not demo_path.is_dir():
        raise FileNotFoundError(f"demo_dir not found: {demo_dir}")

    npz_files = sorted(demo_path.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"no .npz files in {demo_dir}")

    env = AllegroPickupEnv(max_episode_steps=500)
    try:
        obs_list, act_list, next_obs_list, rew_list, done_list = (
            [],
            [],
            [],
            [],
            [],
        )
        for path in npz_files:
            demo = load_demo_npz(path)
            q = demo.q.astype(np.float64)  # (T, 22)
            qdot = demo.qdot.astype(np.float64)  # (T, 22)
            a = demo.a_demo  # (T, 22)
            T = len(q)
            if T < 2:
                continue
            rows = []
            for t in range(T):
                rows.append(
                    _build_obs_from_state(
                        env,
                        q[t],
                        qdot[t],
                        demo.object_pose[t],
                        demo.object_vel[t],
                        demo.contact[t],
                    )
                )
            obs = np.stack(rows, axis=0).astype(np.float32)
            obs_list.append(obs[:-1])
            next_obs_list.append(obs[1:])
            act_list.append(a[:-1].astype(np.float32))
            rew_list.append(np.zeros((T - 1, 1), dtype=np.float32))
            done = np.zeros((T - 1, 1), dtype=np.float32)
            done[-1] = 1.0
            done_list.append(done)
    finally:
        env.close()

    return {
        "obs": np.concatenate(obs_list, axis=0),
        "acts": np.concatenate(act_list, axis=0),
        "next_obs": np.concatenate(next_obs_list, axis=0),
        "rewards": np.concatenate(rew_list, axis=0),
        "dones": np.concatenate(done_list, axis=0).astype(np.float32),
    }
