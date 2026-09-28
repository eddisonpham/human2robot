"""Nominal physics transition provider for residual dynamics augmentation.

Supplies the real MuJoCo one-step observation delta directly in the policy
observation space, so the residual ensemble can learn

    residual = true_delta - physics_delta

with no qpos/qvel conversions in between. ``compute_obs_delta`` is the only
entry point: it reproduces a real ``AllegroPickupEnv`` step, which
``tests/python/test_nominal_physics.py`` pins against the live environment.
"""

from __future__ import annotations

import mujoco
import numpy as np

from human2robot.envs.allegro import ENV_STEP_SUBSTEPS
from human2robot.utils.rotation import rotation_vector as _rotation_vector

# The physics delta must cover exactly the sub-steps the env applies per action,
# so the count is imported from the env rather than duplicated here.
_ENV_STEP_SUBSTEPS = ENV_STEP_SUBSTEPS

# The floating Allegro env treats action[:6] as a direct base-qvel override
# rather than an actuator command, so the nominal step must apply it to match
# the environment the delta is supposed to describe.
_BASE_VEL_SCALE = np.array([0.15, 0.15, 0.15, 0.8, 0.8, 0.8], dtype=np.float64)


def compute_obs_delta(
    model: mujoco._structs.MjModel,
    obs: np.ndarray,
    action: np.ndarray,
    *,
    ctrl_from_action: bool = True,
    action_scale_low: np.ndarray | None = None,
    action_scale_high: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 64-d observation delta from one real MuJoCo step of (obs, action).

    This is the single-source-of-truth physics delta for residual dynamics
    training: it steps the real simulator and returns obs_next - obs.
    """
    action = np.asarray(action, dtype=np.float32).copy()
    if action.ndim == 1:
        action = action.reshape(1, -1)
    if ctrl_from_action:
        ctrl = _action_to_ctrl(model, action, action_scale_low, action_scale_high)
    else:
        # Already two-dimensional by this point, so no reshape is needed here.
        ctrl = np.asarray(action, dtype=np.float64).copy()

    data = mujoco.MjData(model)
    _set_state_from_obs(model, data, obs)
    _apply_action(model, data, action, ctrl)

    for _ in range(_ENV_STEP_SUBSTEPS):
        mujoco.mj_step(model, data)

    obs_next = _obs_from_data(model, data)
    return (obs_next - obs).astype(np.float32)


def _obs_from_data(
    model: mujoco._structs.MjModel, data: mujoco._structs.MjData
) -> np.ndarray:
    """Rebuild the 64-d policy observation from *data* after a step."""
    base_qpos = data.qpos[0:7]
    base_rot = np.zeros((3, 3), dtype=np.float64)
    mujoco.mju_quat2Mat(base_rot.ravel(), base_qpos[3:])
    base_q = np.concatenate(
        (
            base_qpos[:3],
            _rotation_vector(base_rot),
            data.qpos[7 : 7 + 16],
        )
    )
    base_qvel = np.concatenate(
        (
            data.qvel[0:6],
            data.qvel[6 : 6 + 16],
        )
    )
    object_qpos = data.qpos[23 : 23 + 7]
    object_rot = np.zeros((3, 3), dtype=np.float64)
    mujoco.mju_quat2Mat(object_rot.ravel(), object_qpos[3:])
    object_rot_vec = _rotation_vector(object_rot)
    object_qvel = data.qvel[22 : 22 + 6]
    contact = np.array(
        [data.sensordata[i] > 1e-6 for i in range(4)] + [False],
        dtype=np.float64,
    )
    target = np.array([0.0, 0.0, 0.18], dtype=np.float64)
    return np.concatenate(
        (
            base_q,
            base_qvel,
            object_qpos[:3],
            object_rot_vec,
            object_qvel,
            contact,
            target,
        )
    ).astype(np.float64)


def _apply_action(
    model: mujoco._structs.MjModel,
    data: mujoco._structs.MjData,
    action: np.ndarray,
    ctrl: np.ndarray,
) -> None:
    """Apply the env's control semantics: base-qvel override plus actuator ctrl.

    Mirrors AllegroPickupEnv.step, which overwrites the free joint's velocity
    from action[:6] and maps action[6:] into the actuator ctrl range.
    """
    data.qvel[0:6] = action[0, 0:6] * _BASE_VEL_SCALE
    data.ctrl[:] = ctrl[0]


def _action_to_ctrl(
    model: mujoco._structs.MjModel,
    action: np.ndarray,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> np.ndarray:
    """Map the 22-d normalized action to the 16-d actuator ctrl MuJoCo expects."""
    action = np.asarray(action, dtype=np.float64).copy()
    if action.ndim == 1:
        action = action.reshape(1, -1)
    ctrl = np.zeros((action.shape[0], int(model.nu)), dtype=np.float64)
    act_low = low if low is not None else model.actuator_ctrlrange[:, 0]
    act_high = high if high is not None else model.actuator_ctrlrange[:, 1]
    finger_action = action[:, 6 : 6 + int(model.nu)]
    ctrl[:] = act_low + (finger_action + 1.0) * 0.5 * (act_high - act_low)
    ctrl = np.clip(ctrl, act_low, act_high)
    return ctrl


def _set_state_from_obs(
    model: mujoco._structs.MjModel, data: mujoco._structs.MjData, obs: np.ndarray
) -> None:
    """Populate qpos/qvel in *data* from the 64-d policy observation."""
    obs = np.asarray(obs, dtype=np.float64).copy()
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)

    qpos = np.zeros((obs.shape[0], int(model.nq)), dtype=np.float64)
    qpos[:, 0:3] = obs[:, 0:3]
    qpos[:, 3:7] = _rotvec_to_quat(obs[:, 3:6])
    qpos[:, 7 : 7 + 16] = obs[:, 6:22]
    qpos[:, 23 : 23 + 3] = obs[:, 44:47]
    qpos[:, 23 + 3 : 23 + 7] = _rotvec_to_quat(obs[:, 47:50])

    qvel = np.zeros((obs.shape[0], int(model.nv)), dtype=np.float64)
    qvel[:, 0:6] = obs[:, 22:28]
    qvel[:, 6 : 6 + 16] = obs[:, 28:44]
    qvel[:, 22 : 22 + 6] = obs[:, 50:56]

    data.qpos[:] = qpos[0]
    data.qvel[:] = qvel[0]
    data.time = 0.0


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector to quaternion in MuJoCo [w,x,y,z] order."""
    rotvec = np.asarray(rotvec, dtype=np.float64).copy()
    if rotvec.ndim == 1:
        rotvec = rotvec.reshape(1, 3)
    q = np.zeros((rotvec.shape[0], 4), dtype=np.float64)
    for i in range(rotvec.shape[0]):
        angle = float(np.linalg.norm(rotvec[i]))
        if angle < 1e-12:
            q[i] = [1.0, 0.0, 0.0, 0.0]
            continue
        axis = rotvec[i] / angle
        half = angle / 2.0
        s = float(np.sin(half))
        c = float(np.cos(half))
        q[i, 0] = c
        q[i, 1:4] = axis * s
    return q[0]
