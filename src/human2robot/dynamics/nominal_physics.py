"""Nominal physics transition provider for residual dynamics augmentation.

Provides the real MuJoCo one-step observation delta for residual dynamics
training.  The residual ensemble predicts delta = f(s, a) and uses
next_state = s + physics_delta + f(s, a); this module supplies physics_delta
directly in the policy observation space so no qpos/qvel <-> obs conversions
are needed.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from human2robot.dynamics.ensemble import DynamicsEnsemble

# _ENV_STEP_SUBSTEPS mirrors the env's control_decimation so the physics
# delta corresponds to the same number of MuJoCo sub-steps the env applies
# per action.
_ENV_STEP_SUBSTEPS = 10


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
        ctrl = np.asarray(action, dtype=np.float64).copy()
        if ctrl.ndim == 1:
            ctrl = ctrl.reshape(1, -1)

    data = mujoco.MjData(model)
    _set_state_from_obs(model, data, obs)

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


def compute_physics_deltas(
    model: mujoco._structs.MjModel,
    state: np.ndarray,
    action: np.ndarray,
    *,
    ctrl_from_action: bool = True,
    action_scale_low: np.ndarray | None = None,
    action_scale_high: np.ndarray | None = None,
) -> np.ndarray:
    """Return one-step MuJoCo qpos and qvel deltas for (state, action).

    Retained for compatibility; new code should use compute_obs_delta.
    """
    action = np.asarray(action, dtype=np.float32).copy()
    if action.ndim == 1:
        action = action.reshape(1, -1)
    if ctrl_from_action:
        ctrl = _action_to_ctrl(model, action, action_scale_low, action_scale_high)
    else:
        ctrl = np.asarray(action, dtype=np.float64).copy()
        if ctrl.ndim == 1:
            ctrl = ctrl.reshape(1, -1)

    data = mujoco.MjData(model)
    _set_state_from_obs(model, data, state)

    for _ in range(_ENV_STEP_SUBSTEPS):
        mujoco.mj_step(model, data)

    qpos_delta = data.qpos - _initial_qpos_from_obs(model, state)
    qvel_delta = data.qvel - _initial_qvel_from_obs(model, state)
    return np.concatenate([qpos_delta, qvel_delta], dtype=np.float32)


def _quat_delta_to_rotvec_delta(
    quat_delta: np.ndarray, quat_before: np.ndarray
) -> np.ndarray:
    """Convert small quaternion delta to rotation vector delta.

    rotvec_delta ≈ 2 * quat_delta.xyz for small angles.
    """
    rotvec_delta = 2.0 * quat_delta[:, 0:3]
    return rotvec_delta[0] if rotvec_delta.shape[0] == 1 else rotvec_delta


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a bounded axis-angle vector."""
    trace = float(np.trace(matrix))
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(angle)
    return (axis * angle).astype(np.float64)


def _physics_delta_to_obs_delta(
    model: mujoco._structs.MjModel,
    obs: np.ndarray,
    phys_delta: np.ndarray,
) -> np.ndarray:
    """Convert physics delta (nq+nv space) to observation delta (64-d).

    Retained for callers that still have a qpos/qvel-space delta.  Prefer
    compute_obs_delta for new code.
    """
    obs = np.asarray(obs, dtype=np.float64).copy()
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    phys_delta = np.asarray(phys_delta, dtype=np.float64).copy()
    if phys_delta.ndim == 1:
        phys_delta = phys_delta.reshape(1, -1)

    obs_delta = np.zeros((obs.shape[0], 64), dtype=np.float64)

    # base_q: obs[0:3] xyz, obs[3:6] rotvec, obs[6:22] finger_qpos
    obs_delta[:, 0:3] = phys_delta[:, 0:3]  # base xyz
    obs_delta[:, 3:6] = _quat_delta_to_rotvec_delta(
        phys_delta[:, 3:7], obs[:, 3:6]
    )  # base rotvec from quat delta
    obs_delta[:, 6:22] = phys_delta[:, 7:23]  # finger qpos (16)

    # base_qvel: obs[22:28] base_qvel, obs[28:44] finger_qvel
    obs_delta[:, 22:28] = phys_delta[:, 30:36]  # base qvel
    obs_delta[:, 28:44] = phys_delta[:, 36:52]  # finger qvel (16)

    # object state: obs[44:47] xyz, obs[47:50] rotvec, obs[50:56] qvel
    obs_delta[:, 44:47] = phys_delta[:, 23:26]  # object xyz
    obs_delta[:, 47:50] = _quat_delta_to_rotvec_delta(
        phys_delta[:, 26:30], obs[:, 47:50]
    )  # object rotvec from quat delta
    obs_delta[:, 50:56] = phys_delta[:, 52:58]  # object qvel (6)

    # contact: computed from simulation, cannot be derived from physics delta
    # target: constant, so delta = 0
    return obs_delta[0] if obs.shape[0] == 1 else obs_delta


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


def _initial_qpos_from_obs(
    model: mujoco._structs.MjModel, obs: np.ndarray
) -> np.ndarray:
    """Return the qpos we would set from obs, as a 1-d array."""
    obs = np.asarray(obs, dtype=np.float64).copy()
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    qpos = np.zeros((int(model.nq),), dtype=np.float64)
    qpos[0:3] = obs[0, 0:3]
    qpos[3:7] = _rotvec_to_quat(obs[0, 3:6])
    qpos[7 : 7 + 16] = obs[0, 6:22]
    qpos[23 : 23 + 3] = obs[0, 44:47]
    qpos[23 + 3 : 23 + 7] = _rotvec_to_quat(obs[0, 47:50])
    return qpos


def _initial_qvel_from_obs(
    model: mujoco._structs.MjModel, obs: np.ndarray
) -> np.ndarray:
    """Return the qvel we would set from obs, as a 1-d array."""
    obs = np.asarray(obs, dtype=np.float64).copy()
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    qvel = np.zeros((int(model.nv),), dtype=np.float64)
    qvel[0:6] = obs[0, 22:28]
    qvel[6 : 6 + 16] = obs[0, 28:44]
    qvel[22 : 22 + 6] = obs[0, 50:56]
    return qvel


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector to quaternion [x,y,z,w]."""
    rotvec = np.asarray(rotvec, dtype=np.float64).copy()
    if rotvec.ndim == 1:
        rotvec = rotvec.reshape(1, 3)
    q = np.zeros((rotvec.shape[0], 4), dtype=np.float64)
    for i in range(rotvec.shape[0]):
        angle = float(np.linalg.norm(rotvec[i]))
        if angle < 1e-12:
            q[i] = [0.0, 0.0, 0.0, 1.0]
            continue
        axis = rotvec[i] / angle
        half = angle / 2.0
        s = float(np.sin(half))
        c = float(np.cos(half))
        q[i, 0:3] = axis * s
        q[i, 3] = c
    return q[0]


def physics_delta_from_ensemble(
    ensemble: DynamicsEnsemble,
    model: mujoco._structs.MjModel,
    states: np.ndarray,
    actions: np.ndarray,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """Compatibility wrapper: ensemble.predict already accepts physics_deltas.

    This exists so call sites that already have a DynamicsEnsemble can ask
    for the nominal physics delta using the same signature the ensemble uses,
    without needing to import _nominal_mujoco directly.
    """
    raise NotImplementedError(
        "physics_delta_from_ensemble is not wired yet; "
        "use compute_physics_deltas directly"
    )
