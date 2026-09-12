"""Floating Allegro MuJoCo environment for Tier B validation."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ENV_ID = "DynHand-AllegroPickup-v0"
_ACTION_DIM = 22
_FINGER_DIM = 16
_CONTACT_DIM = 5
_OBSERVATION_DIM = 64
_BASE_VEL_SCALE = np.array([0.15, 0.15, 0.15, 0.8, 0.8, 0.8], dtype=np.float64)


def _asset_path() -> Path:
    """Return the local Menagerie Allegro asset path."""
    path = (
        Path(__file__).parents[3]
        / "references"
        / "mujoco_menagerie"
        / "wonik_allegro"
        / "right_hand.xml"
    )
    if not path.exists():
        raise FileNotFoundError(
            "Allegro asset is missing; run scripts/setup_references.sh first"
        )
    return path


def _floating_model_xml() -> str:
    """Build the Tier B XML without modifying the third-party source asset."""
    source = _asset_path()
    xml = source.read_text(encoding="utf-8")
    asset_dir = (source.parent / "assets").as_posix()
    xml = xml.replace('meshdir="assets"', f'meshdir="{asset_dir}"')
    xml = xml.replace(
        '<body name="palm" quat="0 1 0 1" childclass="allegro_right">',
        '<body name="palm" pos="0 0 0.22" quat="0 1 0 1" childclass="allegro_right">'
        '<freejoint name="base_freejoint"/>',
    )
    xml = xml.replace(
        '<body name="ff_tip">',
        '<body name="ff_tip"><site name="ff_tip_site" pos="0 0 0.045" size="0.008"/>',
    )
    xml = xml.replace(
        '<body name="mf_tip">',
        '<body name="mf_tip"><site name="mf_tip_site" pos="0 0 0.045" size="0.008"/>',
    )
    xml = xml.replace(
        '<body name="rf_tip">',
        '<body name="rf_tip"><site name="rf_tip_site" pos="0 0 0.045" size="0.008"/>',
    )
    xml = xml.replace(
        '<body name="th_tip">',
        '<body name="th_tip"><site name="th_tip_site" pos="0 0 0.055" size="0.008"/>',
    )
    object_xml = """
    <geom name="floor" type="plane" pos="0 0 -0.02" size="1 1 0.05"/>
    <body name="pickup_object" pos="0 0 0.08">
      <freejoint name="object_freejoint"/>
      <geom name="pickup_object_geom" type="box" size="0.025 0.025 0.025"
            mass="0.08" friction="0.7 0.01 0.01"/>
    </body>
    <site name="pickup_target" pos="0 0 0.18" size="0.01" rgba="0 1 0 0.5"/>
    """
    xml = xml.replace(
        "  </worldbody>\n\n  <contact>", object_xml + "  </worldbody>\n\n  <contact>"
    )
    sensor_xml = """
  <sensor>
    <touch name="ff_touch" site="ff_tip_site"/>
    <touch name="mf_touch" site="mf_tip_site"/>
    <touch name="rf_touch" site="rf_tip_site"/>
    <touch name="th_touch" site="th_tip_site"/>
  </sensor>
    """
    xml = xml.replace("\n</mujoco>", sensor_xml + "\n</mujoco>")
    return xml


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a bounded axis-angle vector."""
    trace = float(np.trace(matrix))
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        return np.zeros(3, dtype=np.float32)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(angle)
    return (axis * angle).astype(np.float32)


class AllegroPickupEnv(gym.Env[np.ndarray, np.ndarray]):
    """Floating Allegro reach and pickup task.

    The first six action values are bounded base linear/angular velocity
    commands because the source Menagerie model has no floating-base
    actuators. The remaining sixteen values are normalized finger position
    targets mapped to the source position actuator ranges.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 500) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(_floating_model_xml())
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (_OBSERVATION_DIM,), dtype=np.float32
        )
        self._finger_qpos = np.array(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                for name in self._finger_joint_names()
            ],
            dtype=np.int32,
        )
        self._finger_qvel = np.array(
            [
                self.model.jnt_dofadr[self.model.joint(name).id]
                for name in self._finger_joint_names()
            ],
            dtype=np.int32,
        )
        self._base_qpos = self.model.jnt_qposadr[self.model.joint("base_freejoint").id]
        self._base_qvel = self.model.jnt_dofadr[self.model.joint("base_freejoint").id]
        self._object_qpos = self.model.jnt_qposadr[
            self.model.joint("object_freejoint").id
        ]
        self._object_qvel = self.model.jnt_dofadr[
            self.model.joint("object_freejoint").id
        ]
        self._actuator_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._actuator_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._object_initial_z = 0.08
        self._step_count = 0
        self._success_frames = 0
        self._rng = np.random.default_rng()
        self._finger_joints_cache = None

    @staticmethod
    def _finger_joint_names() -> list[str]:
        """Return the source asset's sixteen actuated finger joints."""
        return [
            f"{prefix}j{index}"
            for prefix in ("ff", "mf", "rf", "th")
            for index in range(4)
        ]

    def reset(self, *, seed: int | None = None, options=None):
        """Reset the hand and object to a deterministic initial state."""
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        base = self.data.qpos[self._base_qpos : self._base_qpos + 7]
        base[:] = [0.0, 0.0, 0.22, 1.0, 0.0, 0.0, 0.0]
        object_state = self.data.qpos[self._object_qpos : self._object_qpos + 7]
        object_state[:] = [0.0, 0.0, self._object_initial_z, 1.0, 0.0, 0.0, 0.0]
        object_state[:2] += self._rng.uniform(-0.01, 0.01, size=2)
        self.data.qpos[self._finger_qpos] = 0.2
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._success_frames = 0
        return self._observation(), {"success": False}

    def step(self, action: np.ndarray):
        """Apply one control action and advance the MuJoCo simulation."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_space.shape[0],):
            raise ValueError(f"expected action shape {(22,)}, got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        self.data.qvel[self._base_qvel : self._base_qvel + 6] = (
            action[:6] * _BASE_VEL_SCALE
        )
        self.data.ctrl[:] = self._actuator_low + (action[6:] + 1.0) * 0.5 * (
            self._actuator_high - self._actuator_low
        )
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        object_z = float(self.data.qpos[self._object_qpos + 2])
        lifted = object_z - self._object_initial_z > 0.04
        self._success_frames = self._success_frames + 1 if lifted else 0
        success = self._success_frames >= 5
        palm = self.data.xpos[self.model.body("palm").id]
        object_pos = self.data.qpos[self._object_qpos : self._object_qpos + 3]
        approach = -float(np.sum((palm - object_pos) ** 2))
        lift = max(0.0, object_z - self._object_initial_z)
        energy = float(np.sum(self.data.actuator_force**2)) * 1e-5
        reward = approach + 2.0 * lift - energy + (1.0 if success else 0.0)
        terminated = bool(success)
        truncated = self._step_count >= self.max_episode_steps
        return self._observation(), reward, terminated, truncated, {"success": success}

    def _observation(self) -> np.ndarray:
        """Build the 64-dimensional state vector from MuJoCo data."""
        base_qpos = self.data.qpos[self._base_qpos : self._base_qpos + 7]
        base_rot = np.zeros((3, 3), dtype=np.float64)
        mujoco.mju_quat2Mat(base_rot.ravel(), base_qpos[3:])
        base_q = np.concatenate(
            (
                base_qpos[:3],
                _rotation_vector(base_rot),
                self.data.qpos[self._finger_qpos],
            )
        )
        base_qvel = np.concatenate(
            (
                self.data.qvel[self._base_qvel : self._base_qvel + 6],
                self.data.qvel[self._finger_qvel],
            )
        )
        object_qpos = self.data.qpos[self._object_qpos : self._object_qpos + 7]
        object_rot = np.zeros((3, 3), dtype=np.float64)
        mujoco.mju_quat2Mat(object_rot.ravel(), object_qpos[3:])
        object_qvel = self.data.qvel[self._object_qvel : self._object_qvel + 6]
        contact = np.array(
            [self.data.sensordata[i] > 1e-6 for i in range(4)] + [False],
            dtype=np.float32,
        )
        target = np.array([0.0, 0.0, 0.18], dtype=np.float32)
        observation = np.concatenate(
            (
                base_q,
                base_qvel,
                object_qpos[:3],
                _rotation_vector(object_rot),
                object_qvel,
                contact,
                target,
            )
        )
        return observation.astype(np.float32)

    def _finger_joints(self) -> list[str]:
        """Return cached source joint names."""
        if self._finger_joints_cache is None:
            self._finger_joints_cache = self._finger_joint_names()
        return self._finger_joints_cache

    def close(self) -> None:
        """Release MuJoCo data owned by the environment."""
        self.data = None


def register_allegro() -> None:
    """Register the Tier B environment once."""
    if ENV_ID not in gym.registry:
        gym.register(ENV_ID, entry_point="dynhand.envs.allegro:AllegroPickupEnv")


register_allegro()
