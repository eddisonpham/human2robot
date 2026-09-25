"""Demo trajectory schema (agents/08 §5)."""

from dataclasses import dataclass

import numpy as np

DEMO_SCHEMA_VERSION = "1.0"


@dataclass
class DemoTrajectory:
    """Retargeted robot trajectory ready for BC/replay."""

    q: np.ndarray
    qdot: np.ndarray
    a_demo: np.ndarray
    object_pose: np.ndarray
    object_vel: np.ndarray
    contact: np.ndarray
    trajectory_id: str
    task_id: str
    source: str

    def validate(self) -> None:
        """Raise if any array violates the schema."""
        horizon = len(self.q)
        if horizon == 0:
            raise ValueError("empty trajectory")
        for name in ("q", "qdot", "a_demo"):
            arr = getattr(self, name)
            if arr.shape != (horizon, 22):
                raise ValueError(f"{name} shape {arr.shape} != ({horizon}, 22)")
        if self.object_pose.shape != (horizon, 7):
            raise ValueError(
                f"object_pose shape {self.object_pose.shape} != ({horizon}, 7)"
            )
        if self.object_vel.shape != (horizon, 6):
            raise ValueError(
                f"object_vel shape {self.object_vel.shape} != ({horizon}, 6)"
            )
        if self.contact.shape[0] != horizon:
            raise ValueError("contact horizon mismatch")
        if not 2 <= self.contact.shape[1] <= 7:
            raise ValueError("contact must have 2-7 columns")
