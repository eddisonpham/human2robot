"""C++ trajectory optimization over stored demonstration NPZs.

Reads retargeted demonstrations, runs the C++ optimizer, and writes a
parallel tree of optimized demos plus a metrics manifest. Raw demos are
never modified (docs/HANDOFF_RESPONSE.md section 11).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from human2robot.cpp_bindings import OptimizerConfig, optimize_trajectory
from human2robot.data.allegro_demos import load_demo_npz

_ALLEGRO_LOWER = np.array(
    [0.0] * 6 + [-0.47] + [0.196] * 3 + [-0.175] + [0.0] * 8 + [-0.8] * 3
)
_ALLEGRO_UPPER = np.array(
    [0.0] * 6 + [0.47] + [1.61] * 3 + [1.72] + [1.57] * 8 + [0.0] * 3
)
_ALLEGRO_MAX_VELOCITY = np.full(22, 2.0)
_ALLEGRO_MAX_ACCELERATION = np.full(22, 20.0)


class OptimizationConfig(BaseModel):
    """All hyperparameters for the C++ optimization pipeline."""

    model_config = ConfigDict(extra="forbid")

    dt: float = Field(default=0.01, gt=0.0)
    tracking: float = Field(default=1.0, ge=0.0)
    velocity: float = Field(default=0.1, ge=0.0)
    acceleration: float = Field(default=0.1, ge=0.0)
    jerk: float = Field(default=0.05, ge=0.0)
    collision: float = Field(default=0.0, ge=0.0)
    limits_weight: float = Field(default=10.0, ge=0.0)
    max_iterations: int = Field(default=300, ge=1)
    convergence_tolerance: float = Field(default=1e-4, gt=0.0)
    step_size: float = Field(default=0.5, gt=0.0)
    # Independent per-timestep noise is adversarial for a smoothness-dominated
    # cost, so the default is 0: a pure tracking step, which reaches a 94/100
    # descent rate where the noisy step reached 24/100.
    noise_scale: float = Field(default=0.0, ge=0.0)
    seed: int = 0

    def to_optimizer_config(self, dof: int = 22) -> OptimizerConfig:
        return OptimizerConfig(
            dof=dof,
            lower=_ALLEGRO_LOWER[:dof],
            upper=_ALLEGRO_UPPER[:dof],
            max_velocity=_ALLEGRO_MAX_VELOCITY[:dof],
            max_acceleration=_ALLEGRO_MAX_ACCELERATION[:dof],
            tracking=self.tracking,
            velocity=self.velocity,
            acceleration=self.acceleration,
            jerk=self.jerk,
            collision=self.collision,
            limits_weight=self.limits_weight,
            max_iterations=self.max_iterations,
            convergence_tolerance=self.convergence_tolerance,
            step_size=self.step_size,
            noise_scale=self.noise_scale,
            seed=self.seed,
        )


def optimize_demo_file(
    path: str | Path,
    config: OptimizationConfig,
    output_path: str | Path | None = None,
) -> dict:
    """Optimize one demo NPZ; writes <stem>_opt.npz beside it by default."""
    demo = load_demo_npz(path)
    result = optimize_trajectory(
        initial=demo.q,
        reference=demo.q,
        dt=config.dt,
        config=config.to_optimizer_config(dof=demo.q.shape[1]),
    )
    optimized = np.asarray(result.trajectory, dtype=np.float32)
    metrics = {
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "iterations": result.iterations,
        "converged": result.converged,
    }
    if output_path is None:
        output_path = Path(path).with_name(Path(path).stem + "_opt.npz")
    np.savez_compressed(
        Path(output_path),
        q=optimized,
        qdot=demo.qdot,
        a_demo=demo.a_demo,
        object_pose=demo.object_pose,
        object_vel=demo.object_vel,
        contact=demo.contact,
        trajectory_id=np.array(demo.trajectory_id + "_opt"),
        task_id=np.array(demo.task_id),
        source=np.array(demo.source + "_optimized"),
        schema_version=np.array(0),
    )
    return metrics


def optimize_demo_directory(
    demo_dir: str | Path,
    output_dir: str | Path,
    config: OptimizationConfig | None = None,
) -> dict:
    """Optimize every demo NPZ in demo_dir into output_dir with a manifest."""
    if config is None:
        config = OptimizationConfig()
    demo_dir = Path(demo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(demo_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no NPZ demos in {demo_dir}")
    started = time.perf_counter()
    per_file = {}
    for path in paths:
        per_file[path.name] = optimize_demo_file(
            path, config, output_dir / (path.stem + "_opt.npz")
        )
    manifest = {
        "config": config.model_dump(),
        "wall_seconds": time.perf_counter() - started,
        "count": len(paths),
        "per_file": per_file,
    }
    (output_dir / "optimization_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest
