"""Schema-validated experiment configuration.

Every run is driven by one YAML file validated against these models, so a
typo fails at load time instead of hours into training.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONDITION_FLAGS: dict[str, dict[str, object]] = {
    "A": {
        "demo_enabled": False,
        "dynamics_enabled": False,
        "dynamics_mode": "blackbox",
    },
    "B": {"demo_enabled": True, "dynamics_enabled": False, "dynamics_mode": "blackbox"},
    "C": {"demo_enabled": False, "dynamics_enabled": True, "dynamics_mode": "blackbox"},
    "D": {"demo_enabled": False, "dynamics_enabled": True, "dynamics_mode": "residual"},
    "E": {"demo_enabled": True, "dynamics_enabled": True, "dynamics_mode": "residual"},
}


def apply_condition(
    condition: str,
    demo: "DemoConfig",
    dynamics: "DynamicsAugConfig",
) -> None:
    """Force demo and dynamics flags to match the condition table."""
    flags = CONDITION_FLAGS[condition]
    demo.enabled = bool(flags["demo_enabled"])
    dynamics.enabled = bool(flags["dynamics_enabled"])
    dynamics.mode = str(flags["dynamics_mode"])  # type: ignore[assignment]


class PhysicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestep: float = 0.002
    control_decimation: int = 10
    episode_steps: int = 500


class SACConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = Field(default=1_000_000, ge=1)
    utd_ratio: int = Field(default=1, ge=1)
    target_entropy_scale: float = -1.0
    hidden_dim: int = 256


class DemoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    minari_dataset: str | None = None
    bc_epochs: int = 100
    bc_batch_size: int = 256
    bc_lr: float = 3e-4
    bc_patience: int = 10
    bc_holdout_fraction: float = 0.1
    demo_ratio_start: float = 0.5
    demo_ratio_anneal_steps: int = 300_000
    demo_dir: str | None = None


class DynamicsAugConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: Literal["blackbox", "residual"] = "blackbox"
    ensemble_size: int = Field(default=5, ge=1)
    rollout_horizon: int = Field(default=1, ge=1)
    synthetic_ratio: float = 0.5
    retrain_every_steps: int = 250
    lr: float = 3e-4
    batch_size: int = 256


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interval_steps: int = 10_000
    episodes: int = 5


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    env_id: str = "AdroitHandRelocate-v1"
    seed: int = 0
    condition: Literal["A", "B", "C", "D", "E"] = "A"
    total_env_steps: int = Field(default=1_000_000, ge=1)
    num_envs: int = Field(default=1, ge=1)
    start_steps: int = Field(default=5_000, ge=0)
    results_dir: str = "results"
    checkpoint_interval: int = 50_000
    physics: PhysicsConfig = PhysicsConfig()
    sac: SACConfig = SACConfig()
    demo: DemoConfig = DemoConfig()
    dynamics_aug: DynamicsAugConfig = DynamicsAugConfig()
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def _apply_condition_flags(self) -> "ExperimentConfig":
        apply_condition(self.condition, self.demo, self.dynamics_aug)
        return self
