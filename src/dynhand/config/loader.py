"""YAML config loading with pydantic validation."""

from pathlib import Path

import yaml

from dynhand.config.schema import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML config."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return ExperimentConfig.model_validate(raw)
