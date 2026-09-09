"""Tests for config schema, loading, and condition mapping."""

from pathlib import Path

import pytest
import yaml

from dynhand.config.conditions import CONDITION_FLAGS, apply_condition
from dynhand.config.loader import load_config
from dynhand.config.schema import DemoConfig, DynamicsAugConfig, ExperimentConfig


def test_defaults_are_valid() -> None:
    config = ExperimentConfig(experiment_id="x")
    assert config.sac.gamma == 0.99
    assert config.sac.buffer_size == 1_000_000
    assert config.condition == "A"


def test_load_config_from_yaml(tmp_path: Path) -> None:
    raw = {"experiment_id": "yaml_test", "seed": 3, "total_env_steps": 100}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config.seed == 3
    assert config.total_env_steps == 100


def test_unknown_field_fails_validation(tmp_path: Path) -> None:
    raw = {"experiment_id": "bad", "not_a_field": 1}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_non_mapping_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("condition", ["A", "B", "C", "D", "E"])
def test_condition_table_matches_flags(condition: str) -> None:
    flags = CONDITION_FLAGS[condition]
    demo_on = flags["demo_enabled"]
    dyn_on = flags["dynamics_enabled"]
    expected = {
        "A": (False, False),
        "B": (True, False),
        "C": (False, True),
        "D": (False, True),
        "E": (True, True),
    }[condition]
    assert (demo_on, dyn_on) == expected


def test_apply_condition_forces_flags() -> None:
    demo = DemoConfig(enabled=False)
    dyn = DynamicsAugConfig(enabled=True, mode="blackbox")
    apply_condition("E", demo, dyn)
    assert demo.enabled is True
    assert dyn.enabled is True
    assert dyn.mode == "residual"


def test_apply_condition_disables_for_a(tmp_path: Path) -> None:
    raw = {"experiment_id": "a", "condition": "A", "demo": {"enabled": True}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config.demo.enabled is False
    assert config.dynamics_aug.enabled is False
