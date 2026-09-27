"""Tests that the shipped Tier B ablation configs stay mutually consistent.

A run name alone does not determine which environment a result came from, so
the environment is recorded in each run's ``config.yaml``. These tests pin the
Tier B matrix to the Tier B environment: running the relocate config under a
Tier B run name silently produced results for a different task.
"""

from pathlib import Path

import pytest

from human2robot.config.loader import load_config

CONFIG_DIR = Path(__file__).parents[2] / "configs"
TIER_B_CONFIGS = [
    "tier_b_pickup.yaml",
    "tier_b_cond_b.yaml",
    "tier_b_cond_c.yaml",
    "tier_b_cond_d.yaml",
    "tier_b_cond_e.yaml",
]
TIER_B_ENV = "Human2Robot-AllegroPickup-v0"
TIER_B_STEPS = 2_000_000
TIER_B_ENVS = 12


@pytest.mark.parametrize("name", TIER_B_CONFIGS)
def test_tier_b_configs_target_tier_b_env(name: str) -> None:
    """Every Tier B config runs the Allegro pickup task."""
    assert load_config(CONFIG_DIR / name).env_id == TIER_B_ENV


@pytest.mark.parametrize("name", TIER_B_CONFIGS)
def test_tier_b_configs_share_training_budget(name: str) -> None:
    """Conditions differ only by ablation flags, not by budget."""
    config = load_config(CONFIG_DIR / name)
    assert config.total_env_steps == TIER_B_STEPS
    assert config.num_envs == TIER_B_ENVS


def test_condition_flags_match_the_ablation_table() -> None:
    """Each config enables exactly the demo and dynamics components it claims."""
    expected = {
        "tier_b_pickup.yaml": ("A", False, False),
        "tier_b_cond_b.yaml": ("B", True, False),
        "tier_b_cond_c.yaml": ("C", False, True),
        "tier_b_cond_d.yaml": ("D", False, True),
        "tier_b_cond_e.yaml": ("E", True, True),
    }
    for name, (letter, demo, dynamics) in expected.items():
        config = load_config(CONFIG_DIR / name)
        assert config.condition == letter, name
        assert config.demo.enabled is demo, name
        assert config.dynamics_aug.enabled is dynamics, name


def test_residual_conditions_agree_with_blackbox_conditions() -> None:
    """Dynamics hyperparameters are held fixed so the ablation is clean."""
    blackbox = load_config(CONFIG_DIR / "tier_b_cond_c.yaml").dynamics_aug
    for name in ("tier_b_cond_d.yaml", "tier_b_cond_e.yaml"):
        residual = load_config(CONFIG_DIR / name).dynamics_aug
        assert residual.ensemble_size == blackbox.ensemble_size, name
        assert residual.synthetic_ratio == blackbox.synthetic_ratio, name
        assert residual.retrain_every_steps == blackbox.retrain_every_steps, name
        assert residual.mode == "residual", name
    assert blackbox.mode == "blackbox"


def test_tier_a_config_is_a_different_environment() -> None:
    """The relocate config is deliberately not a Tier B config."""
    assert load_config(CONFIG_DIR / "tier_a_relocate.yaml").env_id != TIER_B_ENV
