"""Ablation condition definitions.

The five conditions from agents/05_RL_ALGORITHM_SPEC.md section 4 are
expressed purely as config flags. The canonical table lives in the config
schema next to the validator that enforces it; this module re-exports it
as the single import point for the rest of the codebase.
"""

from dynhand.config.schema import CONDITION_FLAGS, apply_condition

__all__ = ["CONDITION_FLAGS", "apply_condition"]
