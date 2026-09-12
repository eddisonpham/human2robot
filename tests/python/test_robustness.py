"""Tests for robustness helpers."""

import pytest

from dynhand.evaluation.robustness import generalization_split


def test_generalization_split_detects_leakage() -> None:
    with pytest.raises(ValueError, match="leakage"):
        generalization_split(["a", "b"], ["b", "c"])
    assert generalization_split(["a"], ["b"])["leakage"] is False
