"""Shared pytest fixtures."""

import pytest

from human2robot.utils.seed import seed_everything


@pytest.fixture(autouse=True)
def _seeded() -> None:
    seed_everything(0)
