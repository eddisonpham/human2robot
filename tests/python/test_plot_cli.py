"""Tests for the plot CLI argument parsing."""

from pathlib import Path

from dynhand.evaluation.plot_cli import parse_runs


def test_parse_single_run() -> None:
    runs = parse_runs("A=results/run_a")
    assert set(runs.keys()) == {"A"}
    assert runs["A"] == [Path("results/run_a") / "metrics.jsonl"]


def test_parse_multiple_runs_with_spaces() -> None:
    runs = parse_runs("SAC=results/a, SAC+demo=results/b , C=results/c")
    assert set(runs.keys()) == {"SAC", "SAC+demo", "C"}
    assert runs["SAC+demo"] == [Path("results/b") / "metrics.jsonl"]


def test_parse_ignores_empty_items() -> None:
    runs = parse_runs("A=results/a,,")
    assert set(runs.keys()) == {"A"}
