"""Tests for deterministic actor export."""

import numpy as np
import torch

from dynhand.config.schema import SACConfig
from dynhand.export.onnx import benchmark_latency, export_actor, validate_parity
from dynhand.rl.sac import SAC


def make_actor():
    sac = SAC(
        3,
        2,
        np.array([-1.0, -2.0], dtype=np.float32),
        np.array([1.0, 2.0], dtype=np.float32),
        SACConfig(hidden_dim=16),
        torch.device("cpu"),
    )
    sac.actor.eval()
    return sac.actor


def test_export_actor_manifest_and_parity(tmp_path) -> None:
    actor = make_actor()
    output = export_actor(actor, 3, tmp_path / "policy.onnx")
    assert output.exists()
    report = validate_parity(actor, output, 3, samples=4)
    assert report["passed"] is True
    assert report["max_error"] < 1e-5


def test_benchmark_latency_returns_percentiles(tmp_path) -> None:
    output = export_actor(make_actor(), 3, tmp_path / "policy.onnx")
    report = benchmark_latency(output, 3, samples=3, warmup=1)
    assert report["samples"] == 3.0
    assert report["p50_ms"] >= 0.0
    assert report["p95_ms"] >= report["p50_ms"]
