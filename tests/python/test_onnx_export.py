"""Tests for deterministic actor export."""

import json
import sys

import numpy as np
import pytest
import torch

from human2robot.config.schema import SACConfig
from human2robot.export.onnx import benchmark_latency, export_actor, validate_parity
from human2robot.rl.sac import SAC


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


def _write_checkpoint(tmp_path, obs_dim: int = 3, act_dim: int = 2):
    """Save a SAC checkpoint in the layout _build_sac expects."""
    import torch

    from human2robot.config.schema import SACConfig

    sac = SAC(
        obs_dim,
        act_dim,
        np.full(act_dim, -1.0, dtype=np.float32),
        np.full(act_dim, 1.0, dtype=np.float32),
        SACConfig(hidden_dim=16),
        torch.device("cpu"),
    )
    path = tmp_path / "sac.pt"
    torch.save({"sac": sac.state_dict()}, path)
    return path


def test_build_sac_loads_actor_from_checkpoint(tmp_path) -> None:
    from human2robot.config.schema import ExperimentConfig
    from human2robot.export.onnx import _build_sac

    config = ExperimentConfig(
        experiment_id="onnx_unit",
        env_id="Pendulum-v1",
        total_env_steps=10,
        sac=SACConfig(hidden_dim=16),
    )
    sac = _build_sac(config, _write_checkpoint(tmp_path, act_dim=1))
    assert not sac.actor.training


def test_build_sac_rejects_missing_checkpoint(tmp_path) -> None:
    from human2robot.config.schema import ExperimentConfig
    from human2robot.export.onnx import _build_sac

    config = ExperimentConfig(
        experiment_id="onnx_unit",
        env_id="Pendulum-v1",
        total_env_steps=10,
        sac=SACConfig(hidden_dim=16),
    )
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        _build_sac(config, tmp_path / "absent.pt")


def test_export_checkpoint_writes_model_manifest_and_report(tmp_path) -> None:
    from human2robot.export.onnx import export_checkpoint

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment_id: onnx_smoke\nenv_id: Pendulum-v1\nseed: 0\n"
        "total_env_steps: 400\nnum_envs: 2\nstart_steps: 100\n"
        "checkpoint_interval: 200\nsac:\n  hidden_dim: 16\n",
        encoding="utf-8",
    )
    out = tmp_path / "policy.onnx"
    result = export_checkpoint(config_path, _write_checkpoint(tmp_path, act_dim=1), out)
    assert out.exists()
    assert out.with_suffix(".json").exists()
    assert out.with_suffix(".report.json").exists()
    assert result["parity"]["passed"] is True
    assert result["latency"]["samples"] == 1000.0
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["model"] == str(out)


def test_export_checkpoint_raises_when_parity_fails(tmp_path, monkeypatch) -> None:
    from human2robot.export import onnx as onnx_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment_id: onnx_smoke\nenv_id: Pendulum-v1\nseed: 0\n"
        "total_env_steps: 400\nnum_envs: 2\nstart_steps: 100\n"
        "checkpoint_interval: 200\nsac:\n  hidden_dim: 16\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        onnx_module,
        "validate_parity",
        lambda *args, **kwargs: {"passed": False, "max_error": 9.0},
    )
    with pytest.raises(ValueError, match="parity check failed"):
        onnx_module.export_checkpoint(
            config_path, _write_checkpoint(tmp_path, act_dim=1), tmp_path / "p.onnx"
        )


def test_export_checkpoint_uses_explicit_manifest_path(tmp_path) -> None:
    from human2robot.export.onnx import export_checkpoint

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment_id: onnx_smoke\nenv_id: Pendulum-v1\nseed: 0\n"
        "total_env_steps: 400\nnum_envs: 2\nstart_steps: 100\n"
        "checkpoint_interval: 200\nsac:\n  hidden_dim: 16\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "nested" / "contract.json"
    onnx_out = tmp_path / "policy.onnx"
    export_checkpoint(
        config_path,
        _write_checkpoint(tmp_path, act_dim=1),
        onnx_out,
        manifest_path=manifest,
    )
    assert manifest.exists()
    assert not onnx_out.with_suffix(".json").exists()
    contract = json.loads(manifest.read_text())
    assert contract["input_name"] == "observation"
    assert contract["output_name"] == "action"


def test_onnx_cli_main_prints_report(tmp_path, monkeypatch, capsys) -> None:
    from human2robot.export import onnx as onnx_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment_id: onnx_smoke\nenv_id: Pendulum-v1\nseed: 0\n"
        "total_env_steps: 400\nnum_envs: 2\nstart_steps: 100\n"
        "checkpoint_interval: 200\nsac:\n  hidden_dim: 16\n",
        encoding="utf-8",
    )
    out = tmp_path / "policy.onnx"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "human2robot-onnx",
            "--config",
            str(config_path),
            "--checkpoint",
            str(_write_checkpoint(tmp_path, act_dim=1)),
            "--out",
            str(out),
        ],
    )
    onnx_module.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["parity"]["passed"] is True
