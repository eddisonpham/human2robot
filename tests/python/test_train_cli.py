"""Train CLI argument parsing and device resolution."""

import sys

import pytest
import torch

from human2robot.rl import train as train_mod
from human2robot.rl.train import main, resolve_device


@pytest.fixture
def captured_train(monkeypatch):
    """Replace train() so the CLI can be exercised without a real run."""
    calls = {}

    def fake_train(config, **kwargs):
        calls["config"] = config
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(train_mod, "train", fake_train)
    return calls


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["human2robot-train", *argv])
    main()


def test_cli_requires_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["human2robot-train"])
    with pytest.raises(SystemExit):
        main()


def test_cli_parses_seed_and_run_name(captured_train, monkeypatch, tmp_path):
    _run_cli(
        monkeypatch,
        [
            "--config",
            "configs/smoke.yaml",
            "--seed",
            "11",
            "--run-name",
            "cli_run",
            "--results-dir",
            str(tmp_path),
        ],
    )
    assert captured_train["config"].seed == 11
    assert captured_train["run_name"] == "cli_run"
    assert captured_train["run_dir_override"] == str(tmp_path)
    assert captured_train["resume"] is False
    assert captured_train["device_name"] is None


def test_cli_seed_override_is_optional(captured_train, monkeypatch):
    _run_cli(monkeypatch, ["--config", "configs/smoke.yaml"])
    assert captured_train["config"].seed != 11


def test_cli_forwards_resume_and_device(captured_train, monkeypatch):
    _run_cli(
        monkeypatch,
        ["--config", "configs/smoke.yaml", "--resume", "--device", "cpu"],
    )
    assert captured_train["resume"] is True
    assert captured_train["device_name"] == "cpu"


def test_cli_rejects_unknown_device(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["human2robot-train", "--config", "configs/smoke.yaml", "--device", "tpu"],
    )
    with pytest.raises(SystemExit):
        main()


def test_cli_rejects_missing_config_file(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["human2robot-train", "--config", "configs/does_not_exist.yaml"]
    )
    with pytest.raises(FileNotFoundError):
        main()


def test_resolve_device_explicit_names():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda").type == "cuda"


def test_resolve_device_auto_returns_torch_device():
    device = resolve_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda"}


def test_resolve_device_default_returns_torch_device():
    assert isinstance(resolve_device(None), torch.device)


def test_resolve_device_rejects_unknown_name():
    # torch.device itself rejects the string, so the CLI's --device choices are
    # the only guard a user meets before this.
    with pytest.raises((ValueError, RuntimeError)):
        resolve_device("tpu")
