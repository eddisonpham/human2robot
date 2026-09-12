"""Deterministic SAC actor export and validation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dynhand.config.loader import load_config
from dynhand.evaluation.evaluate import build_single_env
from dynhand.rl.sac import SAC


def _build_sac(config, checkpoint: str | Path) -> SAC:
    """Load a SAC actor using dimensions from the configured environment."""
    env = build_single_env(config.env_id)
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        sac = SAC(
            obs_dim,
            act_dim,
            env.action_space.low.astype(np.float32),
            env.action_space.high.astype(np.float32),
            config.sac,
            torch.device("cpu"),
        )
    finally:
        env.close()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sac.load_state_dict(state["sac"])
    sac.actor.eval()
    return sac


class _DeterministicActor(torch.nn.Module):
    """Export wrapper exposing only deterministic actor inference."""

    def __init__(self, actor: torch.nn.Module) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action for a batch of observations."""
        return self.actor.deterministic(observation)


def export_actor(
    actor: torch.nn.Module,
    observation_dim: int,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    opset_version: int = 18,
) -> Path:
    """Export a deterministic actor and write its tensor contract."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _DeterministicActor(actor).eval()
    sample = torch.zeros(1, observation_dim, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        sample,
        output,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,
    )
    manifest = {
        "format": "onnx",
        "opset_version": opset_version,
        "input_name": "observation",
        "output_name": "action",
        "observation_shape": [None, observation_dim],
        "action_shape": [None, int(actor.action_scale.numel())],
        "dtype": "float32",
    }
    manifest_file = (
        Path(manifest_path) if manifest_path else output.with_suffix(".json")
    )
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def validate_parity(
    actor: torch.nn.Module,
    model_path: str | Path,
    observation_dim: int,
    samples: int = 16,
    atol: float = 1e-5,
) -> dict[str, float | bool]:
    """Compare PyTorch deterministic actions with ONNX Runtime outputs."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    observations = rng.normal(size=(samples, observation_dim)).astype(np.float32)
    with torch.no_grad():
        expected = actor.deterministic(torch.from_numpy(observations)).cpu().numpy()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(["action"], {"observation": observations})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    return {
        "passed": bool(np.allclose(expected, actual, atol=atol)),
        "max_error": max_error,
    }


def benchmark_latency(
    model_path: str | Path,
    observation_dim: int,
    samples: int = 1000,
    warmup: int = 100,
) -> dict[str, float]:
    """Measure ONNX Runtime batch-one inference latency in milliseconds."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    observation = np.zeros((1, observation_dim), dtype=np.float32)
    for _ in range(warmup):
        session.run(["action"], {"observation": observation})
    timings = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        session.run(["action"], {"observation": observation})
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    values = np.asarray(timings)
    return {
        "samples": float(samples),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def export_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Export, validate, and benchmark a checkpoint's deterministic actor."""
    config = load_config(config_path)
    sac = _build_sac(config, checkpoint_path)
    env = build_single_env(config.env_id)
    observation_dim = int(np.prod(env.observation_space.shape))
    env.close()
    output = export_actor(sac.actor, observation_dim, output_path, manifest_path)
    parity = validate_parity(sac.actor, output, observation_dim)
    if not parity["passed"]:
        raise ValueError(f"ONNX parity check failed: {parity}")
    latency = benchmark_latency(output, observation_dim)
    result = {"model": str(output), "parity": parity, "latency": latency}
    Path(output).with_suffix(".report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    """Run actor export from a config and checkpoint."""
    parser = argparse.ArgumentParser(description="Export a DynHand actor to ONNX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    result = export_checkpoint(args.config, args.checkpoint, args.out, args.manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
