# Rust inference server

An axum HTTP service that runs the exported ONNX policy and returns actions.
Added to version control in the September 28 pass; before that it existed only
on one machine, inside an accidental `git init` with zero commits, so it had no
history at all. See `rust/inference_server/src/main.rs`.

## State

**It builds, but has never been run against a real model.** There is no
benchmark and no recorded latency, and the only test covers input validation
rather than inference. Treat it as a scaffold rather than a working service.

It did not compile when this directory was first added to version control. The
code was written against a different `ort` API than the pinned
`2.0.0-rc.13` provides: it imported `ort::SessionBuilder`, `ort::input`, and
`ort::GraphOptimizationLevel`, none of which are public at those paths, and
called `with_intra_op_num_threads`, which in this version belongs to the XNNPACK
execution provider rather than to the session builder. It now uses
`Session::builder()`, `session::builder::GraphOptimizationLevel`,
`with_intra_threads`, and `Tensor::from_array`. The HTTP contract is unchanged.

That is worth recording rather than quietly fixing: the code had sat outside
version control, so nothing had ever compiled it. The ONNX exporter it consumes
still uses the legacy `torch.onnx.export` API, deprecated in torch 2.9, so the
input and output names in the manifest still need to be checked against an
actual export.

## Endpoints

| Route | Method | Body | Returns |
| --- | --- | --- | --- |
| `/v1/health` | GET | none | `model_version`, `state_dim`, `action_dim` |
| `/v1/act` | POST | `{"state": [f32, ...]}` | `{"action": [f32, ...], "latency_us": u64}` |

`/v1/act` returns HTTP 400 with an empty action when `state` does not have
exactly `state_dim` entries, and HTTP 500 when the model returns a number of
actions that disagrees with `action_dim`. The second check is there because a
manifest that lies about its dimensions would otherwise hand the caller a
wrong-length action vector to send to the robot.

## Configuration

All three are read from the environment, with defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | `policy.onnx` | The exported ONNX graph |
| `MANIFEST_PATH` | `policy.json` | Input/output tensor names and dimensions |
| `MODEL_VERSION` | `unknown` | Reported by `/v1/health` |
| `LISTEN_ADDR` | `0.0.0.0:8080` | Bind address |

The manifest is JSON:

```json
{
  "input_name": "observations",
  "output_name": "actions",
  "state_dim": 42,
  "action_dim": 22
}
```

`state_dim` and `action_dim` are not free parameters. They have to match the
observation and action spaces of the environment the policy was trained on, and
`action_dim` in particular is **22 in the demo schema but 22 including 6 pinned
base coordinates that are not actuators**. The real actuator count is 16. See
`src/human2robot/data/limits.py`, which is the canonical statement and is
pinned against the MuJoCo model by a test.

## Building

`cargo` is not on `PATH` by default on this machine; it lives in
`%USERPROFILE%/.cargo/bin`.

```bash
cargo build --release --manifest-path rust/inference_server/Cargo.toml
```

The build tree is ignored. `Cargo.lock` is committed, because a server binary
should be reproducible from its lockfile.

## Tests

```bash
cargo test --manifest-path rust/inference_server/Cargo.toml
```

`cargo test` is part of the gate for this directory. The suite covers manifest
loading and request validation, which is everything testable without an ONNX
model on disk. Inference itself is not covered, because covering it needs a
model file and a runtime, and neither is checked in.

## Before this can be called done

1. Run it against a real export and confirm the tensor names in the manifest
   match, since nothing has ever loaded a model here.
2. Add an inference test against a small checked-in model, so a change to the
   input plumbing is caught rather than discovered at deployment.
3. Record a latency figure. A service whose purpose is inference speed and
   which has never been timed is not evaluated.
4. Update the ONNX exporter off the deprecated `torch.onnx.export` path.
