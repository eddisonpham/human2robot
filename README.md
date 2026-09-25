# Human2Robot

Human2Robot (formerly DynHand) is a reproducible reinforcement learning and
ML systems project for dexterous manipulation, extended with a C++
trajectory-optimization subsystem that turns retargeted human hand motions
into constraint-aware robot demonstrations. The implemented foundation is a
configuration-driven Soft Actor-Critic trainer with demonstration-guided
learning on the standard Adroit relocate benchmark, a floating Allegro hand
environment, and a learned dynamics ensemble. The C++ subsystem validates,
projects, and optimizes retargeted joint trajectories, then hands them back
for MuJoCo replay and downstream learning experiments.

The full build specification lives in `agents/`. Read `agents/00_INDEX.md`
first; it defines the reading order. The C++ extension spec lives in
`docs/HANDOFF_RESPONSE.md`; its current status in `cpp/README.md`.

## Hardware

- NVIDIA GPU with compute capability sm_120 (RTX 5060 tested), 8 GB VRAM
- 24 CPU cores, 32 GB RAM
- Windows or Linux; everything trains locally

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned via
`.python-version`.

```bash
uv sync
```

This installs PyTorch from the CUDA 12.8 index, which is required: torch
builds older than 2.7.0 have no kernels for Blackwell GPUs.

Verify the stack:

```bash
uv run pytest tests/python/test_phase0_stack.py -v
```

Clone third-party reference repos when a phase needs them:

```bash
bash scripts/setup_references.sh
```

## Usage

Tier A is the implemented benchmark. It uses `AdroitHandRelocate-v1`
with dense reward, 39 observations, 30 actions, and a 200-step episode
horizon. The Minari dataset `D4RL/relocate/human-v2` uses the same
environment specification and contains 25 episodes and 9,942 transitions.
The exact local package versions are recorded in the lockfile and each run's
manifest.

```bash
uv run human2robot-train --config configs/tier_a_relocate.yaml --seed 0
```

Tier B is a registered floating Allegro pickup environment
(`Human2Robot-AllegroPickup-v0`, 64 observations, 22 actions) whose training and
demonstration pipeline remain in progress. Use
`configs/tier_a_relocate*.yaml` for Tier A results.

Evaluate a checkpoint:

```bash
uv run human2robot-eval --config configs/tier_a_relocate.yaml --checkpoint results/<run>/checkpoints/final.pt
```

Run names must be unique per seed. The recorder takes an operating-system
lock on each run directory and refuses a second writer. The queue scripts
use this convention automatically:

```bash
bash scripts/kill_stale_trainers.sh
bash scripts/queue_phase1.sh
bash scripts/status_tier_a.sh
```

The queue runs jobs sequentially, stops on the first failure, and records
separate output for each seed. Audit a metrics stream before plotting:

```bash
uv run python -c "from human2robot.evaluation.audit import audit_metrics; print(audit_metrics('results/<run>/metrics.jsonl'))"
```


## Project layout

```text
src/human2robot/    Library code: config, environments, RL, evaluation
cpp/                C++20 trajectory library (h2r_traj), tests, benchmarks
src/human2robot/cpp_bindings/  pybind11 module exposing the C++ optimizer
configs/            YAML run configurations, validated against pydantic schemas
tests/python/       pytest suite, coverage-gated at 90 percent
tests/cpp/          Python-side integration tests for the C++ bindings
scripts/            Setup and utility scripts
data/               Raw datasets, processed trajectories, demonstration files
results/            Experiment outputs: configs, metrics, checkpoints, plots
references/         Cloned third-party repos (gitignored)
agents/             Build specification and resume documentation
```

## Scope status

Implemented: Tier A SAC, BC initialization, Minari demonstration replay, the
floating Allegro Tier B environment, deterministic ONNX export with parity
and latency reporting, benchmark summaries, the audited experiment
infrastructure, and the C++ trajectory-optimization subsystem
(`cpp/`, 57 tests) with its Python pipeline (`human2robot.optimization`).
The pipeline optimized 100 retargeted demonstrations with 100/100
convergence, reducing max jerk 44 percent and jerk-based smoothness cost 93
percent while preserving tracking, and optimized demos replay cleanly in
MuJoCo (see `results/trajectory_optimization/report.json`).

In progress: real DexYCB retargeting (subject data downloading via
`scripts/download_dexycb.sh`) and clean multi-seed Tier A evidence.

Planned: residual physics-based dynamics, additional Tier B generalization
cases, and the final multi-condition ablation curves.

## Development

```bash
uv run pytest              # full suite with coverage
uv run ruff check .        # lint
uv run ruff format .       # format
```

## Licenses

Code is MIT. Datasets used for demonstrations carry their own terms:
DexYCB is CC BY-NC 4.0 (non-commercial). See `agents/13_REPRODUCIBILITY_AND_CONVENTIONS.md`
section 6 for the full table.
