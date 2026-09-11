# DynHand

Demonstration-guided reinforcement learning for dexterous manipulation.
Human hand demonstrations are retargeted onto a floating Allegro hand in
MuJoCo and used to accelerate a Soft Actor-Critic trainer. The project
answers one question: do human demonstrations plus physics-structured
learned dynamics improve sample efficiency, stability, and generalization
over RL from scratch?

The full build specification lives in `agents/`. Read `agents/00_INDEX.md`
first; it defines the reading order.

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

Tier A validates the trainer on the standard Adroit hand relocate task
with human demonstrations from Minari:

```bash
uv run dynhand-train --config configs/tier_a_relocate.yaml --seed 0
```

Tier B trains on the custom floating Allegro pickup environment:

```bash
uv run dynhand-train --config configs/tier_b_pickup.yaml --seed 0
```

Evaluate a checkpoint:

```bash
uv run dynhand-eval --config configs/tier_a_relocate.yaml --checkpoint results/<run>/checkpoints/final.pt
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
uv run python -c "from dynhand.evaluation.audit import audit_metrics; print(audit_metrics('results/<run>/metrics.jsonl'))"
```


## Project layout

```text
src/dynhand/        Library code: config, environments, RL, dynamics, data, export
configs/            YAML run configurations, validated against pydantic schemas
tests/python/       pytest suite, coverage-gated at 90 percent
scripts/            Setup and utility scripts
data/               Raw datasets, processed trajectories, demonstration files
results/            Experiment outputs: configs, metrics, checkpoints, plots
references/         Cloned third-party repos (gitignored)
agents/             Build specification, read-only
```

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
