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

### Running experiments

Training runs execute sequentially (concurrent processes contend for CPU
and drop aggregate throughput roughly fivefold). Launch the Tier A queue:

```bash
bash scripts/launch_tier_a.sh
```

Check progress at any time:

```bash
bash scripts/status_tier_a.sh
```

Runs resume from their latest checkpoint if interrupted: just relaunch
with the same config. Compare learning curves across runs:

```bash
uv run dynhand-plot --runs "SAC=results/tier_a_relocate_cond_a, SAC+demo=results/tier_a_relocate_cond_b" --out results/plots/tier_a.png
```

Cross-check the environment with Stable-Baselines3's SAC (requires
`uv sync --group sb3`):

```bash
uv run dynhand-sb3-check --config configs/tier_a_relocate.yaml
```

The five ablation conditions (A plain RL, B demo-guided, C black-box
dynamics, D physics plus residual dynamics, E full) are selected through
the config files in `configs/`. All conditions run the same trainer.

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
