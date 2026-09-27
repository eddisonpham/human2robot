# Human2Robot

Demonstration-guided reinforcement learning for dexterous hand manipulation,
extended with a C++ trajectory-optimization subsystem that turns retargeted
human hand motions into constraint-aware robot demonstrations.

The research question this repository exists to answer is deliberately
adversarial: **do human demonstrations and physics priors actually help SAC
learn to control a robot hand, or is the extra machinery wasted engineering?**
The honest answer so far is "mostly no", and proving that rigorously is most of
the work. See [`REVIEW_STATUS.md`](REVIEW_STATUS.md) for current results.

- Build specification: [`agents/`](agents/) — start at
  [`agents/00_INDEX.md`](agents/00_INDEX.md)
- Experiment results and open issues: [`REVIEW_STATUS.md`](REVIEW_STATUS.md)
- Investigations: [`docs/`](docs/)

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.11 (pinned by `.python-version`)
- NVIDIA GPU with compute capability sm_120 (RTX 5060 tested), 8 GB VRAM
- 24 CPU cores, 32 GB RAM
- Windows or Linux; everything trains locally

## Setup

```bash
uv sync
```

This installs PyTorch from the CUDA 12.8 index, which is **required**: the
Blackwell GPU in use has no kernels in torch builds older than 2.7.0.

Verify the Python stack:

```bash
uv run pytest tests/python/test_phase0_stack.py -v
```

The C++ subsystem is a separate build. It is optional; only the trajectory
optimization pipeline needs it. See [`cpp/README.md`](cpp/README.md) for the
toolchain and build commands, then confirm the bindings import:

```bash
uv run python scripts/check_bindings.py
```

Third-party reference repositories, cloned into `references/` when a phase
needs them:

```bash
bash scripts/setup_references.sh
```

## Usage

### Training

Every ablation condition is the same trainer driven by a different config, so
there is one command:

```bash
uv run human2robot-train --config configs/tier_b_pickup.yaml --seed 0
```

| Config | Condition | What it varies |
| --- | --- | --- |
| `tier_b_pickup.yaml` | A | from-scratch SAC |
| `tier_b_cond_b.yaml` | B | BC-init + demo-seeded replay |
| `tier_b_cond_c.yaml` | C | + blackbox dynamics augmentation |
| `tier_b_cond_d.yaml` | D | + residual (physics-prior) augmentation |
| `tier_b_cond_e.yaml` | E | demos and residual augmentation together |
| `tier_a_relocate.yaml` | Tier A | `AdroitHandRelocate-v1` correctness gate |
| `smoke.yaml` | - | 400-step end-to-end check |

`--run-name` sets the output directory under `results/`. Run names must be
unique: the recorder takes an OS lock on each run directory and refuses a
second writer.

### Running long jobs

The Tier B matrix is 2,000,004 steps per run and takes a few hours each. Three
scripts cover the operational loop.

```bash
bash scripts/resume_run.sh tier_b_pickup_cond_c_s1   # start or continue a run
bash scripts/pause_run.sh  tier_b_pickup_cond_c_s1   # stop at a clean checkpoint
bash scripts/watchdog_runs.sh 45 tier_b_pickup_cond_c_s1   # auto-restart on stall
```

**Resume** restores SAC weights, the replay buffer, the dynamics ensemble, and
RNG state, so a run continues from its checkpoint step rather than restarting.
`resume_run.sh` infers the config and seed from the run name and prints the
checkpoint it will resume from.

**Pause** writes `results/<run>/PAUSE`, which the training loop polls, so the
run stops at a clean checkpoint and records a `paused` status. Windows cannot
deliver `SIGTERM` to a detached process, so this sentinel file is the only
pause channel; it costs zero steps. `pause_run.sh --now` skips the graceful
path and kills immediately, which rewinds to the last periodic checkpoint.

**Watchdog** restarts a run from its latest checkpoint if it stops writing
metrics for the given number of minutes. A single-instance PID lock prevents
two watchdogs from fighting and from resurrecting a deliberately cancelled run.

### Results

Regenerate every result table directly from the recorded metrics, so the
numbers in `REVIEW_STATUS.md` cannot drift from what is on disk:

```bash
uv run python scripts/summarize_ablation.py --markdown
```

Plot learning curves and benchmark several conditions:

```bash
uv run human2robot-plot --runs "SAC=results/run_a, SAC+demo=results/run_b" \
  --metric eval_return_mean --out results/plots/curves.png
uv run human2robot-benchmark --group "A=results/run_a,results/run_b" \
  --out results/analysis/summary.json
```

### Other entry points

```bash
uv run human2robot-eval --config <cfg> --checkpoint <ckpt>   # evaluate a checkpoint
uv run human2robot-export --config <cfg> --checkpoint <ckpt> --out policy.onnx
uv run human2robot-sb3-check --config <cfg>                 # cross-check vs SB3 SAC
```

ONNX export validates parity against PyTorch and reports inference latency
before writing a manifest. The SB3 cross-check is an independent implementation
used to validate the environment and matched hyperparameters; it is a debugging
tool, not the primary training path, and needs `uv sync --group sb3`.

## Project layout

```text
src/human2robot/
  config/        Pydantic schemas, condition flags, YAML loader
  envs/          Floating Allegro environment, vector env, run recorder
  rl/            SAC, behaviour cloning, replay, demos, the single trainer
  dynamics/      Bootstrap ensemble and the nominal physics provider
  data/          DexYCB loaders, retargeting, demo schemas
  evaluation/    Eval harness, benchmarking, plots, robustness, SB3 check
  optimization/  Trajectory optimization pipeline
  export/        Deterministic ONNX actor export
  utils/         Rotation conversion, seeding, git metadata
  cpp_bindings/  pybind11 wrapper around the C++ optimizer
cpp/             C++20 trajectory library (h2r), GoogleTest suites, examples
configs/         Run configurations, validated against the schemas
tests/python/    pytest suite, coverage-gated at 90 percent
scripts/         Setup, data, and long-run operations
agents/          Read-only build specification
docs/            Investigations and design notes
```

`data/`, `results/`, `logs/`, and `references/` are gitignored: they hold
datasets, run outputs, and cloned third-party repos, none of which belong in
version control.

## Development

```bash
uv run pytest              # full suite, coverage-gated at 90 percent
uv run ruff check .        # lint
uv run ruff format .       # format
```

The gate is: `ruff check`, `ruff format --check`, and `pytest` all pass. There
is no CI configuration, so run it locally before committing.

`tests/python/test_invariants.py` deserves attention before changing anything in
`envs/`, `dynamics/`, or `rl/`. It pins the conventions whose violations have
each silently corrupted results: the MuJoCo `[w,x,y,z]` quaternion order, the
substep count shared between the environment and the physics provider, the
config fields the trainer branches on, and same-seed reproducibility down to
the action sequence.

## Scope status

**Implemented.** Tier A SAC with BC initialization and Minari demonstration
replay; the floating Allegro Tier B environment; blackbox and residual dynamics
augmentation behind one trainer; deterministic ONNX export with parity and
latency validation; benchmark and audit tooling; the C++ trajectory subsystem
(`cpp/`, 57 GoogleTest cases) with its Python pipeline.

**In progress.** The Tier B five-condition ablation. Conditions A, B, and C have
results; D and E are blocked on domain randomization, which the specification
requires to make a residual model non-trivial and which is not yet implemented.
See [`docs/FINDINGS_residual_degeneracy.md`](docs/FINDINGS_residual_degeneracy.md).

**Not started.** Real DexYCB retargeting end to end, the Rust inference server,
and the Shadow Hand stretch work.

## Licenses

Code is MIT. Datasets carry their own terms: DexYCB is CC BY-NC 4.0
(non-commercial). See
[`agents/13_REPRODUCIBILITY_AND_CONVENTIONS.md`](agents/13_REPRODUCIBILITY_AND_CONVENTIONS.md)
section 6 for the full table.
