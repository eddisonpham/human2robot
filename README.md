# Human2Robot

A reinforcement learning and trajectory-optimization system for dexterous hand
manipulation. Recorded human hand motion is retargeted onto a simulated
22-DoF Allegro hand, smoothed by a C++ constrained optimizer, and used as
demonstrations for a SAC trainer.

This is primarily an **engineering** project. The ML is the application; the
substance is the pipeline, the correctness tooling, and the discipline that keeps
long experiments trustworthy. Four separate results in this repository turned out
to be artifacts of the code rather than properties of the method, each caught by
reading code or checking a control, never by watching a metric:

1. A singular rotation conversion produced observations up to 6.35e8 and critic
   loss to 1.9e15, and had produced a "demos beat from-scratch SAC by 48 percent"
   headline.
2. The hand-written physics model disagreed with its simulator by a median 12.9
   per step, in a file with 13 percent test coverage.
3. Condition C's apparent advantage was a single lucky seed.
4. On real data, a 66.5 percent imitation-error "gain" from the optimizer turned
   out to be 64.7 points of resampling and 5.2 points of optimizer.
5. The optimizer's convergence flag reported 85/100 real and 100/100 synthetic
   because it compared against the unprojected input, so projection alone
   counted as optimization. Fixing that exposed a worse problem: the search
   itself was achieving exactly zero improvement on 85 percent of sequences,
   because independent per-timestep noise is adversarial for a
   smoothness-dominated cost. Removing it took the held-out descent rate to
   47/50 with a median 18.0 percent cost reduction, and the controlled
   real-data imitation figure from 5.2 to 31 percent.

Each is written up in [`docs/`](docs/), and each has a regression test.

**What the pipeline does, on real data.** 100 DexYCB hand motion sequences are
ingested, MANO poses retargeted to Allegro joints, and the result projected onto
joint, velocity, and acceleration limits. On those sequences the optimizer cuts
jerk 42 percent and smoothness cost 65 percent, and cuts worst-case imitation
error from 0.291 to 0.065.

**The research question is adversarial on purpose:** do human demonstrations and
physics priors actually help SAC learn to control a robot hand, or is the extra
machinery wasted engineering? The honest answer so far is "mostly no". Proving
that rigorously is the point. See [`docs/RESULTS.md`](docs/RESULTS.md).

## Documentation map

| Document | What it covers |
| --- | --- |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Every measured result, with the data set each was measured on |
| [`REVIEW_STATUS.md`](REVIEW_STATUS.md) | Ablation status, critic health, open issues |
| [`docs/FINDINGS_residual_degeneracy.md`](docs/FINDINGS_residual_degeneracy.md) | Why conditions D and E cannot produce a result |
| [`docs/FINDINGS_training_hang.md`](docs/FINDINGS_training_hang.md) | The CUDA driver hang, diagnosed from a native stack dump |
| [`docs/FINDINGS_metrics_integrity.md`](docs/FINDINGS_metrics_integrity.md) | Resume corrupting the metrics stream, and the recovery |
| [`agents/`](agents/) | Read-only build specification, start at [`00_INDEX.md`](agents/00_INDEX.md) |

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

Runs default to **CPU**, which is deliberate. Force-killing a training process
leaves a CUDA context behind, and after a few of those the next process to
touch the GPU blocks forever inside `cudaStreamSynchronize`. The networks are
small enough that the GPU is not the bottleneck: a 40k-step run takes 129s on
CPU against 120s on CUDA. Set `DEVICE=cuda` to opt back in. The full
investigation is in
[`docs/FINDINGS_training_hang.md`](docs/FINDINGS_training_hang.md).

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

#### Human-to-robot trajectory conversion

Kinematic quality, measured on **100 real DexYCB sequences**:

| Quantity | Real DexYCB | Synthetic |
| --- | --- | --- |
| max jerk | -41.9% | -50.6% |
| smoothness cost | -64.7% | -93.7% |
| max velocity | -57.2% | -65.4% |
| search improved on its starting point | 47/50 held out | 93/100 |
| median cost reduction achieved | 18.0% held out | 14.8% |

Imitation quality, measured on **100 real DexYCB sequences**. The middle row is a
control that isolates the optimizer, because DexYCB captures at 30 Hz and the
environment runs at 20 ms:

| Arm | BC holdout MSE | Max error | Transitions |
| --- | --- | --- | --- |
| raw (30 Hz) | 5.89e-4 | 0.291 | 6,146 |
| resampled control (no optimizer) | 2.08e-4 | 0.180 | 10,280 |
| optimized | **1.44e-4** | **0.065** | 10,280 |

Comparing the first and last rows suggests a 75.6 percent gain. Most of that is
resampling: 64.7 percent comes from resampling alone, and the optimizer adds
**31 percent** on top. Worst-case error falls furthest, 0.291 to 0.065.

On the **synthetic** set, where both arms are already at 20 ms so the comparison
is clean, the optimizer cuts held-out error **56.8 percent** (MSE 7.08e-4 to
3.06e-4, max error 0.120 to 0.065). The two data sets disagree, and the likely
reason is input roughness, which is untested.

Caveats worth stating rather than hiding: optimized `max_velocity` is 2.0000000000000018
with a standard deviation of 4e-16, so the velocity "gain" is constraint
saturation rather than optimization headroom; and the BC metric measures how
learnable the trajectories are, not task success.

The descent rate is quoted **held out**. `scripts/validate_optimizer_split.py`
sweeps `step_size` on the first 50 sequences, selects from that half alone, then
evaluates once on the last 50, which no selection decision has seen: 47/50
improved, median 18.0 percent, with no meaningful gap against the tuning half.
An earlier version of this table was swept and reported on the same 100
sequences, which is choosing hyperparameters on the evaluation set; the 94/100
it produced is not the number quoted above. See
[`docs/RESULTS.md`](docs/RESULTS.md), `scripts/diagnose_convergence.py`, and
`results/trajectory_optimization/`.

#### The SAC ablation (null result)

Three seeds each on the Tier B pickup task, 2,000,004 steps per run:

| Condition | Seed means (eval return) | Mean +/- sd |
| --- | --- | --- |
| A from-scratch SAC | -46.0, -46.8, -54.9 | -49.2 +/- 4.9 |
| B BC-init + demo replay | -49.1, -51.6, -52.2 | -51.0 +/- 1.6 |
| C blackbox dynamics augmentation | -43.9, -54.8, -41.9 | -46.9 +/- 6.9 |

**The three conditions are indistinguishable**, and no run reaches a positive
return, so the task is unsolved. D and E are not in the table because they
cannot produce a meaningful number: the nominal physics model is the simulator
itself, so the residual they are meant to learn is identically zero. See
[`docs/FINDINGS_residual_degeneracy.md`](docs/FINDINGS_residual_degeneracy.md).

These numbers only became measurable after a fix. The observation rotation
conversion divided by `2 * sin(angle)`, which is singular at 180 degrees, and a
free-falling object hits that routinely. The affected dimensions feed the
critic, so every critic target was poisoned:

| Measurement | Before | After |
| --- | --- | --- |
| max abs observation over 60k env steps | 6.35e8 | 71-84 |
| max `qf_loss` in a 2M-step run | 1.9e15 | 1,844 (worst of 9 runs) |
| `qf_loss` spikes above 1e3 per run | hundreds to 1000+ | 0 (A, C) or 15-16 (B) |

No run in any of the three conditions records a `qf_loss` spike above 1e6.

The bug had produced a headline result of its own: "BC-init beats from-scratch
SAC by 48 percent". After the fix that gap does not reproduce. Two other
apparently empirical findings turned out to be code artifacts the same way, and
all three are written up in
[`REVIEW_STATUS.md`](REVIEW_STATUS.md).

Regenerate every result table directly from the recorded metrics, so the
numbers above cannot drift from what is on disk:

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

## Architecture

Two subsystems with a narrow interface between them. Neither knows about the
other's internals.

```text
DexYCB raw  ->  data/  ->  optimization/  ->  cpp/ (C++20)  ->  demos on disk
                                                                     |
                                                          rl/  <- demos
                                                             |
                                                        envs/ (Allegro)
                                                             |
                                                    evaluation/  -> metrics
```

**Trajectory subsystem.** `data/dexycb.py` turns DexYCB sequences into the demo
schema; `optimization/pipeline.py` calls the `cpp/` optimizer through pybind11.
The C++ side owns interpolation, smoothing, constraint projection, and the
cost function, and has its own GoogleTest suite.

**Learning subsystem.** One SAC trainer in `rl/train.py` drives every ablation
condition; conditions differ by config only, and `rl/` contains no
condition-specific branches. `envs/record.py` owns run directories, an OS-level
lock per run, the metrics stream, and checkpoint retention.

**The contract between them is the demo schema** (`data/schema.py`), which
validates array shapes on every load and write. That is the one place a
trajectory crosses from one subsystem to the other, so it is the one place
strictly validated.

## Project layout

```text
src/human2robot/
  config/        Pydantic schemas, condition flags, YAML loader
  envs/          Floating Allegro environment, vector env, run recorder
  rl/            SAC, behaviour cloning, replay, demos, the single trainer
  dynamics/      Bootstrap ensemble and the nominal physics provider
  data/          DexYCB loaders, retargeting, demo schemas
  evaluation/    Eval harness, auditing, benchmarking, plots, SB3 check
  optimization/  Trajectory optimization pipeline
  export/        Deterministic ONNX actor export
  utils/         Rotation conversion, seeding, git metadata
  cpp_bindings/  pybind11 wrapper around the C++ optimizer
cpp/             C++20 trajectory library, GoogleTest suite, examples
configs/         Run configurations, validated against the schemas
tests/python/    pytest suite, coverage-gated at 90 percent
cpp/tests/       GoogleTest suite for the trajectory library
scripts/         Setup, data, experiment runners, long-run operations
agents/          Read-only build specification
docs/            Results, investigations, and design notes
```

`data/`, `results/`, `logs/`, and `references/` are gitignored: they hold
datasets, run outputs, and cloned third-party repos, none of which belong in
version control.

## Engineering practices

**Correctness is defended by tests, not by review.** 97 percent line coverage
across 300+ Python tests, enforced at a 90 percent floor in `pyproject.toml`, plus
60 GoogleTest cases on the C++ side.

**Invariants are pinned explicitly.** `tests/python/test_invariants.py` pins the
conventions whose violations each silently corrupted results: the MuJoCo
`[w,x,y,z]` quaternion order, the substep count shared between the environment
and the physics provider, the config fields the trainer branches on, and
same-seed reproducibility down to the action sequence. Read it before changing
anything in `envs/`, `dynamics/`, or `rl/`.

**Experiments self-audit.** `envs/record.py` writes a metrics stream that
`evaluation/audit.py` will reject if it contains duplicate or out-of-order steps.
Resumes rewind the stream so this cannot happen silently.

**Every comparison gets a control.** The DexYCB result carries a resampled
pre-optimization arm precisely because the naive comparison was confounded and
said so.

**Runs are resumable without loss.** Checkpoints carry SAC weights, the replay
buffer, the trained dynamics ensemble, and RNG state, so a resumed run continues
from its checkpoint step with no rewind.

**No personal paths.** Nothing in `src/`, `scripts/`, `configs/`, `tests/`, or
`cpp/` hardcodes a machine-specific location.

## Development

```bash
uv run pytest              # full suite, coverage-gated at 90 percent
uv run ruff check .        # lint
uv run ruff format .       # format
```

The gate is: `ruff check`, `ruff format --check`, and `pytest` all pass. There is
no CI configuration, so run it locally before committing.

The C++ suite is separate and must be run by hand after a change under `cpp/`:

```bash
cmake --build cpp/build && ctest --test-dir cpp/build --output-on-failure
```

## Scope status

**Implemented.** The full conversion pipeline, end to end and scripted:
DexYCB download, real MANO ingestion and retargeting onto the Allegro hand
(`data/dexycb.py`, `python -m human2robot.data.dexycb`), a C++20 constrained
trajectory optimizer with 60 GoogleTest cases (`cpp/`) and its pybind11 bridge,
and downstream imitation evaluation with a matched control arm. On 100 real
DexYCB sequences the optimizer cuts jerk 36 percent and smoothness cost 56
percent, and more than halves worst-case imitation error.

Also Tier A SAC with BC initialization and Minari demonstration replay; the
floating Allegro Tier B environment; blackbox and residual dynamics augmentation
behind one trainer; deterministic ONNX export with parity and latency
validation; run recording with OS-level locking, metrics auditing, and lossless
checkpoint/resume; benchmarking and plotting CLIs.

**In progress.** The Tier B five-condition ablation. Conditions A, B, and C are
complete at three seeds each and none separates from the others; D and E are
blocked on domain randomization, which the specification requires to make a
residual model non-trivial and which is not yet implemented. See
[`docs/FINDINGS_residual_degeneracy.md`](docs/FINDINGS_residual_degeneracy.md).

**Not started.** The Rust inference server, the Shadow Hand stretch work, and
DexYCB subjects beyond subject-01. The ONNX exporter still uses the legacy
TorchScript path, which PyTorch 2.9 will retire in favour of `torch.export`.

**Known weaknesses.** The optimizer's effect on mean imitation error is 31
percent on real data against 57 percent on synthetic, and the reason for that gap
is untested. Only DexYCB subject-01 is processed, so the held-out half is a split
of one subject's sequences rather than a different subject. The ONNX exporter
still uses the legacy TorchScript path, which PyTorch 2.9 will retire in favour
of `torch.export`. There is no CI, so the gate and the C++ suite are run by hand.

## Licenses

Code is MIT. Datasets carry their own terms: DexYCB is CC BY-NC 4.0
(non-commercial). See
[`agents/13_REPRODUCIBILITY_AND_CONVENTIONS.md`](agents/13_REPRODUCIBILITY_AND_CONVENTIONS.md)
section 6 for the full table.
