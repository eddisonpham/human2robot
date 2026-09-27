# Finding: the training hang is a CUDA driver hang, now root-caused

Date: 2026-09-27. Seventh occurrence, and the first one captured with a live
stack dump.

## Symptom

A run stops writing metrics while staying alive and burning CPU at exactly one
core-second per wall second. One thread reads `Running` with tens of minutes of
accumulated CPU while every other thread waits. Resuming from the latest
checkpoint always works, then the same thing happens again later.

## Root cause

Sampling a hung process with `py-spy` shows a Python stack that looks
innocuous, pointing at distribution argument validation:

```
__init__ (torch\distributions\distribution.py:76)
__init__ (torch\distributions\normal.py:66)
forward (human2robot\rl\networks.py:48)      # Normal(mean, std) in the actor
act (human2robot\rl\sac.py:60)
train (human2robot\rl\train.py:252)
```

The native stack is where the truth is:

```
cuStreamSynchronize (nvcuda64.dll)
c10::cuda::memcpy_and_sync (c10_cuda.dll)
at::native::searchsorted_out_cuda
at::native::_local_scalar_dense_cuda
at::native::item
at::native::is_nonzero
```

PyTorch validates a `Normal` by calling `is_nonzero(...)` and reducing it with
`.item()`. On CUDA, `.item()` is a synchronizing call. The process is not slow,
it is **blocked forever waiting on a GPU stream that never completes**.

The Python frames are a red herring: the CPU simply parks at whatever call
happened to issue the sync.

## Why the GPU gets into that state

`nvidia-smi --query-compute-apps` listed seven PIDs attached to the GPU during
the investigation, of which only one belonged to a live process. Every
force-killed training run during this project left a CUDA context behind, and
the accumulated contexts eventually put the driver into a state where a new
process's first synchronizing call never returns.

The trigger is therefore not the algorithm, the environment, or the model. It is
**repeatedly killing training processes on a shared GPU**, which is exactly what
happened while managing the ablation matrix.

## Fix

`--device` on the trainer, defaulting to `auto` but selectable as `cpu`:

```bash
uv run human2robot-train --config <cfg> --seed 0 --device cpu
```

`scripts/resume_run.sh` now passes `--device cpu` by default, overridable with
`DEVICE=cuda`.

Cost measured on a 40,000-step Condition C run with 12 environments:

| Device | Wall clock |
| --- | --- |
| CPU | 129 s |
| CUDA | 120 s |

Seven and a half percent, or roughly 8 minutes on a 2,000,004-step run, against
losing 25 to 45 minutes and a checkpoint rewind every time the driver wedges.
CPU is the right default for this workload: the actor and critics are small
256-wide MLPs and the run is dominated by MuJoCo stepping, which is CPU work
either way.

## What was ruled out before this

Each was measured rather than assumed, and each was a reasonable guess:

- **MuJoCo degradation**: 3000 steps, worst single step 3 ms, observations
  finite. Not degrading.
- **Evaluation looping**: every logged evaluation completed its full 500
  env steps; the environment truncates at `max_episode_steps`.
- **Ensemble retraining**: one full retrain at the capped 10,000-sample buffer
  is about 3 s. Cannot produce a 24-minute stall.
- **Unbounded loops**: every `while` and `for` in `train.py` and `ensemble.py`
  is bounded by a config value or a range.
- **Deadlock**: a blocked thread burns no CPU, and this burns exactly one core
  continuously.

## How to catch it in future

The watchdog recovers from the hang but cannot diagnose it. A cheap upgrade
would be for the watchdog to dump a stack when metrics go quiet, using
`uv run --with py-spy py-spy dump --pid <pid>`, which installs transiently and
does not touch project dependencies. That is what turned this from a mystery
into a one-line fix.
