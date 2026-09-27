# Finding: the training hang is a single-threaded busy spin

Date: 2026-09-27. Fifth occurrence, and the first one caught in the act.

## Signature

Observed on Condition C seed 1 at step 542,004, with the process alive and
metrics unwritten for 24 minutes:

- Process CPU advanced at exactly 1.0 core-seconds per wall second, sampled
  three times at 10s intervals: 2576.1 -> 2586.3 -> 2596.3.
- Thread-level inspection showed one thread `Running` with 38 minutes of
  accumulated CPU, and every other thread `Wait` with under 5 minutes.
- Memory was not a factor; the box had 12GB free and, by then, 20 idle cores.

So this is not a deadlock (a blocked thread burns no CPU) and not a parallel
workload (that would show several running threads). It is one thread spinning
without making progress.

## What was ruled out

Measured directly rather than assumed:

- **MuJoCo stepping.** 3000 steps driven by a deterministic policy: 3.5s total,
  worst single step 3ms, observations finite with `absmax` 0.22. The simulator
  is not degrading.
- **Evaluation.** Every logged eval ran the full 500 env steps and completed.
  The env truncates at `max_episode_steps=500`, so the `while not done` loop in
  `evaluate` is bounded.
- **Ensemble retraining.** One full retrain at the capped 10,000-sample buffer
  with 5 members, 2 epochs, batch 256 takes ~3s. Even at the
  `retrain_every_steps=2500` cadence that cannot produce a 24-minute stall.
- **Unbounded loops.** Every `while` and `for` in `train.py` and
  `ensemble.py` is bounded by a config value or a range.

## The remaining suspect

The spin is single-threaded Python or native code that does not terminate
promptly, and the training loop's own structure does not contain it. Resolving
this needs a Python-level stack sample from the live process, which means
`py-spy` (`py-spy dump --pid <pid>`). It is not currently a project
dependency, so it was not installed during this session.

Suggested next step: install `py-spy` as a dev dependency, then run a
Condition A or C run with a sampling watchdog that dumps the stack every time
metrics go quiet for more than 10 minutes. That converts the next occurrence
from "recovered by resume" into a diagnosis.

## Mitigation in place

`scripts/watchdog_runs.sh` restarts a run from its latest checkpoint after 45
minutes without a metric write. Checkpoint trimming and the dual-format loader
mean every resume so far has succeeded, at the cost of rewinding to the last
checkpoint (Condition C seed 1 rewound 542,004 -> 510,000, about 30k steps, and
recovered past the previous stall within two minutes).

Related: [[FINDINGS_residual_degeneracy]].
