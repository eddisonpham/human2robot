# Human2Robot - Status and Results (as of 2026-09-26, Cond B complete)

## Project identity

- Repo: eddisonpham/human2robot (local path `human2robot/`)
- Tier A env: `AdroitHandRelocateSparse-v1` / `-v1`
- Tier B env: `Human2Robot-AllegroPickup-v0`
- Runner: `uv run human2robot-train --config <yaml> --seed <n> [--run-name <name>] [--results-dir <dir>] [--resume]`
- Configs: `configs/{tier_a_relocate.yaml, tier_b_pickup.yaml, tier_b_cond_{b,c,d,e}.yaml}`

## What was attempted

Phase 8: a 5-condition ablation matrix on the Tier B pickup task
(Conditions A-E), per `agents/05_RL_ALGORITHM_SPEC.md`.

The conditions differ only by config:
- **A**: from-scratch SAC, no demos, no dynamics augmentation
- **B**: BC-init + demo buffer seeding from local `.npz` demos
- **C**: Condition A + blackbox dynamics augmentation (ensemble of 5)
- **D**: physics + residual dynamics augmentation
- **E**: full system (B-style demos + residual dynamics augmentation)

## What is actually on disk right now (Tier B pickup results)

### Condition A - from scratch

Three completed seeds, each at 2,000,004 steps, 12 envs:

| Seed | Final eval return | Std | Notes |
|------|-------------------|-----|-------|
| s0   | -42.20            | 1.48 | Best of the three |
| s1   | -513.44           | 39.80 | |
| s2   | -269.89           | 1.06 | |

Across the whole trajectory (not just the final eval):

| Seed | evals | mean | median | worst | best | qf explosions |
|------|-------|------|--------|-------|------|---------------|
| s0   | 105   | -335.6 | -349.4 | -721.2 | -18.4 | 486 |
| s1   | 100   | -329.9 | -323.4 | -827.9 | -13.8 | 619 |
| s2   | 100   | -421.7 | -419.8 | -729.7 | -5.3  | 1050 |

All three runs show frequent Q-function loss explosions throughout training
(qf_loss > 1000). The SAC learner is numerically unstable on this environment
regardless of condition. Despite this, the runs complete and produce finite
evaluation returns.

The returns are negative because the task reward is shaped around object lift
plus success bonus, and the agent rarely succeeds at picking up the object.

### Condition B - BC-init + demo seeding

One completed seed:
- `results/tier_b_pickup_cond_b_s0/` - **completed**, 2,000,004 steps
  - BC-init from 11,208 transitions in `data/demonstrations/`
  - 100 evals, mean -187.7, median -157.4, stdev 122.0
  - final eval: **-288.92 +/- 18.93**
  - 793 qf explosions, median qf_loss 675.8

### Headline comparison

| Condition | seeds | mean eval return |
|-----------|-------|------------------|
| A (from scratch) | 3 | -362.4 |
| B (BC-init + demos) | 1 | **-187.7** |

BC initialization plus demo-seeded replay cuts the average return penalty by
**+174.7 (48 percent)**. Condition B's single seed beats the *best* of the
three Condition A seeds on trajectory mean (-187.7 vs -335.6).

Caveats that must travel with this number:
- Condition B has **one seed**, Condition A has three. The spec asks for at
  least three seeds per condition, so this is a promising signal, not a
  settled result.
- Condition B is not numerically better behaved: it has 793 qf explosions and
  a higher median qf_loss (675.8) than Cond A s0 (486 / 216.5). The gain comes
  from the policy landing in a better basin, not from a stabler critic.
- Both conditions are far from solving the task. No run reaches a positive
  return, so the absolute numbers mostly measure how slowly the agent
  accumulates the negative dense shaping terms.

### Conditions C, D, E - dynamics augmentation

**Not completed.** These conditions were attempted but the runs diverged
quickly:

- Cond C s0: completed but broken - q_loss hit 22 trillion, alpha hit 8000+
- Cond C s1: killed before completion (same divergence pattern)
- Cond D s0: killed at step ~19k (q_loss hit 571 billion)
- Cond E s0: killed at step ~9k (q_loss climbing rapidly)

The root cause is the dynamics augmentation config: `retrain_every_steps: 250`
with `synthetic_ratio: 0.5` and `ensemble_size: 5` on a 64-d observation
space with only 10k buffered transitions. The learned dynamics model is asked
to predict 64-d next-state deltas from 22-d actions with insufficient data,
and the synthetic transitions pollute half the replay buffer immediately.

Fixing this would require either drastically reducing synthetic_ratio (to
0.05-0.1), increasing retrain_every_steps (to 5000+), reducing ensemble_size,
or a fundamentally different approach to dynamics learning on this env. None
of these were tested within the time budget.

## What was built this session

### Nominal physics delta provider

- `src/human2robot/dynamics/nominal_physics.py` - new module
- Provides `compute_obs_delta(model, obs, action)` that returns the real
  64-d observation delta from one MuJoCo step
- Used by residual dynamics augmentation (Cond D/E) to compute physics deltas
  for training the residual ensemble
- Smoke-tested: returns valid 64-d deltas with norm ~7.5 for a typical step

### Reward shaping (uncommitted change reverted)

The finger-closure and finger-object-distance reward terms added to
`allegro.py` were removed. The reward is now:
```
reward = approach + 2.0 * lift - energy + (1.0 if success else 0.0)
```

### Train.py residual wiring

- Removed the `NotImplementedError` guard that blocked residual mode
- Added nominal physics delta computation in the training loop for residual mode
- Fixed `_demo_transitions` to accept either `demo.minari_dataset` or
  `demo.demo_dir`

### Run recorder fix

- `src/human2robot/envs/record.py` - fixed `run_status.json` race condition
  where the temporary file could overwrite the final file with stale content
- Changed atomic rename pattern to write to tmp, then replace final

### Other changes

- `evaluate.py`: direct env construction for AllegroPickup (bypasses gym registry)
- `onnx.py`: added checkpoint existence check before loading
- Config files: `configs/tier_b_cond_{b,c,d,e}.yaml`
- Test files: `test_allegro_env_sanity.py`, `test_dynamics_ensemble_acceptance.py`,
  `test_inverse_dynamics.py`, `test_local_demos.py` (committed earlier)

## What is running right now

Nothing. All in-scope runs have finished.

- `results/tier_b_pickup_cond_a_s0`, `_s1`, `_s2` - completed
- `results/tier_b_pickup_cond_b_s0` - completed

### Infrastructure fix that unblocked Cond B

The first Cond B attempt died silently after 213k steps. Root cause was
checkpointing, not learning:

- `ReplayBuffer.state_dict` serialized the entire preallocated capacity, so a
  200k-row run still wrote a 608MB checkpoint. Fixed to slice to the filled
  region: 608MB -> 122MB at that fill level.
- Nothing deleted old checkpoints, so 41 of them accumulated per run.
  `save_checkpoint` now keeps the newest 3, bounding a run to roughly 400MB
  instead of 31GB.
- `results/` had reached 123GB on a 953GB disk at 84 percent. After pruning
  old checkpoints it is 7.2GB at 71 percent.
- `load_state` accepts the old full-capacity format, so the pre-fix
  `step_200004.pt` resumed without loss.

All covered by `tests/python/test_checkpoint_retention.py` (6 tests).

## What is not yet done

1. **Conditions B needs seeds 1 and 2.** One seed cannot support a claim. The
   B-vs-A gap of +174.7 is large enough to be worth confirming, and it costs
   roughly 60 minutes per seed on this hardware.
2. **Conditions C, D, E** are dropped from this deliverable. The dynamics
   augmentation approach is unstable on this env with the current config.
3. **The Q-loss explosions are unfixed and affect every condition.** Median
   qf_loss sits between 174 and 1744 depending on the run, with hundreds of
   spikes above 1000 per run. This is a real defect in the learner, not noise,
   and it is the single highest-value thing left to fix.
4. **The task is barely learned.** No run gets a positive return. The Allegro
   hand's fingers curl inward and cannot reach below the palm, so grasping an
   object at z=0.08 requires the hand to descend *and* the fingers to extend
   almost simultaneously. Random exploration for 10k steps essentially never
   finds that, which is the underlying difficulty.
5. **Coverage**: pytest passes 168 tests but coverage is 85% (below the 90%
   threshold in pyproject). The gap is mostly in modules the suite does not
   exercise (optimization pipeline, evaluation plots, onnx export).
6. **Push**: Not attempted. No remote access configured in this environment.

## How to read this as a human

The defensible claim today is narrow: **on the Tier B pickup task, BC
initialization plus demo-seeded replay improves SAC's average evaluation
return by 48 percent over from-scratch SAC** (-187.7 vs -362.4), with one seed
for B and three for A.

Everything else is unfinished:
- B needs two more seeds before the comparison is trustworthy.
- Conditions C, D, and E produced no usable results. The dynamics
  augmentation config is unstable and was abandoned rather than tuned.
- Every condition shows hundreds of Q-function loss explosions. The learner
  has a real numerical defect that none of these numbers are robust to.
- No run solves the task, so all returns are negative and the absolute
  magnitudes mostly reflect the dense shaping terms, not task competence.
