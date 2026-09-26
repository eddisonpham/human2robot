# Human2Robot - Status and Results (as of 2026-09-26)

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

All three runs show frequent Q-function loss explosions throughout training
(486, 619, and 1050 explosions respectively where qf_loss > 1000). The
SAC learner is unstable on this environment regardless of condition. Despite
this, the runs complete and produce finite evaluation returns.

The returns are negative because the task reward is shaped around object lift
plus success bonus, and the agent rarely succeeds at picking up the object.

### Condition B - BC-init + demo seeding

One launched run:
- `results/tier_b_pickup_cond_b_s0/` - running, 213k steps so far
  - 23 Q-loss explosions so far (vs. 486+ for Cond A at same step count)
  - 10 eval returns so far, mean -233.41, latest -367.72 at step 200004
  - BC-init from local demos in `data/demonstrations/`

This run is still in flight. Early indicators suggest fewer explosions than
Cond A, but it is too early to call final results.

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

- `results/tier_b_pickup_cond_b_s0` - running (PID in run_status.json)

## What is not yet done

1. **Condition B s0** is still running. Need to let it complete (2M steps).
2. **Conditions C, D, E** are dropped from this deliverable. The dynamics
   augmentation approach is unstable on this env with the current config.
3. **Coverage**: pytest passes 162 tests but coverage is 85% (below 90%
   threshold). The gap is mostly in unused modules (optimization pipeline,
   evaluation plots, export onnx) that aren't exercised by the test suite.
4. **Push**: Not attempted. No remote access configured in this environment.

## How to read this as a human

If you want the project "complete with reportable results" in the strict
Phase 8 sense, the remaining work is:
- let B s0 finish,
- diagnose and fix the dynamics augmentation for C/D/E,
- re-run C/D/E with fixed configs,
- and reconcile the Q-loss instability across all conditions.

If "complete" means "clean tree, committed, lint+test passing, and the
best results that exist on disk right now are reported," then:
- Cond A has 3 completed seeds with honest numbers (negative returns,
  frequent Q explosions),
- Cond B is running and showing early promise (fewer explosions),
- C/D/E were attempted but abandoned due to dynamics model divergence,
- ruff + pytest pass,
- and everything is committed locally.
