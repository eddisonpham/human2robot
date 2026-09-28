# Architecture

## The conversion pipeline

The deliverable. Everything else consumes its output.

```
  DexYCB video (2 subjects x 100 sequences)
  hand-pose + MANO mesh per frame, 30 Hz
              |
              v
  scripts/retarget_dexycb_ik.py
  DexPilot vector retargeting: MANO keypoints -> Allegro joint targets
  22-DoF q = [6 pinned base coords | 16 finger actuators]
              |
              v
  cpp/  (h2r_traj, C++20)
  cubic resample 30 Hz -> 20 ms control period
  constrained optimizer: tracking + velocity + acceleration + jerk + limits
  projected gradient with backtracking line search, noise_scale = 0
              |
              v
  data/demonstrations_dexycb_ik_optimized/*.npz   (demo schema v2)
  q, qdot, a_demo, object/contact placeholders
              |
              +--> scripts/run_downstream_bc.py   behaviour cloning quality
              +--> scripts/score_feasibility.py   open-loop MuJoCo replay
              +--> data/processed                replay buffer for SAC
```

### Where the joints are defined

`src/human2robot/data/limits.py` is the single source of truth for the joint
layout, the actuator bounds, and both time constants. It is a MuJoCo-free
mirror of the model's `actuator_ctrlrange`, so retargeting runs without a
simulator installed. `tests/python/test_invariants.py` pins it against a live
`AllegroPickupEnv` and fails if the two ever diverge.

The model has **16 actuators, not 22**. The leading 6 coordinates are base
motion that this pipeline never fills, so they are pinned at zero by giving
them `[0, 0]` bounds.

### Where measurements come from

Every published number is produced by one script writing one JSON under
`results/trajectory_optimization/`.

| Script | Produces | Question it answers |
| --- | --- | --- |
| `compare_dexycb_synthetic.py` | `real_vs_synthetic_s*.json` | is the trajectory smoother? |
| `validate_optimizer_split.py` | `optimizer_split.json` | does the search descend on unseen data? |
| `run_downstream_bc.py` | `bc_downstream*.json` | is the trajectory easier to learn? |
| `check_bc_split_granularity.py` | `bc_split_granularity_*.json` | does the advantage survive a harder holdout? |
| `score_feasibility.py` | `feasibility_*.json` | **can the hand perform it?** |
| `diagnose_tail_extrapolation.py` | `tail_extrapolation_*.json` | why does one holdout split reverse? |

`src/human2robot/evaluation/published.py` resolves the figures quoted in
documentation out of those files, and `tests/python/test_published_results.py`
asserts that the prose matches. The committed snapshot
`docs/published_figures.json` keeps the check working where `results/` is absent.

## The RL half

A secondary consumer of the trajectories, which returned a null result.

```
  data/processed (replay buffer from demo q, qdot)
              |
              v
  human2robot-train --config configs/tier_b_*.yaml --seed <n>
  SAC, 2,000,004 steps, CPU by default
  conditions: A from-scratch | B BC-init + demo replay | C dynamics augmentation
  D and E are structurally degenerate: the nominal physics model IS the
  simulator, so the residual they would learn is identically zero
              |
              v
  results/<experiment_id>/{config.yaml, metrics.jsonl, checkpoints, git_commit}
              |
              +--> human2robot-eval        returns under domain randomization
              +--> human2robot-plot       learning curves
              +--> human2robot-benchmark  per-condition summary
              +--> human2robot-export     ONNX policy
              +--> human2robot-sb3-check  cross-check against SB3 SAC
```

`src/human2robot/evaluation/robustness.py` supplies real domain randomization:
body mass, geom friction, dof damping, and actuator gain are perturbed over
scales, and all conditions share the same initial states.

## Auxiliary services

```
  rust/inference_server/    axum + ONNX Runtime, serves /v1/act and /v1/health
                            builds, has 6 tests, never run against a model
```

## Layout

```
  src/human2robot/
    data/          schemas, retargeting, joint limits (the single source of truth)
    cpp_bindings/  pybind11 bridge
    optimization/  shared optimizer config, splits, metric aggregation
    evaluation/    executability, robustness, RL eval, published-figure lookup
    envs/          MuJoCo Allegro task
    rl/            SAC trainer, demo loading
    export/        ONNX export
  cpp/               h2r_traj C++20 library and its GoogleTest suite
  scripts/           every experiment and data entry point
  configs/           experiment configs
  tests/python/      pytest, coverage-gated at 90 percent
  docs/              results, findings, architecture
```

## Conventions

- No hyperparameters outside validated configs.
- One definition per concept: shared helpers live in `data/limits.py` and
  `optimization/experiment.py` rather than being copied into scripts.
- Every new functional module ships with unit tests in the same change.
- `data/` and `results/` are gitignored; a `.gitkeep` keeps the directories
  present so their absence is explicit rather than confusing.
