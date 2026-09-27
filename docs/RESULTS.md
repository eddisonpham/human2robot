# Results

Two parts: the human-to-robot conversion pipeline, which is the project's actual
deliverable, and the SAC ablation, which is a null result. Every number here was
read from an artifact on disk, and each is labeled with the data set it was
measured on. The distinction matters, because the two halves of the conversion
result have not been joined yet.

## 1. Human-to-robot trajectory conversion

### What runs

Real DexYCB hand motion sequences are ingested, MANO poses are retargeted to
22-DoF Allegro joint targets, and a C++20 constrained optimizer projects the
result onto joint, velocity, and acceleration limits.

- `src/human2robot/data/dexycb.py`: sequence discovery, MANO pose to joint
  targets, per-subject demo building.
- `cpp/` (library `h2r_traj`): Hermite interpolation, finite differences,
  moving-window smoothing, projection onto joint/velocity/acceleration limits,
  weighted cost, seeded stochastic optimizer with projection. **57 GoogleTest
  cases, verified 57/57 passing in 1.45s.**
- `src/human2robot/cpp_bindings/`: pybind11 bridge.

### 1a. Kinematic quality, on 100 real DexYCB sequences

From `results/trajectory_optimization/real_vs_synthetic.json`. DexYCB MANO
capture runs at 30 Hz; trajectories are resampled to the 20 ms Allegro control
period so both data sets are optimized under identical settings.

| Quantity | Real DexYCB | Synthetic |
| --- | --- | --- |
| sequences converged | 85 / 100 | 100 / 100 |
| max jerk | -35.5% | -44.2% |
| smoothness cost | -56.3% | -93.0% |
| max velocity | -57.2% | -65.4% |
| max acceleration | -13.0% | -16.8% |

**Read the velocity and acceleration rows with suspicion.** Optimized
`max_velocity` is 2.0000000000000018 with a standard deviation of 4.4e-16 on
both data sets, and `max_acceleration` lands on 200.00000000000017. The limits
are binding and the optimizer is clipping to them, so those "reductions" measure
constraint enforcement rather than optimization headroom. Jerk and smoothness are
the unbounded quantities, so they are the honest wins here.

**15 of 100 real sequences do not converge.** The synthetic set converges
100/100, so something about real retargeted trajectories produces cases the
optimizer cannot satisfy. That is unexplained and worth understanding before
trusting the pipeline on new data.

### 1b. Imitation quality, on the 100 synthetic sequences

From `results/trajectory_optimization/bc_downstream.json`. Identical BC heads
and hyperparameters across arms; only the demonstration source differs.

| Demo set | BC holdout MSE | MAE | Max error | Transitions |
| --- | --- | --- | --- | --- |
| raw retargeted | 7.08e-4 | 0.0182 | 0.120 | 10,088 |
| optimized | **3.43e-4** | **0.0122** | **0.069** | 10,088 |
| mixed | 5.18e-4 | 0.0151 | 0.114 | 20,175 |

Optimizing cut held-out imitation error **52 percent**, with the reported error
matching the early-stopped holdout best, so it is not an overfit figure.
Optimized demonstrations are easier to imitate, which is the useful property:
smooth, limit-respecting motion is what a behavior-cloning or RL agent can fit.

This measures learnability, not task success. Nothing here shows a robot
performing better.

### 1c. The gap between the two halves

**The imitation comparison has never been run on the real DexYCB data.** The two
measurements above sit on different data sets, and the obvious experiment, raw
versus optimized on the real sequences, is missing.

The reason is mechanical: `scripts/compare_dexycb_synthetic.py` optimizes the
real trajectories in memory and writes only aggregate statistics, discarding the
optimized output. `scripts/run_downstream_bc.py` needs both variants on disk.

### 1d. Reproducing this, and where the chain breaks

| Step | Command | Produces |
| --- | --- | --- |
| 1 | `bash scripts/download_dexycb.sh` | `data/raw/dexycb/` |
| 2 | `bash scripts/wsl_setup_retargeting.sh` | WSL venv with `dex-retargeting` |
| 3 | `uv run python scripts/retarget_dexycb_ik.py [link_length] [subject] [dir]` | `data/demonstrations_dexycb_ik/` (DexPilot IK) |
| 4 | **no committed command** | `data/demonstrations_dexycb/` (vector retargeting, 100 seq) |
| 5 | `uv run python scripts/compare_dexycb_synthetic.py` | `real_vs_synthetic.json` |
| 6 | `uv run python scripts/run_trajectory_experiment.py` | `report.json` (synthetic) |
| 7 | `uv run python scripts/run_downstream_bc.py` | `bc_downstream.json` (synthetic) |

Step 4 is the break. `data/demonstrations_dexycb/` exists on disk with 100
sequences, but the function that builds it, `dexycb.build_subject_demos`, has
no CLI and no production caller; only `tests/python/test_dexycb_pipeline.py`
invokes it. `compare_dexycb_synthetic.py` responds to a missing directory with
"run build_subject_demos first" and no command to run. So the primary
deliverable's input set is not regenerable from the committed code.

Note also that the optimized demonstrations on disk in
`data/demonstrations_optimized/` are the **synthetic** set, which is why 1b is
synthetic. The real optimized trajectories were never written.

## 2. SAC ablation: a null result

Three conditions, three seeds each, 2,000,004 steps per run, on the floating
Allegro pickup task. Full analysis in [`../REVIEW_STATUS.md`](../REVIEW_STATUS.md).

| Condition | Seed means | Mean +/- sd |
| --- | --- | --- |
| A from-scratch SAC | -46.0, -46.8, -54.9 | -49.2 +/- 4.9 |
| B BC-init + demo replay | -49.1, -51.6, -52.2 | -51.0 +/- 1.6 |
| C blackbox dynamics augmentation | -43.9, -54.8, -41.9 | -46.9 +/- 6.9 |

No run reaches a positive return, so the task is unsolved and the experiment
lacked the signal to separate conditions. D and E cannot be run meaningfully at
all: the nominal physics model is the simulator itself, so the residual target is
identically zero, and the domain randomization the spec prescribes to fix that is
unimplemented. See [`FINDINGS_residual_degeneracy.md`](FINDINGS_residual_degeneracy.md).

### Three findings that were artifacts, not methods

1. **BC-init's apparent 48 percent advantage did not survive.** A singular
   rotation conversion, dividing by `2 * sin(angle)`, generated observations up
   to 6.35e8 and critic loss to 1.9e15. Max observation fell to 71-84 and max
   qf_loss to 1,844 across 9 valid runs. See
   [`FINDINGS_metrics_integrity.md`](FINDINGS_metrics_integrity.md) for the
   related resume defect.
2. **Residual augmentation's apparent weakness was a broken target.** The
   hand-written physics model disagreed with its simulator by a median 12.9 per
   step, in a file with 13 percent coverage. After fixing that, the residual
   turned out to be identically zero for a structural reason.
3. **Condition C's apparent promise was one lucky seed.** C s2 at -41.9 was the
   best single run in the matrix. With s1 at -54.8, the spread is -41.9 to
   -54.8, indistinguishable from A.

All three were found by reading code or checking ground truth, never by watching
a metric. That is the transferable lesson, and it is why
`tests/python/test_invariants.py` now pins the conventions whose violations
produced them.

## 3. Honest summary

The conversion pipeline works on real human motion and demonstrably improves
trajectory kinematics, and on the data set where both variants exist it halves
imitation error. What it has not done is demonstrate either benefit on real data
end to end, and the input set for that experiment is not regenerable from
committed code.

The ablation is a null result on an unsolved task.
