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
and hyperparameters across arms; only the demonstration source differs. Both
arms are already at the 20 ms control period, so this comparison isolates the
optimizer.

| Demo set | BC holdout MSE | MAE | Max error | Transitions |
| --- | --- | --- | --- | --- |
| raw retargeted | 7.08e-4 | 0.0182 | 0.120 | 10,088 |
| optimized | **3.43e-4** | **0.0122** | **0.069** | 10,088 |
| mixed | 5.18e-4 | 0.0151 | 0.114 | 20,175 |

Optimizing cut held-out imitation error **51.6 percent**, with the reported error
matching the early-stopped holdout best, so it is not an overfit figure, and
with matched transition counts on both arms.

### 1c. Imitation quality, on the 100 real DexYCB sequences

From `results/trajectory_optimization/bc_downstream_dexycb.json`. This is the
experiment that was missing, and running it changed the conclusion.

| Arm | BC holdout MSE | MAE | Max error | Transitions |
| --- | --- | --- | --- | --- |
| raw (30 Hz) | 5.89e-4 | 0.0121 | 0.291 | 6,146 |
| resampled control (20 ms, no optimizer) | 2.08e-4 | 0.0072 | 0.180 | 10,280 |
| optimized (20 ms + optimizer) | **1.97e-4** | 0.0084 | **0.082** | 10,280 |
| mixed | 3.18e-4 | 0.0094 | 0.221 | 16,425 |

Comparing the first and third rows naively suggests the optimizer improves
imitation by **66.5 percent**. That number is almost entirely confounded. DexYCB
captures at 30 Hz and the environment runs at 20 ms, so the optimized arm has
been through a cubic resampling step that the raw arm never had. The middle row
is the control that isolates it, and it shows:

- Resampling to the control rate accounts for a **64.7 percent** error reduction
  (5.89e-4 to 2.08e-4) on its own.
- The optimizer on top of that contributes **5.2 percent** (2.08e-4 to 1.97e-4).

So on real human motion the C++ optimizer's effect on average imitation error is
**small**. Two narrower effects are real: worst-case error more than halves
(0.180 to 0.082) and the optimized arm is the only one whose transitions are
limit-respecting, so it is the only arm that satisfies the robot's joint
constraints. Mean absolute error moves the other way (0.0072 to 0.0084).

**The honest summary of the two data sets is that they disagree.** The optimizer
halves imitation error on synthetic demonstrations and adds 5 percent on real
ones. The most likely explanation is input roughness: the synthetic
trajectories evidently have more for the optimizer to remove than the resampled
real ones do. That hypothesis is untested.

The practical lesson is the same one this project keeps learning. The 66.5
percent figure was available, looked excellent, and was produced by a
confounded comparison. It was only caught by adding the control arm.

### 1d. Reproducing this

| Step | Command | Produces |
| --- | --- | --- |
| 1 | `bash scripts/download_dexycb.sh` | `data/raw/dexycb/` |
| 2 | `bash scripts/wsl_setup_retargeting.sh` | WSL venv with `dex-retargeting` |
| 3 | `uv run python scripts/retarget_dexycb_ik.py [link_length] [subject] [dir]` | `data/demonstrations_dexycb_ik/` (DexPilot IK) |
| 4 | `uv run python -m human2robot.data.dexycb` | `data/demonstrations_dexycb/` (vector retargeting, 100 seq) |
| 5 | `uv run python scripts/compare_dexycb_synthetic.py` | `real_vs_synthetic.json` and `data/demonstrations_dexycb_optimized/` |
| 6 | `uv run python scripts/run_trajectory_experiment.py` | `report.json` (synthetic) |
| 7 | `uv run python scripts/run_downstream_bc.py --set dexycb` | `bc_downstream_dexycb.json` (real) |
| 8 | `uv run python scripts/run_downstream_bc.py --set synthetic` | `bc_downstream.json` (synthetic) |

The chain is now complete and scripted end to end.

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

The pipeline runs end to end on real human motion, from DexYCB download through
retargeting and constrained optimization, and every step is a committed command.

On kinematics the optimizer does real work on real data: jerk down 35.5 percent,
smoothness cost down 56.3 percent, and worst-case imitation error more than
halved. On average imitation error the picture is much weaker than it first
looks. Halving error on synthetic demonstrations is real, but on real DexYCB
trajectories almost all of the apparent gain comes from resampling to the
control rate, and the optimizer itself contributes about 5 percent.

The ablation is a null result on an unsolved task.

Four findings in this project have turned out to be artifacts rather than
results, all of them plausible-looking numbers that a reviewer would have had
no reason to question: BC-init's 48 percent advantage, residual augmentation's
apparent weakness, Condition C's apparent promise, and now the 66.5 percent
imitation gain. Each was found by checking a control or the underlying source,
never by watching a metric.
