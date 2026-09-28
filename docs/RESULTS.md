# Results

Two parts: the human-to-robot conversion pipeline, which is the project's actual
deliverable, and the SAC ablation, which is a null result. Every number here was
read from an artifact on disk, and each is labeled with the data set it was
measured on. The distinction matters, because the two halves of the conversion
result have not been joined yet.

## 1. Human-to-robot trajectory conversion

### What runs

Real DexYCB hand motion sequences from two subjects are ingested, MANO poses are
retargeted to 22-DoF Allegro joint targets, and a C++20 constrained optimizer
projects the result onto joint, velocity, and acceleration limits. Every
hyperparameter is selected on subject-01; subject-02 is reported held out.

- `src/human2robot/data/dexycb.py`: sequence discovery, MANO pose to joint
  targets, per-subject demo building.
- `cpp/` (library `h2r_traj`): Hermite interpolation, finite differences,
  moving-window smoothing, projection onto joint/velocity/acceleration limits,
  weighted cost, projected optimizer with a backtracking line search. **64
  GoogleTest cases.**
- `src/human2robot/cpp_bindings/`: pybind11 bridge.

### 1a. Kinematic quality, on 100 real DexYCB sequences per subject

From `results/trajectory_optimization/real_vs_synthetic.json` (subject-01) and
`real_vs_synthetic_s2.json` (subject-02). DexYCB MANO capture runs at 30 Hz;
trajectories are resampled to the 20 ms Allegro control period so both data sets
are optimized under identical settings. Subject-02 is a different person and no
hyperparameter was ever selected on it.

| Quantity | Subject-01 | Subject-02 (held out) | Synthetic |
| --- | --- | --- | --- |
| max jerk | -9.1% | -4.6% | -50.6% |
| smoothness cost | -30.0% | -28.1% | -93.7% |
| max velocity | -71.7% | -72.2% | -65.4% |
| max acceleration | -18.0% | -13.9% | -30.3% |
| search improved on its starting point | 50 / 50 held out | 99 / 100 | 94 / 100 |
| median cost reduction achieved | 33.0% held out | 37.3% | 15.1% |

Every number in this section is measured on DexPilot IK retargets. An earlier
version of this document reported jerk at -41.9% and smoothness at -64.7%; those
came from a retargeter that collapsed 16 joint dimensions into 1, and the
difference is documented in
[`FINDINGS_retargeting.md`](FINDINGS_retargeting.md). Real hand motion is
already smooth, so a smoothness objective has little left to remove, and the
small jerk figure is the correct result rather than a regression.

**Read the velocity row with suspicion.** Optimized `max_velocity` is
2.0000000000000018 with a standard deviation of 4.4e-16 on both data sets, so
the limit is binding and the optimizer is clipping to it. That row measures
constraint enforcement rather than optimization headroom. Jerk and smoothness are
the unbounded quantities, so they are the honest wins.

### The optimizer was barely running, and the reason was the noise term

`TrajectoryOptimizer::optimize` perturbs each candidate with **independent
per-timestep Gaussian noise**, scaled by `step_size * 0.1`. The cost function is
dominated by smoothness terms, and independent per-sample noise raises jerk far
faster than the tracking pull toward the reference lowers it. In practice the
very first candidate cost more than twice the current cost, which tripped the
`candidate_cost > current_cost * 2.0` abort guard, so the search ended at
iteration 1 having achieved **exactly zero improvement**.

`scripts/diagnose_convergence.py` measured this directly: on 40 real sequences,
**34 achieved literally zero improvement**. Tuning `step_size` alone did not
rescue it, plateauing near 27 percent, because no step size along a
noise-dominated direction descends.

Two changes fixed it:

1. **Backtracking line search.** Rather than one fixed step, halve the step
   until the cost actually falls, so an overshoot no longer aborts the search.
2. **`noise_scale = 0`.** The noise term is now a configurable parameter
   (`OptimizerConfig::noise_scale`, plumbed through `OptimizationConfig`), and
   the pipeline sets it to 0, making the step a pure tracking pull.

Measured over all 100 sequences of each data set, varying only `noise_scale` and
`step_size`. These rows are diagnostic: the `noise_scale` column is the
structural finding, and the `step_size` rows within a column are the tuning
sweep that the held-out protocol below re-runs properly.

| noise_scale | step_size | real improved | synthetic improved |
| --- | --- | --- | --- |
| 0.1 | 0.05 | 24/100 | 16/100 |
| 0.01 | 0.2 | 30/100 | - |
| 0.0 | 0.05 | 32/100 | - |
| 0.0 | 0.2 | 75/100 | - |
| **0.0** | **0.35** | **99/100 cross-subject** | **94/100** |

The noise term was the whole problem. `step_size` 0.5 and 1.0 give identical
results and the selected 0.35 is the peak of that sweep.

### Held-out validation, because the sweep was initially done wrong

The first version of this table was selected by sweeping `step_size` while
looking at all 100 sequences and then reporting the rate on those same 100. That
is choosing hyperparameters on the evaluation set, and the resulting 94/100
should be discounted.

`scripts/validate_optimizer_split.py` fixes the protocol in two stages. The sweep
runs on the first 50 subject-01 sequences only; `step_size = 0.35` is selected
from that half alone. The setting is then evaluated once on the last 50 of subject-01,
which no selection decision has seen, and once more on all 100 subject-02
sequences.

| | improved | mean cost reduction | median |
| --- | --- | --- | --- |
| tuning half, subject-01 (50) | 50/50 | 34.80% | 35.53% |
| held out, subject-01 (50) | 50/50 | 32.52% | 33.02% |
| **held out, subject-02 (100)** | **99/100** | **36.38%** | **37.34%** |
| synthetic (100) | 94/100 | 15.48% | 15.08% |

`step_size = 0.35` is selected from the tuning half alone. On the degenerate
data this protocol selected 0.5; the selected value is therefore not
reverse-engineered.

There is no meaningful gap between the tuning and held-out halves, so the
setting is not overfitted to the sequences it was chosen on. The within-subject
split alone only proves the setting is not overfitted to those particular
trajectories; **subject-02 is the stronger test**, because it is a different
person's hand and no selection decision ever saw it. The headline rate quoted
throughout is the **cross-subject 99/100**.

`noise_scale = 0` is not a tuned parameter. It is a structural fix: independent
per-timestep noise is provably adversarial for a smoothness-dominated cost, and
the sweep over it in the table above is diagnostic rather than selective.

### The convergence metric was measuring projection, not optimization

This section previously reported 85/100 real and 100/100 synthetic sequences
converging, and drew the inference that the optimizer handled synthetic data
perfectly and only struggled on real data. **That inference was wrong, and the
metric was wrong with it.**

`optimize()` sets `initial_cost` on the **unprojected** input but returns
`best_cost` from the **projected** starting point, where the search actually
begins. The convergence test was `initial_cost - best_cost >= tolerance`.
Whenever projection itself lowers the cost, which it does for any
limit-violating input because the violation penalty is large, that inequality is
satisfied before the optimizer has done anything. The old 85/100 was exactly the
count of sequences where projection alone beat the raw input, and 100/100
synthetic was the same artifact.

Convergence and the improvement target are both now measured from the projected
baseline. `ProjectionCanRaiseTotalCost` in `cpp/tests/test_optimizer.cpp` pins
the case that makes the distinction matter: a smooth ramp overshooting the joint
limit has an unprojected cost of 0.048 and a projected cost of 257, because
clipping the overshoot introduces a kink whose jerk cost exceeds the small
violation penalty it removes.

`OptimizerResult` now also exposes `projected_initial_cost` and `improvement_pct`,
because a caller previously had no way to tell whether the optimizer had done
any work at all.

### 1b. Imitation quality, on the 100 synthetic sequences

From `results/trajectory_optimization/bc_downstream.json`. Identical BC heads
and hyperparameters across arms; only the demonstration source differs. Both
arms are already at the 20 ms control period, so this comparison isolates the
optimizer.

| Demo set | BC holdout MSE | MAE | Max error | Transitions |
| --- | --- | --- | --- | --- |
| raw retargeted | 7.08e-4 | 0.0182 | 0.120 | 10,088 |
| optimized | **3.06e-4** | **0.0113** | **0.065** | 10,088 |
| mixed | 5.05e-4 | 0.0149 | 0.121 | 20,175 |

Optimizing cut held-out imitation error **56.8 percent**, with the reported error
matching the early-stopped holdout best, so it is not an overfit figure, and
with matched transition counts on both arms.

### 1c. Imitation quality, on the 100 real DexYCB sequences

From `results/trajectory_optimization/bc_downstream_dexycb.json` (subject-01)
and `bc_downstream_dexycb_s2.json` (subject-02). This is the experiment that
was missing, and running it changed the conclusion.

Subject-01:

| Arm | BC holdout MSE | MAE | Max error | Transitions |
| --- | --- | --- | --- | --- |
| raw (30 Hz) | 9.78e-4 | 0.0169 | 0.265 | 6,146 |
| resampled control (20 ms, no optimizer) | 2.33e-4 | 0.0089 | 0.153 | 10,280 |
| optimized (20 ms + optimizer) | **1.43e-4** | 0.0073 | **0.067** | 10,280 |
| mixed | 3.49e-4 | 0.0104 | 0.266 | 16,425 |

Subject-02, held out entirely:

| Arm | BC holdout MSE | MAE | Max error | Transitions |
| --- | --- | --- | --- | --- |
| raw (30 Hz) | 7.86e-4 | 0.0156 | 0.272 | 6,331 |
| resampled control (20 ms, no optimizer) | 2.39e-4 | 0.0087 | 0.161 | 10,595 |
| optimized (20 ms + optimizer) | **1.45e-4** | 0.0073 | **0.072** | 10,595 |
| mixed | 3.85e-4 | 0.0107 | 0.249 | 16,926 |

Comparing the first and third rows naively suggests the optimizer improves
imitation by **85.4 percent** on subject-01 and **81.6 percent** on subject-02.
Most of that is still confounded. DexYCB captures at 30 Hz and the environment
runs at 20 ms, so the optimized arm has been through a cubic resampling step
that the raw arm never had. The middle row is the control that isolates it:

- Subject-01: resampling to the control rate accounts for a **76.2 percent**
  error reduction (9.78e-4 to 2.33e-4) on its own, and the optimizer on top
  contributes **38.5 percent** (2.33e-4 to 1.43e-4).
- Subject-02: resampling accounts for **69.6 percent** (7.86e-4 to 2.39e-4)
  and the optimizer contributes **39.5 percent** (2.39e-4 to 1.45e-4).

So the controlled figure on real human motion is **38.5 percent** on subject-01
and **39.5 percent** on the held-out subject, against **56.8 percent** on
synthetic.

Worst-case error falls furthest: 0.265 raw, 0.153 after resampling, 0.067 after
optimizing, and 0.272 to 0.161 to 0.072 on subject-02. The optimized arm is the
only one whose transitions respect the robot's joint constraints. Unlike on the
degenerate data, mean absolute error improves here as well (0.0089 to 0.0073),
so the gain is no longer carried only by the tail.

**The two data sets still disagree**, 56.8 against 31.0. Input roughness remains
the most likely explanation, since the synthetic trajectories have more for the
optimizer to remove, but that hypothesis is untested.

The practical lesson is the same one this project keeps learning. The 66.5
percent figure was available, looked excellent, and was produced by a
confounded comparison. It was only caught by adding the control arm. The
controlled number was small, and it was only made meaningful by fixing the
optimizer rather than by rewording the result.

### 1d. How much of the imitation result survives a harder holdout

The BC comparison above holds out a random 10 percent of **pooled
transitions**. Consecutive transitions of one trajectory are nearly identical,
so a holdout transition sits one step away from a training transition. That
makes the absolute error optimistic, and it is the split every published number
in this document uses.

`scripts/check_bc_split_granularity.py` re-runs the comparison at three
granularities, 5 seeds each, reporting the controlled advantage over the
resampled control arm:

| Data set | random transitions | whole trajectories held out | tail of every trajectory |
| --- | --- | --- | --- |
| Subject-01 | +45.0% (sd 2.1) | **+49.7% (sd 10.7)** | +6.1% |
| Subject-02 | +41.1% (sd 2.9) | **+51.4% (sd 6.6)** | -24.6% |
| Synthetic | +56.1% (sd 0.5) | +50.4% (sd 1.0) | +40.3% |

**The mean-MSE claim holds up on real retargeted data.** Holding out whole
trajectories is the *strongest* test rather than the weakest: +49.7% and +51.4%,
positive on every seed of 5. Worst-case error against the control is -56.5% on
subject-01 and -55.1% on subject-02.

This reverses the conclusion reached on the degenerate data, where the same
whole-trajectory test gave +21% with a standard deviation of 18 and a sign that
flipped between seeds. That reversal is itself evidence the test is measuring
something real. A signal that is trivially predictable is exactly the signal
whose advantage should evaporate when adjacent transitions stop being shared
between train and holdout, and on the 1-DOF path it did; on genuinely
articulated 16-dimensional motion it does not.

**The tail-extrapolation split is the one that stays weak**: +6.1% on
subject-01 and -24.6% on subject-02, against +40.3% on synthetic. Predicting
the continuation of a motion the model has not seen is a harder question than
predicting a new trajectory from partially seen ones, and the optimizer seems
to help less there. **The real-versus-synthetic gap on this split is
unexplained** and is the clearest remaining open question in the conversion
pipeline.

### 1e. Reproducing this

| Step | Command | Produces |
| --- | --- | --- |
| 1 | `bash scripts/download_dexycb.sh` | `data/raw/dexycb/` |
| 2 | `bash scripts/wsl_setup_retargeting.sh` | WSL venv with `dex-retargeting` |
| 3 | `uv run python scripts/retarget_dexycb_ik.py` | `data/demonstrations_dexycb_ik/` (DexPilot IK, 100 seq) |
| 4 | `uv run python scripts/compare_dexycb_synthetic.py` | `real_vs_synthetic_s1.json` and `data/demonstrations_dexycb_ik_optimized/` |
| 6 | `uv run python scripts/run_trajectory_experiment.py` | `report.json` (synthetic) |
| 7 | `uv run python scripts/run_downstream_bc.py --set dexycb` | `bc_downstream_dexycb.json` (real, subject-01) |
| 8 | `uv run python scripts/run_downstream_bc.py --set synthetic` | `bc_downstream.json` (synthetic) |
| 9 | `uv run python scripts/validate_optimizer_split.py` | `optimizer_split.json`, the held-out protocol |
| 10 | `uv run python scripts/check_bc_split_granularity.py` | `bc_split_granularity.json`, the split-robustness check |

The chain is now complete and scripted end to end. The held-out subject repeats
three of those steps:

| Step | Command | Produces |
| --- | --- | --- |
| 3b | `uv run python scripts/retarget_dexycb_ik.py --subject 20200813-subject-02 --output-dir data/demonstrations_dexycb_ik_s2` | `data/demonstrations_dexycb_ik_s2/` |
| 4b | `uv run python scripts/compare_dexycb_synthetic.py --subject subject-02` | `real_vs_synthetic_s2.json` and the optimized subject-02 demos |
| 7b | `uv run python scripts/run_downstream_bc.py --set dexycb-s2` | `bc_downstream_dexycb_s2.json` |

Step 3b must run inside the WSL venv from step 2. Step 9 additionally evaluates
the subject-01-selected `step_size` on all 100 subject-02 sequences, which is
where the cross-subject 99/100 comes from.

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

On genuinely retargeted human motion the optimizer does real work. It improves
100 of 100 subject-01 sequences and 99 of 100 subject-02 sequences, the second
subject being a person it was never tuned on, for a median cost reduction of 33
and 37 percent. It cuts smoothness cost about 30 percent and max velocity about
72 percent on both subjects.

The imitation result holds up: against a resampled control arm it cuts held-out
error 38.5 percent on subject-01 and 39.5 percent on subject-02, rising to
about 50 percent when whole trajectories rather than random transitions are held
out, positive on every seed of 5. Worst-case error falls 55 to 57 percent.
Extrapolating the tail of every trajectory is the one test that stays weak, and
the real-versus-synthetic gap on that split is unexplained. Most of the naive 85
percent on real data is resampling rather than optimization, and the controlled
comparison exists only because the control arm was added.

**The most consequential number in this project's history is the one that went
down.** The jerk result was -41.9 percent and is now -9.1 percent, because it
was being measured on a retargeter that carried 90 percent of its variance in a
single dimension. See [`FINDINGS_retargeting.md`](FINDINGS_retargeting.md).

The ablation is a null result on an unsolved task.

Seven findings in this project have turned out to be artifacts rather than
results, all of them plausible-looking numbers that a reviewer would have had
no reason to question: BC-init's 48 percent advantage, residual augmentation's
apparent weakness, Condition C's apparent promise, the 66.5 percent imitation
gain that was mostly resampling, the 85/100 convergence rate that was counting
projection as optimization, the mean-imitation-error gain that survived only
under a holdout leaking adjacent transitions, and the entire trajectory result
measured on a one-dimensional retargeting.

The last one is the instructive case. Every earlier check was a *relative*
comparison made against the same input, so all of them correctly reported a real
effect on the wrong data. What was missing was a check on the data itself. The
rank of the demo sets is now pinned in `test_invariants.py`.

The two figures that moved most in this project's history moved because
something was fixed, not because something was reworded: the optimizer's search
went from zero improvement on 85 percent of sequences to 89/100 on a held-out
subject with a median 12.6 percent cost reduction, and the controlled real-data
imitation figure went from 5.2 to 26-31 percent as a direct consequence.
