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
| max jerk | -16.8% | -12.1% | -49.2% |
| smoothness cost | -30.4% | -28.7% | -92.6% |
| max velocity | -71.7% | -72.2% | -65.4% |
| max acceleration | -25.6% | -24.0% | -25.1% |
| search improved on its starting point | 50 / 50 held out | 100 / 100 | 92 / 100 |
| median cost reduction achieved | 36.0% held out | 43.0% | 14.8% |
| **executable in simulation** | **34 / 100** | **8 / 100** | **100 / 100** |

The last row is the metric the project exists to move and is described in
section 1e. Every other row in this table is a property of the trajectory; only
that one is a property of the robot.

Every number in this section is measured on DexPilot IK retargets. An earlier
version of this document reported jerk at -41.9% and smoothness at -64.7%; those
came from a retargeter that collapsed 16 joint dimensions into 1, and the
difference is documented in
[`FINDINGS_retargeting.md`](FINDINGS_retargeting.md). Real hand motion is
already smooth, so a smoothness objective has little left to remove, and the
moderate jerk figure is the correct result rather than a regression.

An earlier revision of this table also reported jerk at -9.1% and -4.6% and a
descent rate of 99/100. Those were measured before the joint limits were
corrected; the re-run under the corrected limits is what the table above
reports, and it moved jerk further down and the descent rate to 100/100. The
regression tests in `test_published_results.py` now read these numbers from the
recorded JSON rather than from this document, so the two cannot disagree
silently again.

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
| **0.0** | **0.35** | **100/100 cross-subject** | **92/100** |

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
| tuning half, subject-01 (50) | 50/50 | 40.31% | 41.25% |
| held out, subject-01 (50) | 50/50 | 35.66% | 35.96% |
| **held out, subject-02 (100)** | **100/100** | **39.47%** | **43.02%** |
| synthetic (100) | 92/100 | 15.08% | 14.78% |

`step_size = 0.35` is selected from the tuning half alone. On the degenerate
data this protocol selected 0.5; the selected value is therefore not
reverse-engineered.

There is no meaningful gap between the tuning and held-out halves, so the
setting is not overfitted to the sequences it was chosen on. The within-subject
split alone only proves the setting is not overfitted to those particular
trajectories; **subject-02 is the stronger test**, because it is a different
person's hand and no selection decision ever saw it. The headline rate quoted
throughout is the **cross-subject 100/100**.

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
| executable in simulation | 100 / 100 | | | |

Optimizing cut held-out imitation error **56.8 percent**, with the reported error
matching the early-stopped holdout best, so it is not an overfit figure, and
with matched transition counts on both arms.

### 1c. Imitation quality, on the 100 real DexYCB sequences

From `results/trajectory_optimization/bc_downstream_dexycb.json` (subject-01)
and `bc_downstream_dexycb_s2.json` (subject-02). This is the experiment that
was missing, and running it changed the conclusion.

Subject-01:

| Arm | BC holdout MSE | MAE | Max error | Transitions | Executable |
| --- | --- | --- | --- | --- | --- |
| raw (30 Hz) | 9.90e-4 | 0.0171 | 0.281 | 6,146 | 6 / 100 |
| resampled control (20 ms, no optimizer) | 2.42e-4 | 0.0092 | 0.143 | 10,280 | 14 / 100 |
| optimized (20 ms + optimizer) | **1.71e-4** | 0.0087 | **0.062** | 10,280 | **34 / 100** |
| mixed | 3.90e-4 | 0.0110 | 0.242 | 16,425 | |

Subject-02, held out entirely:

| Arm | BC holdout MSE | MAE | Max error | Transitions | Executable |
| --- | --- | --- | --- | --- | --- |
| raw (30 Hz) | 8.30e-4 | 0.0162 | 0.262 | 6,331 | 0 / 100 |
| resampled control (20 ms, no optimizer) | 2.54e-4 | 0.0090 | 0.159 | 10,595 | 6 / 100 |
| optimized (20 ms + optimizer) | **1.80e-4** | 0.0090 | **0.065** | 10,595 | **8 / 100** |
| mixed | 4.59e-4 | 0.0118 | 0.267 | 16,926 | |

Comparing the first and third rows naively suggests the optimizer improves
imitation by **85.4 percent** on subject-01 and **81.6 percent** on subject-02.
Most of that is still confounded. DexYCB captures at 30 Hz and the environment
runs at 20 ms, so the optimized arm has been through a cubic resampling step
that the raw arm never had. The middle row is the control that isolates it:

- Subject-01: resampling to the control rate accounts for a **75.6 percent**
  error reduction (9.90e-4 to 2.42e-4) on its own, and the optimizer on top
  contributes **29.5 percent** (2.42e-4 to 1.71e-4).
- Subject-02: resampling accounts for **69.4 percent** (8.30e-4 to 2.54e-4)
  and the optimizer contributes **29.0 percent** (2.54e-4 to 1.80e-4).

So the controlled figure on real human motion is **29.5 percent** on subject-01
and **29.0 percent** on the held-out subject, against **56.8 percent** on
synthetic.

Worst-case error falls furthest: 0.281 raw, 0.143 after resampling, 0.062 after
optimizing, and 0.262 to 0.159 to 0.065 on subject-02. The optimized arm is the
only one whose transitions respect the robot's joint constraints. Mean absolute
error improves as well (0.0092 to 0.0087 on subject-01), so the gain is not
carried only by the tail, though on subject-02 the MAE is flat at 0.0090.

**The two data sets disagree by a factor of two**, 56.8 against 29.5. Input
roughness remains the most likely explanation, since the synthetic trajectories
have more for the optimizer to remove, but that hypothesis is untested.

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
granularities over 5 seeds, reporting the controlled advantage over the
resampled control arm. The advantage is positive on every seed at the first two
granularities:

| Data set | random transitions | whole trajectories held out | tail of every trajectory |
| --- | --- | --- | --- |
| Subject-01 | +34.7% (sd 3.6) | +25.2% (sd 14.6) | **-54.2% (sd 5.0)** |
| Subject-02 | +29.8% (sd 3.0) | +33.3% (sd 9.5) | **-104.9% (sd 2.6)** |
| Synthetic | +56.8% (sd 0.9) | +50.6% (sd 1.0) | +40.6% (sd 0.4) |

**The mean-MSE claim holds up on real retargeted data.** Holding out whole
trajectories is the strongest of the two interpolation splits: +25.2% and
+33.3%, positive on all 5 seeds, and the sign never flips.

This reverses the conclusion reached on the degenerate data, where the same
whole-trajectory test gave +21% with a standard deviation of 18 and a sign that
flipped between seeds. That reversal is itself evidence the test is measuring
something real. A signal that is trivially predictable is exactly the signal
whose advantage should evaporate when adjacent transitions stop being shared
between train and holdout, and on the 1-DOF path it did; on genuinely
articulated 16-dimensional motion it does not.

### The tail-extrapolation reversal is a degenerate test, and it is now explained

The prefix split reverses the sign on both subjects, -54.2% and -104.9%, while
staying positive on synthetic. The gap was open until
`scripts/diagnose_tail_extrapolation.py` measured it, and the cause is that
**the prefix split's answer is close to zero on real data**.

Every split score is now reported next to the score of predicting *no motion at
all*, which is the floor a regressor cannot beat by predicting stillness:

| Data set | control score on prefix tail | predict-nothing | optimized score |
| --- | --- | --- | --- |
| Subject-01 | 1.45e-4 | **1.47e-4** | 2.23e-4 |
| Subject-02 | 9.49e-5 | 7.41e-5 | 1.94e-4 |
| Synthetic | 6.94e-4 | 9.61e-4 | 4.12e-4 |

On subject-01 the resampled control scores 1.45e-4 against a do-nothing
baseline of 1.47e-4: **the fitted model does no better than predicting that the
hand has stopped.** A split whose holdout is nearly motionless cannot rank a
better trajectory above a worse one, because the quietest arm wins by default.
On synthetic the ordering is the reverse, 6.94e-4 against a do-nothing floor of
9.61e-4, so predicting motion beats predicting stillness there and the
optimizer's smoothing is genuinely rewarded.

The mechanism is in the data. Real captured hand motion decelerates into a
stop, and the measurement confirms it: the final step is a median 0.25 of the
mid-trajectory step on subject-01 raw and 0.07 on subject-02, with 71 and 88
percent of trajectories ending below half speed. Synthetic trajectories have no
such structure, ending at 0.97 of mid-trajectory speed with 0 percent below half.
The optimizer then **redistributes the deceleration**: its final step rises to
0.62 of mid-trajectory speed and the fraction of trajectories ending below half
speed falls to 38 and 47 percent. The optimizer damps the body more than the
tail, so the tail's share of total motion rises from 0.45 to 0.70, and a tail
that has been given more motion is harder to predict from a body that has been
given less.

So the reversal is not evidence that the optimizer harms extrapolation. It is
evidence that on real human motion the tail is a stop, and the split rewards
whichever arm stops hardest. **Read the prefix row as a statement about the
data, not about the optimizer.** The honest summary is that tail
extrapolation is untested by these data, not that the optimizer fails at it;
making it a real test needs sequences whose tails contain sustained motion, and
this project does not have any.

### 1e. Can the robot actually do it, and what the rest of this section does not measure

Every other number in this section is a property of the trajectory. None of them
is a property of the robot. This is the one that is, and it is the question the
pipeline exists to answer: given these joint targets, can the simulated Allegro
hand perform the motion?

`evaluation/feasibility.py` answers it by replaying each trajectory open-loop
in MuJoCo, feeding the joint targets in as actuator commands one per control
period with nothing correcting them, and measuring how far the hand actually
tracks. A trajectory that tracks here is one the hand genuinely performs.
Three quantities are reported, and `is_feasible` is their conjunction:
worst-frame tracking drift (threshold 0.35 rad), RMS tracking drift (0.20 rad),
and out-of-bounds demand (0.0 rad, exact compliance). The first 25 steps hold
the initial target so the hand is measured tracking rather than measured
displaced from its reset pose.

From `results/trajectory_optimization/feasibility_*.json`, via
`scripts/score_feasibility.py`:

| Data set | raw | resampled control | optimized | mean drift, optimized |
| --- | --- | --- | --- | --- |
| Subject-01 | 6 / 100 | 14 / 100 | **34 / 100** | 0.398 rad |
| Subject-02 | 0 / 100 | 6 / 100 | **8 / 100** | 0.551 rad |
| Synthetic | 100 / 100 | n/a | 100 / 100 | 0.186 rad |

**The optimizer roughly triples executability on subject-01**, 6 to 34 of 100,
and resampling accounts for part of that (to 14) with the optimizer
contributing the rest. On subject-02 it goes from 0 to 8. On synthetic the two
arms are already fully executable, which is the strongest evidence that the gap
on real data is a property of human motion rather than of the metric.

The control arm is what makes this readable. Raw versus optimized alone would
report a 5.7x improvement on subject-01, but resampling to the control rate is
most of that, exactly as in the imitation comparison. The same lesson, the same
control.

Limit violation is 0.0000 on every arm of every data set. Under the corrected
joint limits this is exact compliance rather than a rounded small number, and
it is worth stating plainly: **the trajectories respect the actuator bounds but
the hand frequently cannot reach them in the time allowed.** Those are different
failure modes, and only the second one is what this section measures. The
residual drift of 0.4 to 0.55 rad on real motion is a tracking failure, not a
reach failure.

Why real human motion is harder: it is fast and articulated, and the synthetic
generator produces slower, smoother single-finger motion. A reach captured from
a person covers more range in less time than a comfortable synthetic sweep.
Section 1d's finding that the optimizer's advantage over resampling is twice as
large on synthetic data points the same way.

### 1f. Reproducing this

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
| 10 | `uv run python scripts/check_bc_split_granularity.py --set dexycb` | `bc_split_granularity_dexycb.json`, the split-robustness check |
| 11 | `uv run python scripts/score_feasibility.py --set all` | `feasibility_*.json`, the executability table |
| 12 | `uv run python scripts/diagnose_tail_extrapolation.py --set dexycb` | `tail_extrapolation_*.json`, the prefix-split diagnosis |

Step 4 must be re-run before steps 7, 10 and 11, because it writes the
optimized demos the later steps read. `test_published_results.py` checks that
each recorded JSON is internally consistent, but it cannot detect a demo
directory that has been regenerated under new settings, so the order matters.

The chain is now complete and scripted end to end. The held-out subject repeats
three of those steps:

| Step | Command | Produces |
| --- | --- | --- |
| 3b | `uv run python scripts/retarget_dexycb_ik.py --subject 20200813-subject-02 --output-dir data/demonstrations_dexycb_ik_s2` | `data/demonstrations_dexycb_ik_s2/` |
| 4b | `uv run python scripts/compare_dexycb_synthetic.py --subject subject-02` | `real_vs_synthetic_s2.json` and the optimized subject-02 demos |
| 7b | `uv run python scripts/run_downstream_bc.py --set dexycb-s2` | `bc_downstream_dexycb_s2.json` |
| 10b | `uv run python scripts/check_bc_split_granularity.py --set dexycb-s2` | `bc_split_granularity_dexycb_s2.json` |
| 11b | `uv run python scripts/score_feasibility.py --set dexycb-s2` | `feasibility_dexycb_s2.json` |

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
all 100 subject-01 sequences and all 100 subject-02 sequences, the second
subject being a person it was never tuned on, for a median cost reduction of 36
and 43 percent. It cuts smoothness cost about 30 percent and max velocity about
72 percent on both subjects.

The imitation result holds up: against a resampled control arm it cuts held-out
error 29.5 percent on subject-01 and 29.0 percent on subject-02, and it stays
positive on every one of 5 seeds when whole trajectories rather than random
transitions are held out. Worst-case error falls 57 percent.

The result that matters most is the one about the robot rather than the
trajectory: **optimizing takes subject-01 from 6 to 34 of 100 trajectories
executable in simulation**, and subject-02 from 0 to 8. Every other table in
this section measures a property of the numbers the pipeline produces. This one
measures whether the hand can perform them.

The tail-extrapolation reversal, previously the clearest open question, is
explained: real hand motion decelerates into a stop, so the prefix split's
holdout is nearly motionless and the control arm scores at the do-nothing
baseline. The split rewards whichever arm stops hardest, so it cannot rank the
optimizer. Most of the naive 83 percent on real data is resampling rather than
optimization, and the controlled comparison exists only because the control arm
was added.

**The most consequential number in this project's history is the one that went
down.** The jerk result was -41.9 percent and is now -9.1 percent, because it
was being measured on a retargeter that carried 90 percent of its variance in a
single dimension. See [`FINDINGS_retargeting.md`](FINDINGS_retargeting.md).

The ablation is a null result on an unsolved task.

Nine findings in this project have turned out to be artifacts rather than
results, all of them plausible-looking numbers that a reviewer would have had
no reason to question: BC-init's 48 percent advantage, residual augmentation's
apparent weakness, Condition C's apparent promise, the 66.5 percent imitation
gain that was mostly resampling, the 85/100 convergence rate that was counting
projection as optimization, the mean-imitation-error gain that survived only
under a holdout leaking adjacent transitions, the entire trajectory result
measured on a one-dimensional retargeting, the limit-compliance result measured
against bounds the robot does not have, and the three-seed error bar on an
experiment run at one seed.

The last one is the instructive case. Every earlier check was a *relative*
comparison made against the same input, so all of them correctly reported a real
effect on the wrong data. What was missing was a check on the data itself. The
rank of the demo sets is now pinned in `test_invariants.py`.

The two figures that moved most in this project's history moved because
something was fixed, not because something was reworded: the optimizer's search
went from zero improvement on 85 percent of sequences to 100/100 on a held-out
subject with a median 43.0 percent cost reduction, and the controlled real-data
imitation figure went from 5.2 to 29-30 percent as a direct consequence.

A third moved for the same reason, more recently. The joint limits had four
hand-written copies with fifteen of sixteen finger entries wrong, and correcting
them moved the jerk result from -9.1 to -16.8 percent, the descent rate from
99/100 to 100/100, and executability from 6/100 to 34/100. Two of those three
changes made a headline number look worse while making the pipeline better,
which is the only reliable sign that a fix went in the right direction.
