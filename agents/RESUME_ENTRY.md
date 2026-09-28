# Human2Robot resume entry

Status as of 2026-09-28. The deliverable is the **human-to-robot trajectory
conversion pipeline**, validated across two subjects, and now measured against
the only question that matters about it: can the simulated hand perform the
motion. The SAC ablation is a secondary consumer of the output and returned a
null result.

## Lead with the conversion pipeline

- **Real DexYCB ingestion and IK retargeting, two subjects.**
  `scripts/retarget_dexycb_ik.py` runs DexPilot vector retargeting of MANO
  keypoints onto the 22-DoF Allegro, 100 sequences per subject.
- **C++20 constrained trajectory optimizer.** `cpp/` (library `h2r_traj`):
  Hermite interpolation, finite differences, moving-window smoothing, projection
  onto joint/velocity/acceleration limits, weighted cost (tracking, velocity,
  acceleration, jerk, collision, limits), and a projected optimizer with
  backtracking line search. **64 GoogleTest cases pass.** Exposed to Python
  through pybind11 (`src/human2robot/cpp_bindings/`).
- **The pipeline makes the motion executable, which is the point.**
  `scripts/score_feasibility.py` replays every trajectory open-loop in MuJoCo
  and measures how far the hand actually tracks. Optimizing takes subject-01
  from **6 of 100 executable to 34 of 100**, and subject-02 from 0 to 8, while
  synthetic data is 100/100 before any processing. Of the subject-01 gain, 6 to
  14 is resampling to the control rate and the rest is the optimizer. Limit
  violation is 0.0000 on every arm: the trajectories respect the actuator
  bounds, and what remains is a tracking failure, not a reach failure. This is
  the strongest number in the project because it is the only one that is a
  property of the robot rather than of the trajectory.
- **Validated across subjects, with hyperparameters selected on one.** Sweeping
  `step_size` on 50 subject-01 sequences and selecting from that half alone, the
  chosen setting improves **100 of 100 subject-01 sequences and 100 of 100
  subject-02 sequences**, a different person's hand it has never seen, for a
  median cost reduction of 36 and 43 percent. It cuts smoothness cost about
  **30 percent** and max velocity about **72 percent** on both subjects.
- **Controlled downstream result.** A resampled pre-optimization control arm
  isolates the optimizer from the 30 Hz to 20 ms resampling, which otherwise
  accounts for most of the apparent gain; naive raw-versus-optimized would claim
  78 to 83 percent. Against that control the optimizer cuts held-out
  behavior-cloning error **29.5 percent on subject-01 and 29.0 percent on the
  held-out subject-02**, and **57 percent** on worst-case error. Holding out
  whole *trajectories* rather than random transitions, the harder test, gives
  **+25.2 and +33.3 percent, positive on every seed of 5**.
- **Fixed a search that was doing nothing.** The optimizer perturbed every
  timestep with independent Gaussian noise, which raises jerk faster than the
  tracking pull lowers it, so the first candidate always tripped the abort guard
  and the search ended with exactly zero improvement on 85 percent of sequences.
  Adding a backtracking line search and making the noise scale configurable
  took the descent rate to 100/100 on an unseen subject.
- **Every published number is pinned to its artifact.**
  `tests/python/test_published_results.py` reads 50 figures from
  `results/trajectory_optimization/*.json` via `evaluation/published.py` and
  asserts the exact strings that appear in `README.md` and `docs/RESULTS.md`.
  A committed snapshot in `docs/published_figures.json` keeps the check running
  on a fresh clone, where `results/` is absent.

### Caveats to state if probed

- Velocity "improvements" are mostly **constraint saturation**: optimized
  `max_velocity` is 2.0000000000000018 with standard deviation 4e-16, so the
  limit is binding and being clipped. The unbounded quantity that moves is
  smoothness cost, about 30 percent.
- **The jerk result was inflated and I would not quote the old number.** It was
  -41.9 percent until I found that the retargeter feeding it collapsed 16 joint
  dimensions into 1, carrying 90 percent of its variance in a single direction.
  It then read -9.1 percent until the joint limits were corrected, and is
  **-16.8 percent** on real DexPilot IK retargets today. The reason a smoothness
  objective looked so effective on degenerate data is that a one-dimensional
  curl has nothing but jitter to remove. This is the finding I would lead with,
  because the number went down twice and the underlying effect got stronger both
  times.
- **The joint limits were wrong in four hand-written copies at once**, with 15
  of 16 finger entries disagreeing with the MuJoCo model, and the model has 16
  actuators rather than the 22 the code assumed. Optimized trajectories were
  violating the real limits by 0.75 rad. Correcting them took violation to
  0.0000, executability from 6/100 to 34/100, and the jerk result *down* from
  -9.1 to -16.8 percent, because the optimizer had been partly succeeding by
  clamping. See `docs/FINDINGS_joint_limits.md`.
- The convergence figure was wrong twice. It first reported 85/100 real and
  100/100 synthetic because it compared the final cost against the unprojected
  input, so projection alone satisfied the test. Correcting that exposed the
  search achieving **exactly zero improvement on 85 percent of sequences**.
- The first hyperparameter protocol was also wrong: the `step_size` sweep ran on
  all 100 sequences and the rate was reported on those same 100, which is
  choosing hyperparameters on the evaluation set. Replaced with a
  tune-on-subject-01, report-on-subject-02 protocol
  (`scripts/validate_optimizer_split.py`). If asked, the answer is the sweep was
  contaminated and was redone. The contaminated 94/100 is not quoted anywhere.
- The split-granularity table used to quote three-seed error bars for an
  experiment that had only ever been run at **one seed**, and the script wrote
  every data set to the same file so the last run silently overwrote the others.
  It now runs 5 seeds and writes a file per set.
- `noise_scale = 0` is a structural argument, not a tuned value: independent
  per-timestep noise provably raises jerk faster than the tracking pull lowers
  it. `step_size = 0.35` *is* a tuning choice, but it is selected from the
  tuning half alone and the held-out half and held-out subject agree with it.
- **The tail-extrapolation reversal is explained, not open.** The prefix split
  gives -54.2 and -104.9 percent on the two real subjects against +40.6 percent
  on synthetic, and the cause is that real hand motion decelerates into a stop.
  The held-out tail is nearly motionless, and the control arm scores 1.45e-4
  against a do-nothing baseline of 1.47e-4, so the fitted model does no better
  than predicting that the hand stopped. A split whose answer is near zero
  rewards whichever arm stops hardest. The optimizer redistributes the
  deceleration rather than removing it, so it loses a test it cannot win. On
  synthetic there is no deceleration, the tail genuinely moves, and the same
  split is positive. Tail extrapolation is **untested** by these data rather than
  failed by the optimizer; measuring it properly needs sequences whose tails
  contain sustained motion, and this project has none.
- Only 2 of the 10 DexYCB subjects are processed, so generalization is
  demonstrated across two people, not ten. This is now a cross-subject split
  rather than a within-subject one, which is the strongest form of the claim
  this data allows, but two is still two.
- Subject-02 is a second subject, not a second *dataset*. Both come from the
  same capture rig, lighting, and task, so this does not test robustness to
  capture conditions.
- The BC metric measures how *learnable* the optimized trajectories are, not
  task success. The feasibility score measures executability, not whether the
  hand grasps the object; neither measures task reward, and no result here does.
- Executability is measured against a simulation, not hardware. A tracking drift
  of 0.4 to 0.55 rad on real motion is large in absolute terms: **even the
  optimized arm is only a third executable on subject-01**, so the pipeline
  improves a real problem without solving it.

## Secondary: the RL ablation (a null result, reported as one)

The converted trajectories feed a SAC trainer. The question was whether human
demonstrations and physics priors help SAC control a robot hand. Three ablation
conditions at 2,000,004 steps, three seeds each, **none separates from the
others**:

| Condition | Mean eval return |
| --- | --- |
| A from-scratch SAC | -49.2 +/- 4.9 |
| B BC-init + demo replay | -51.0 +/- 1.6 |
| C blackbox dynamics augmentation | -46.9 +/- 6.9 |

No run reaches a positive return, so the task is unsolved and the experiment
never had the signal to answer the question. D and E cannot be run meaningfully
at all: the nominal physics model is the simulator itself, so the residual they
would learn is identically zero. Spec-required domain randomization is not
implemented.

Present this as a rigorous null, not a success. It is the weakest part of the
project and belongs in one paragraph, not the lead.

## Secondary: experimental-rigor work worth mentioning

Nine conclusions in this project were **artifacts, not methods**, each found by
reading code, checking a control, or checking the data rather than by watching a
metric:

1. A 66.5 percent imitation-error gain on real data turned out to be 64.7
   points of resampling and 5.2 points of optimizer, once a matched control arm
   was added.
2. The optimizer's convergence flag reported 85/100 because it compared against
   the unprojected input, counting projection as optimization. Correcting it
   exposed that the search achieved zero improvement on 85 percent of sequences.
3. The `step_size` sweep was originally run on the same sequences its result was
   reported on. Replaced with a tune-on-one-subject, report-on-another protocol.
4. A singular rotation conversion in the RL environment (division by
   `2*sin(angle)`) produced observations up to 6.35e8 and critic loss to 1.9e15.
   Fixing it removed the earlier headline that demos beat from-scratch SAC by
   48 percent.
5. The hand-written physics model disagreed with its simulator by a median 12.9
   per step, in a file with 13 percent test coverage.
6. The mean-imitation-error gain held out random pooled transitions, so each
   holdout item sat one step from a training item. This one was partly an
   artifact of the data: on real retargeted motion the whole-trajectory test
   passes comfortably.
7. **Every trajectory result was measured on a retargeter that collapsed 16
   joint dimensions into 1**, while a real DexPilot IK path sat unused in the
   repo. The informative part is that every prior check was a relative
   comparison against the same degenerate input, so all of them correctly
   reported a real effect on the wrong data. What was missing was a check on
   the input itself. Demo-set rank is now pinned in `test_invariants.py`.
8. **The joint limits existed in four hand-written copies with no test against
   the robot**, 15 of 16 finger entries wrong, and 6 phantom actuators. Every
   published limit-compliance claim was about bounds the machine does not have.
9. **Two "evaluations" never ran the simulator.** One named for MuJoCo replay
   compared arrays against stub bounds of `zeros(22)` and `ones(22)`; the
   domain-randomization function randomized nothing. Both are replaced by real
   implementations, and the deleted replay is what `evaluation/feasibility.py`
   now does properly.

Also: resume-induced duplicate metric records silently corrupted 6 of 12 runs,
and an intermittent multi-hour training hang was traced with a native stack dump
(`py-spy --native`) to leaked CUDA contexts wedging a synchronizing `.item()`
call. Built: metrics-stream integrity auditing, lossless checkpoint/resume
(weights, replay buffer, dynamics ensemble, RNG; zero rewind), and invariant
tests in `test_invariants.py` pinning exactly the conventions whose violations
caused the failures above.Also: resume-induced duplicate metric records silently corrupted 6 of 12 runs,
and an intermittent multi-hour training hang was traced with a native stack dump
(`py-spy --native`) to leaked CUDA contexts wedging a synchronizing `.item()`
call. Built: metrics-stream integrity auditing, lossless checkpoint/resume
(weights, replay buffer, dynamics ensemble, RNG; zero rewind), and invariant
tests in `test_invariants.py` pinning exactly the conventions whose violations
caused the failures above.

## Toolchain notes (Windows machine)

- clang++ 22.1.6 targeting MSVC (BuildTools 14.50), Ninja via the project venv,
  CMake 4.1.3. Exact configure command in `cpp/README.md`. Set
  `PYTHON_EXECUTABLE` to the venv python or the pybind11 module silently targets
  the wrong interpreter.
- GPU: RTX 5060 8GB, sm_120, torch cu128. CPU: 24 cores. Training defaults to
  CPU deliberately; see `docs/FINDINGS_training_hang.md`.

## State

- Python: `src/human2robot/` (renamed from dynhand; entry points `human2robot-*`,
  env id `Human2Robot-AllegroPickup-v0`). Over 400 tests, 96 percent coverage.
- Tests: 126 -> 400+ Python cases; 57 -> 64 C++ cases. Gate: `ruff check`,
  `ruff format --check`, `pytest`, `ctest`. There is no CI, so the gate is
  manual.
- `rust/inference_server/` holds 121 lines of axum + ONNX Runtime code that
  exists only on this machine: it has an accidental `git init` with zero
  commits and no remote, so folding it into this repository requires deleting
  that directory first. Not done, because removing a git repository is not an
  agent's call to make.
- Not started: the 8 remaining DexYCB subjects, Shadow Hand stretch work.
