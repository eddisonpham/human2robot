# Human2Robot resume entry

Status as of 2026-09-27. The deliverable is the **human-to-robot trajectory
conversion pipeline**, validated across two subjects. The SAC ablation is a
secondary consumer of the output and returned a null result.

## Lead with the conversion pipeline

- **Real DexYCB ingestion, two subjects.** `src/human2robot/data/dexycb.py` loads
  DexYCB hand motion, extracts MANO poses, and retargets them to 22-DoF Allegro
  joint targets. 100 sequences from each of two subjects, plus an IK-based
  variant (`test_dexycb_ik_pipeline.py`) as an independent path to the same
  targets. Invocable as `python -m human2robot.data.dexycb`.
- **C++20 constrained trajectory optimizer.** `cpp/` (library `h2r_traj`):
  Hermite interpolation, finite differences, moving-window smoothing, projection
  onto joint/velocity/acceleration limits, weighted cost (tracking, velocity,
  acceleration, jerk, collision, limits), and a projected optimizer with
  backtracking line search. **64 GoogleTest cases pass.** Exposed to Python
  through pybind11 (`src/human2robot/cpp_bindings/`).
- **Validated across subjects, with hyperparameters selected on one.** Sweeping
  `step_size` on 50 subject-01 sequences and selecting from that half alone, the
  chosen setting improves **89 of 100 subject-02 sequences**, a different
  person's hand it has never seen, with a median 12.6 percent cost reduction.
  Within subject-01 the held-out half gives 47/50 at 18.0 percent. On
  subject-01 the optimizer also cuts jerk **41.9 percent** and smoothness cost
  **64.7 percent**.
- **Controlled downstream result.** A resampled pre-optimization control arm
  isolates the optimizer from the 30 Hz to 20 ms resampling, which otherwise
  accounts for most of the apparent gain. Against that control it cuts held-out
  behavior-cloning error **26.0 percent on subject-02** (MSE 1.63e-4 to 1.21e-4)
  and **31.0 percent on subject-01**, with worst-case error on subject-01 falling
  from 0.291 to 0.065. Naive raw-versus-optimized would claim 69 to 76 percent.
- **Fixed a search that was doing nothing.** The optimizer perturbed every
  timestep with independent Gaussian noise, which raises jerk faster than the
  tracking pull lowers it, so the first candidate always tripped the abort guard
  and the search ended with exactly zero improvement on 85 percent of sequences.
  Adding a backtracking line search and making the noise scale configurable
  took the descent rate to 89/100 on an unseen subject and the controlled
  imitation figure from 5 to 26-31 percent.
- Coverage: 7 test files for this subsystem alone, including MuJoCo open-loop
  replay of the optimized demos (`test_mujoco_replay_optimized.py`).

### Caveats to state if probed

- Velocity "improvements" are mostly **constraint saturation**: optimized
  `max_velocity` is 2.0000000000000018 with standard deviation 4e-16, so the
  limit is binding and being clipped. The unbounded quantities (jerk,
  smoothness) are the honest wins.
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
- `noise_scale = 0` is a structural argument, not a tuned value: independent
  per-timestep noise provably raises jerk faster than the tracking pull lowers
  it. `step_size = 0.5` *is* a tuning choice, but it sits on a flat plateau
  where 0.35, 0.5, and 1.0 give the same result on the tuning half.
- Only 2 of the 10 DexYCB subjects are processed, so generalization is
  demonstrated across two people, not ten. This is now a cross-subject split
  rather than a within-subject one, which is the strongest form of the claim
  this data allows, but two is still two.
- Subject-02 is a second subject, not a second *dataset*. Both come from the
  same capture rig, lighting, and task, so this does not test robustness to
  capture conditions.
- The BC metric measures how *learnable* the optimized trajectories are, not
  task success.
- Real human motion is harder for the optimizer than synthetic demonstrations
  (26-31 percent versus 57 percent on mean error). Input roughness is the likely
  cause and it is untested.

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

Five conclusions in this project were **artifacts, not methods**, each found by
reading code or checking a control rather than by watching a metric:

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
   per step, in a file with 13 percent test coverage.Also: resume-induced duplicate metric records silently corrupted 6 of 12 runs,
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
  env id `Human2Robot-AllegroPickup-v0`). 342 tests, 96.98 percent coverage.
- Tests: 126 -> 342 Python cases; 57 -> 64 C++ cases. Gate: `ruff check`,
  `ruff format --check`, `pytest`. There is no CI, so the gate is manual.
- Not started: Rust inference server, the 8 remaining DexYCB subjects, Shadow
  Hand stretch work.
