# Human2Robot resume entry

Status as of 2026-09-27. The deliverable is the **human-to-robot manipulation
pipeline**; the SAC ablation is a secondary validation and came back null.

## Lead with the conversion pipeline

- **Real DexYCB ingestion.** `src/human2robot/data/dexycb.py` loads DexYCB hand
  motion, extracts MANO poses, and retargets them to 22-DoF Allegro joint
  targets. 100 real sequences produced, plus an IK-based variant
  (`test_dexycb_ik_pipeline.py`) as an independent path to the same targets.
- **C++20 constrained trajectory optimizer.** `cpp/` (library `h2r_traj`):
  Hermite interpolation, finite differences, moving-window smoothing, projection
  onto joint/velocity/acceleration limits, weighted cost (tracking, velocity,
  acceleration, jerk, collision, limits), and a seeded stochastic optimizer with
  projection. **60 GoogleTest cases pass** (verified: 60/60 in 1.36s). Exposed to
  Python through pybind11 (`src/human2robot/cpp_bindings/`).
- **Measured on real data** (`results/trajectory_optimization/`): optimizing the
  100 real DexYCB trajectories cut jerk **41.9 percent** and the smoothness cost
  **64.7 percent**. Worst-case imitation error fell from 0.291 to **0.065**.
- **Controlled downstream result on real data.** A resampled pre-optimization
  control arm isolates the optimizer from the 30 Hz to 20 ms resampling, which
  otherwise accounts for most of the apparent gain. Against that control the
  optimizer cuts held-out behavior-cloning error **31 percent** (MSE 2.08e-4 to
  1.44e-4). Naive raw-versus-optimized would claim 75.6 percent.
- **On the synthetic set**, where both arms are already at 20 ms so the
  comparison is clean, held-out error fell **56.8 percent** (MSE 7.08e-4 to
  3.06e-4, max error 0.120 to 0.065) over 10,088 transitions.
- **Fixed a search that was doing nothing.** The optimizer perturbed every
  timestep with independent Gaussian noise, which raises jerk faster than the
  tracking pull lowers it, so the first candidate always tripped the abort guard
  and the search ended with exactly zero improvement. Adding a backtracking line
  search and making the noise scale configurable took the descent rate from
  24/100 to **94/100** real sequences with a median 16.8 percent cost reduction,
  and the controlled imitation figure from 5 to 31 percent.
- Coverage: 7 test files for this subsystem alone, including MuJoCo open-loop
  replay of the optimized demos (`test_mujoco_replay_optimized.py`).

### Caveats to state if probed

- Velocity "improvements" are mostly **constraint saturation**: optimized
  `max_velocity` is 2.0000000000000018 with standard deviation 4e-16, so the
  limit is binding and being clipped. The unbounded quantities (jerk,
  smoothness) are the honest wins.
- The convergence figure was twice wrong. It first reported 85/100 real and
  100/100 synthetic because it compared the final cost against the unprojected
  input, so projection alone satisfied the test. Correcting that exposed the
  search achieving **exactly zero improvement on 85 percent of sequences**. The
  current rate is **94/100** real and **93/100** synthetic, with a median 16.8
  percent cost reduction.
- `noise_scale` and `step_size` were tuned on the same 100 sequences the
  convergence rate is reported on, so **94/100 is a training-set figure** with
  no held-out confirmation. Say so if asked.
- The BC metric measures how *learnable* the optimized trajectories are, not
  task success.
- The synthetic and real data sets disagree (56.8 percent versus 31.0 percent
  on mean error). If asked why, the honest answer is that input roughness is the
  likely cause and that it is untested.

## Secondary: the RL ablation (null result, reported as such)

Tier A is a correctness gate on `AdroitHandRelocate-v1`. Tier B is the
5-condition ablation on the floating Allegro hand. Three conditions are complete
at 2,000,004 steps, three seeds each, and **none separates from the others**:

| Condition | Mean eval return |
| --- | --- |
| A from-scratch SAC | -49.2 +/- 4.9 |
| B BC-init + demo replay | -51.0 +/- 1.6 |
| C blackbox dynamics augmentation | -46.9 +/- 6.9 |

No run reaches a positive return, so the task is unsolved and the experiment
lacked the signal to separate conditions. D and E cannot be run meaningfully at
all: the nominal physics model is the simulator itself, so the residual they
would learn is identically zero. Spec-required domain randomization is not
implemented.

## Secondary: experimental-rigor work worth mentioning

Five conclusions in this project were **artifacts, not methods**, each found by
reading code or checking a control rather than by watching a metric:

1. A singular rotation conversion (division by `2*sin(angle)`) produced
   observations up to 6.35e8 and critic loss to 1.9e15. Fixing it removed the
   earlier headline that demos beat from-scratch SAC by 48 percent.
2. The hand-written physics model disagreed with its simulator by a median 12.9
   per step, in a file with 13 percent test coverage.
3. Resume-induced duplicate metric records silently corrupted 6 of 12 runs.
4. A 66.5 percent imitation-error gain on real data turned out to be 64.7
   points of resampling and 5.2 points of optimizer, once a matched control arm
   was added.
5. The optimizer's convergence flag reported 85/100 because it compared against
   the unprojected input, counting projection as optimization. Correcting it
   exposed that the search achieved zero improvement on 85 percent of sequences;
   removing the per-timestep noise took it to 94/100 with a median 16.8 percent
   cost reduction.

An intermittent multi-hour training hang was traced with a native stack dump
(`py-spy --native`) to leaked CUDA contexts wedging a synchronizing `.item()`
call. Also built: metrics-stream integrity auditing, lossless checkpoint/resume
(weights, replay buffer, dynamics ensemble, RNG; zero rewind), and invariant tests
in `test_invariants.py` pinning exactly the conventions whose violations caused
the failures above.

## Toolchain notes (Windows machine)

- clang++ 22.1.6 targeting MSVC (BuildTools 14.50), Ninja via the project venv,
  CMake 4.1.3. Exact configure command in `cpp/README.md`. Set
  `PYTHON_EXECUTABLE` to the venv python or the pybind11 module silently targets
  the wrong interpreter.
- GPU: RTX 5060 8GB, sm_120, torch cu128. CPU: 24 cores. Training defaults to
  CPU deliberately; see `docs/FINDINGS_training_hang.md`.

## State

- Python: `src/human2robot/` (renamed from dynhand; entry points `human2robot-*`,
  env id `Human2Robot-AllegroPickup-v0`). 338 tests, 96.97 percent coverage.
- Tests: 126 -> 338 Python cases; 57 -> 60 C++ cases. Gate: `ruff check`,
  `ruff format --check`, `pytest`. There is no CI, so the gate is manual.
- Not started: Rust inference server, real DexYCB subject-02+ beyond the current
  set, Shadow Hand stretch work.
