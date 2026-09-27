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
  projection. **57 GoogleTest cases pass** (verified: 57/57 in 1.45s). Exposed to
  Python through pybind11 (`src/human2robot/cpp_bindings/`).
- **Measured on real data** (`results/trajectory_optimization/`): optimizing the
  100 real DexYCB trajectories cut jerk **35.5 percent** and the smoothness cost
  **56.3 percent**, with 85 of 100 sequences converged. Worst-case imitation
  error more than halved (0.180 to 0.082).
- **Measured downstream on the synthetic set**, where both arms are already at
  the 20 ms control period: held-out behavior-cloning error fell **51.6 percent**
  (MSE 7.08e-4 -> 3.43e-4, max error 0.120 -> 0.069) over 10,088 transitions.
- **On real data the optimizer's effect on average imitation error is about 5
  percent.** A naive raw-versus-optimized comparison suggests 66.5 percent, but
  DexYCB captures at 30 Hz and the environment runs at 20 ms, so resampling to
  the control rate accounts for 64.7 of those points. A resampled control arm
  isolates the optimizer. Lead with the controlled number.
- Coverage: 7 test files for this subsystem alone, including MuJoCo open-loop
  replay of the optimized demos (`test_mujoco_replay_optimized.py`).

### Caveats to state if probed

- Velocity and acceleration "improvements" are mostly **constraint saturation**:
  optimized `max_velocity` is 2.0000000000000018 with standard deviation 4e-16,
  so the limit is binding and being clipped. The unbounded quantities (jerk,
  smoothness) are the honest wins.
- **85 of 100** real sequences converge; the synthetic set converges 100/100.
- The BC metric measures how *learnable* the optimized trajectories are, not
  task success.
- The synthetic and real data sets disagree (51.6 percent versus 5.2 percent
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

Three conclusions in this project were **artifacts, not methods**, each found by
reading code or checking ground truth rather than by watching a metric:

1. A singular rotation conversion (division by `2*sin(angle)`) produced
   observations up to 6.35e8 and critic loss to 1.9e15. Fixing it removed the
   earlier headline that demos beat from-scratch SAC by 48 percent.
2. The hand-written physics model disagreed with its simulator by a median 12.9
   per step, in a file with 13 percent test coverage.
3. Resume-induced duplicate metric records silently corrupted 6 of 12 runs.

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
  env id `Human2Robot-AllegroPickup-v0`). 270 tests, 95.09 percent coverage.
- Tests: 126 -> 270 Python cases; 57 C++ cases. Gate: `ruff check`,
  `ruff format --check`, `pytest`.
- Not started: Rust inference server, real DexYCB subject-02+ beyond the current
  set, Shadow Hand stretch work.
