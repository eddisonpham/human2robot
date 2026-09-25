# CLAUDE.md

Guidance for coding agents working in this repository.

## Project

Human2Robot (formerly DynHand): demonstration-guided SAC for dexterous
manipulation in MuJoCo, plus a C++ trajectory-optimization subsystem for
improving retargeted human demonstrations (spec in
`docs/HANDOFF_RESPONSE.md`). The build specification lives in `agents/`
(read-only). `agents/00_INDEX.md` is the entry point and defines the
reading order. Phase order and acceptance tests are in
`agents/09_PHASE_PLAN.md`; do not start phase N+1 before phase N's
acceptance test passes. `agents/12_NON_GOALS_AND_CUT_SCOPE.md` lists what
must not be built; treat it as a guardrail against scope creep.

## Current state (2026-09-25)

Renamed from dynhand to human2robot: package is `src/human2robot/`, entry
points are `human2robot-*`, Tier B env id is `Human2Robot-AllegroPickup-v0`.
Legacy `results/` directories still use old experiment ids; that is fine.

Python phases (see `docs/HANDOFF.md`): 0-2 done, 3 done inside dynamics
tests, 4/5 scaffolded but unvalidated (no real retargeted demos yet;
`data/processed` empty; dex-retargeting cannot install on Windows),
6 done, 7/8 not started, 9 scaffolded, 10 partially (ONNX export done,
Rust server not started). `results/tier_a_relocate_demo_seed0` has a
stale 883k-step run; treat its "running" status as stale.

C++ subsystem (active work): `cpp/` builds a standalone C++20 library
`h2r_traj` plus pybind11 module `h2r_traj` bindings in `src/human2robot/cpp_bindings/`.
Toolchain on this machine: clang++ 22 (MSVC target) via CMake, no g++ or
MSVC cl. Use the generator and flags recorded in `cpp/README.md`.
C++ phases B-G follow `docs/HANDOFF_RESPONSE.md` sections 14, 17, 18, 20.

## Environment

- Windows, bash shell (Git Bash). Use POSIX syntax in commands.
- Python 3.11 pinned via `.python-version`; manage everything through `uv`
  (`uv sync`, `uv run ...`). Never use the system miniconda directly.
- PyTorch must come from the cu128 index (see `pyproject.toml`); the GPU is
  Blackwell sm_120 and older builds fail or silently JIT-compile.
- Rust stable via rustup, only needed for `rust/` (Phase 10 onward).
  If `cargo` is missing from PATH, it lives in `%USERPROFILE%\.cargo\bin`.
- C++: clang++ only (no g++, no MSVC cl). CMake and ctest exist under the
  miniconda install and are on PATH. GoogleTest and Eigen are fetched by
  CMake FetchContent at configure time; no vcpkg or conan.

## Commands

```bash
uv sync                                   # install/sync environment
uv run pytest                             # full test suite (coverage-gated)
uv run pytest tests/python/test_rl.py -v  # one file
uv run ruff check . && uv run ruff format .   # lint + format before committing
uv run human2robot-train --config configs/tier_a_relocate.yaml --seed 0
```

C++ library (configure once, then build with all cores):

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release -j 16
cd cpp/build && ctest --output-on-failure
```

## Conventions

- Code layout: `src/human2robot/` package. The spec's `python/` paths map here
  (`python/rl/sac.py` is `src/human2robot/rl/sac.py`).
- No comments except short docstrings on public functions and classes.
  No emoji, no em dashes in any committed text file. The same rules apply
  to C++ sources: no comments beyond short public API doc comments.
- Type hints everywhere; all hyperparameters come from validated configs,
  never hardcoded in trainers or environments.
- Module boundaries: `envs/` owns reward and observation; `dynamics/`
  never touches the optimizer; `rl/` contains no condition-specific
  branches, the five ablation conditions differ only by config.
- Every new functional module gets unit tests in the same change: Python
  modules in `tests/python/` (coverage enforced at 90 percent by pytest
  config), C++ modules in `cpp/tests/` run by ctest.
- Seeds: one top-level seed drives random, numpy, torch, and each vector
  env worker gets `seed + worker_index`, never the same seed repeated.
- Results are never committed: `data/` and `results/` are gitignored.
  Each run writes config, metrics, git commit, and system info into
  `results/<experiment_id>/`.
- Reference repos are cloned into `references/` (gitignored) by
  `scripts/setup_references.sh`. Read them there; never vendor their code.

## Verification before done

Run `uv run pytest` and `uv run ruff check .` and make both pass before
declaring any change complete. For training changes, additionally run the
short smoke config `configs/smoke.yaml` end to end.
