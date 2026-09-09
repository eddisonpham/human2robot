# CLAUDE.md

Guidance for coding agents working in this repository.

## Project

DynHand: demonstration-guided SAC for dexterous manipulation in MuJoCo.
The build specification lives in `agents/` (read-only). `agents/00_INDEX.md`
is the entry point and defines the reading order. Phase order and
acceptance tests are in `agents/09_PHASE_PLAN.md`; do not start phase N+1
before phase N's acceptance test passes. `agents/12_NON_GOALS_AND_CUT_SCOPE.md`
lists what must not be built; treat it as a guardrail against scope creep.

## Environment

- Windows, bash shell (Git Bash). Use POSIX syntax in commands.
- Python 3.11 pinned via `.python-version`; manage everything through `uv`
  (`uv sync`, `uv run ...`). Never use the system miniconda directly.
- PyTorch must come from the cu128 index (see `pyproject.toml`); the GPU is
  Blackwell sm_120 and older builds fail or silently JIT-compile.
- Rust stable via rustup, only needed for `rust/` (Phase 10 onward).
  If `cargo` is missing from PATH, it lives in `%USERPROFILE%\.cargo\bin`.

## Commands

```bash
uv sync                                   # install/sync environment
uv run pytest                             # full test suite (coverage-gated)
uv run pytest tests/python/test_rl.py -v  # one file
uv run ruff check . && uv run ruff format .   # lint + format before committing
uv run dynhand-train --config configs/tier_a_relocate.yaml --seed 0
```

## Conventions

- Code layout: `src/dynhand/` package. The spec's `python/` paths map here
  (`python/rl/sac.py` is `src/dynhand/rl/sac.py`).
- No comments except short docstrings on public functions and classes.
  No emoji, no em dashes in any committed text file.
- Type hints everywhere; all hyperparameters come from validated configs,
  never hardcoded in trainers or environments.
- Module boundaries: `envs/` owns reward and observation; `dynamics/`
  never touches the optimizer; `rl/` contains no condition-specific
  branches, the five ablation conditions differ only by config.
- Every new functional module gets unit tests in `tests/python/` in the
  same change. Coverage is enforced at 90 percent by pytest config.
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
