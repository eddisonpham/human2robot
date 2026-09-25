# Human2Robot progress handoff (as of 2026-09-25, updated after C++ phases B-G)

Addendum: the repository was renamed dynhand -> human2robot (package,
entry points, env id `Human2Robot-AllegroPickup-v0`). The C++ trajectory
subsystem under `cpp/` is built and validated (57/57 ctest, pybind11
module `h2r_cpp`), the optimization pipeline runs over 100 synthetic
demos with 100/100 convergence, and MuJoCo replay of optimized demos
passes. Metrics and comparisons live in
`results/trajectory_optimization/report.json`. Real DexYCB downloads
(subject-01 + subject-02 + models + calibration) are in progress under
`data/raw/dexycb/` via `scripts/download_dexycb.sh` (Google Drive, gdown).
See `cpp/README.md` for toolchain specifics (clang++ targeting MSVC,
Ninja from the venv).

For the next coding agent. High level only; read `agents/00_INDEX.md` and
`agents/09_PHASE_PLAN.md` for the spec. Acceptance tests there define "done".

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 0 | Env verification, reference repos, NOTES.md | Done (commit e48a75b) |
| 1 | Tier A SAC on Adroit relocate, demo seeding, SB3 cross-check | Done, validated (probes + SB3 check) |
| 2 | Tier B floating-base Allegro MJCF, touch sensors | Done, sanity tests pass (346fbe3) |
| 3 | Inverse dynamics validation, computed-torque baseline | Present in dynamics module tests |
| 4/5 | Demo pipeline: DexYCB loaders, retargeting, a_demo labels | Code scaffolded (data/), retargeted replay validation not confirmed |
| 6 | Dynamics ensemble (black-box + residual), SAC integration | Done, independently validated, wired into training (c6ba07d, 60e0d1d) |
| 7/8 | Tier B RL conditions A-E, ablation matrix | Not started; only Condition A baseline configs exist |
| 9 | Robustness and generalization eval | Helpers scaffolded, runs not done |
| 10 | ONNX export, Rust inference server | ONNX export + parity validation done early (b734cd9); Rust server not started |
| 11 | Stretch (Shadow Hand, HaMeR live, Rust stepper) | Not started |

Per `09_PHASE_PLAN.md`: Phase 8 (five-condition ablation matrix) is the
primary deliverable and is the next real work item after Phase 4/5
validation is confirmed.

## Metrics observed (results/, gitignored)

- Tier A relocate, demo-seeded, seed 0 (`results/tier_a_relocate_demo_seed0`):
  883k steps logged; recent eval returns noisy, roughly 9 to 21
  (last: 20.6 at step 880k). `run_status.json` says "running" but was last
  updated 2026-09-12; treat as stale, check the PID before trusting it.
- Phase 1 probes at 5k steps: Condition A 7.05, Condition B 19.05,
  SB3 SAC 8.41. B ahead of A and SB3, consistent with Phase 1 acceptance
  (demo integration helps).
- `results/bench`: baseline Condition A config on AdroitHandRelocate-v1,
  dynamics and demos disabled; essentially cold (2k steps).
- Smoke and adroit probe runs exist and completed.

## Infrastructure in place

- SAC trainer with resume, sequential training queue, status script,
  learning-curve comparison CLI, process-safe run ownership.
- Config system (validated YAML configs under `configs/`), experiment
  artifacts: config, git commit, system info, metrics.jsonl, TB.
- 22 test files under tests/python; 90 percent coverage gate configured.
  A full `uv run pytest` takes minutes; run it, do not assume it passes.
- ONNX actor export with numerical parity validation.

## Conventions the next agent must keep

- Everything through `uv`; never system miniconda. PyTorch from cu128 index.
- Phase N+1 only after Phase N acceptance passes (09_PHASE_PLAN.md).
- Hyperparameters only from validated configs; no condition-specific
  branches in rl/; envs/ owns reward and observation.
- One top-level seed; vector env workers get seed + worker_index.
- No comments except short public docstrings; no emoji; no em dashes.
- New functional modules get tests in the same change; never commit
  data/ or results/.
- Before done: `uv run pytest`, `uv run ruff check .`, and a
  `configs/smoke.yaml` run for training changes.
