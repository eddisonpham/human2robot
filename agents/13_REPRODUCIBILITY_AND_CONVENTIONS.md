# 13 — Reproducibility and Conventions

## 1. Config schema (pydantic + YAML, kept from original plan's yaml-config instinct, made schema-validated)

Every run is driven by one YAML file validated against a `pydantic`
model, so a typo'd hyperparameter fails at load time, not silently three
hours into a training run.

```python
# python/config/schema.py — sketch
class PhysicsConfig(BaseModel):
    timestep: float = 0.002
    control_decimation: int = 10


class SACConfig(BaseModel):
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 1_000_000
    warmup_steps: int = 10_000
    utd_ratio: int = 1


class DemoConfig(BaseModel):
    enabled: bool = False
    bc_epochs: int = 100
    demo_ratio_start: float = 0.5
    demo_ratio_anneal_steps: int = 300_000


class DynamicsAugConfig(BaseModel):
    enabled: bool = False
    mode: Literal["blackbox", "residual"] = "blackbox"
    ensemble_size: int = 5
    rollout_horizon: int = 1
    synthetic_ratio: float = 0.5
    retrain_every_steps: int = 250


class ExperimentConfig(BaseModel):
    experiment_id: str
    seed: int
    condition: Literal["A", "B", "C", "D", "E"]
    physics: PhysicsConfig = PhysicsConfig()
    sac: SACConfig = SACConfig()
    demo: DemoConfig = DemoConfig()
    dynamics_aug: DynamicsAugConfig = DynamicsAugConfig()
    total_env_steps: int = 1_000_000
```

`condition` is a single field that sets `demo.enabled` and
`dynamics_aug.*` consistently, per the table in
`05_RL_ALGORITHM_SPEC.md` §4 — the mapping lives in one place
(`python/config/conditions.py`) so the five conditions can never silently
drift out of sync with that table.

## 2. Seeds

- One top-level `seed` per run seeds: Python's `random`, `numpy`,
  `torch` (CPU and CUDA), the environment's own RNG, and the parallel
  worker RNGs (each worker gets `seed + worker_index`, not the same seed
  repeated — a common, easy-to-miss bug that silently correlates
  "parallel" environments).
- Ablation runs use ≥3–5 seeds per condition (`10_EXPERIMENTS_AND_EVALUATION.md`);
  never report or plot a single-seed result as if it were a stable
  estimate.

## 3. Git and experiment tracking

- Every experiment logs `git_commit.txt` (full SHA, plus a `git diff`
  if the working tree isn't clean — a run against uncommitted changes is
  still loggable, just flagged as dirty, rather than silently
  unreproducible).
- `results/<experiment_id>/` holds `config.yaml`, `metrics.json`,
  `checkpoint.pt`, `git_commit.txt`, `system_info.json` (hardware, driver
  versions, package versions — `pip freeze` / `cargo tree` snapshot).
- One command reproduces any run:
  `python -m python.rl.train --config results/<experiment_id>/config.yaml --seed <n>`.

## 4. Testing requirements

- `tests/python/`: `pytest`. Minimum coverage: environment sanity checks
  (`06_ROBOT_AND_SIMULATION_SPEC.md` §7), dynamics-validation checks
  (`09_PHASE_PLAN.md` Phase 3), retargeting round-trip check (Phase 5),
  ONNX parity check (`11_PRODUCTIONIZATION.md` §1).
- `tests/rust/`: `cargo test`. Minimum coverage: trajectory-preprocessing
  crate output matches the Python reference implementation on a fixed
  sample input (bit-for-bit or float-tolerance); inference-server parity
  test (`11_PRODUCTIONIZATION.md` §3).
- CI is optional for a solo local project, but structure tests so they
  *could* run in CI (no interactive prompts, no hardcoded absolute paths
  outside `data/`/`results/`, deterministic given a seed).

## 5. Code conventions

- Python: type hints throughout, `ruff`/`black` formatting, docstrings on
  every public function in `python/rl/`, `python/envs/`, and
  `python/dynamics/` at minimum (these are the modules a reviewer or
  future-you will need to understand fastest).
- Rust: `cargo fmt` + `cargo clippy` clean, `unsafe` blocks (if any,
  e.g. inside a `mujoco-rs`/FFI boundary) commented with a justification.
- No condition-specific (`if condition == "E"`) branches scattered through
  the trainer — all five conditions are expressed purely through the
  config schema in §1, per `07_SYSTEM_ARCHITECTURE.md` §4's module
  boundary rule.

## 6. Licensing summary (cross-referenced from files 03 and 08 — collected here for one-stop checking)

| Asset | License | Commercial use |
|---|---|---|
| MuJoCo | Apache-2.0 | Yes |
| MuJoCo Menagerie models | BSD-2 / Apache-2.0 (per model, check each) | Yes |
| Gymnasium / Gymnasium-Robotics / Minari | MIT | Yes |
| dex-retargeting | MIT | Yes |
| CleanRL | MIT | Yes |
| DexYCB dataset | CC BY-NC 4.0 | **No** (research only) |
| ARCTIC / GRAB datasets | check each repo's LICENSE | Verify before any commercial use |
| Minari `D4RL/relocate/*` demos (sourced from original DAPG repo) | verify at Phase 0 | Verify before any commercial use |
| ort / mujoco-rs / rusty_mujoco | MIT/Apache-2.0 (verify per crate) | Yes (pending verification) |

This table is a planning aid, not legal advice — verify current license
text directly for any use beyond a personal portfolio project.
