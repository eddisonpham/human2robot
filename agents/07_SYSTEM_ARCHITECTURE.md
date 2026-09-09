# 07 — System Architecture

## 1. Language division — three scoped, justified tiers

The original plan wanted "Rust for simulation infrastructure... Python for
ML." We keep that spirit but sequence it deliberately so Rust is never on
the critical path for scientific correctness: **prove things work in
Python first, then move the parts that actually benefit from Rust.**

### Tier 1 — Python (build first, this is most of the project)

Everything that touches learning, MuJoCo stepping for the MVP, and the
retargeting call sites: `python/rl/` (SAC + extensions), `python/envs/`
(the Allegro/Tier-A env wrappers), `python/retargeting/` (calls into
`dex-retargeting`), `python/data/` (dataset loading), `python/evaluation/`
(metrics, plots).

Parallel simulation for the MVP uses `gymnasium.vector.AsyncVectorEnv`
with 12–16 subprocess workers — the simplest correct baseline, matching
the hardware budget in `02_TECH_STACK.md`. Benchmark it; if profiling
shows subprocess IPC overhead dominates, try a thread-pool variant that
exploits the fact that MuJoCo's C `mj_step` releases the Python GIL (a
single shared, read-only `mjModel` with one `mjData` per thread — this is
literally the design of the official `mujoco.rollout` module and was
discussed by MuJoCo's own maintainers for exactly this use case). Document
whichever wins with a steps/sec/core number; don't guess.

### Tier 2 — Rust, performance layer (stretch, only after Tier 1 works end-to-end)

`rust/trajectory/`: batch trajectory preprocessing (smoothing,
differentiation, resampling of retargeted demonstration trajectories,
§`08_DATA_AND_RETARGETING_PIPELINE.md`) — a genuinely good fit for Rust
(CPU-bound, embarrassingly parallel, no need for Python's ecosystem here),
exposed to Python via `pyo3`/`maturin` or run as a standalone CLI that
reads/writes the same NPZ/Parquet files as the Python pipeline.

`rust/env_stepper/` (optional, only pursued if Tier 1's Python parallelism
is profiled as a genuine bottleneck): a native environment stepper using
the `mujoco-rs` crate, exposed to Python as a `pyo3` extension module
implementing the same `gymnasium.vector.VectorEnv` interface so it's a
drop-in replacement. **Do not start this before Tier 1 is validated and
profiled** — per the original plan's own "do not force all ML into Rust in
version 1" instruction, generalized here to "do not build a Rust
simulation layer before you have evidence Python's is the bottleneck."

### Tier 3 — Rust, production serving (the "productionizable" deliverable)

`rust/inference_server/`: loads the ONNX-exported trained actor via the
`ort` crate and serves it over a small HTTP/gRPC API for real-time control.
See `11_PRODUCTIONIZATION.md` for the full contract. This is the piece
that makes the project "something you could hand to a company" — a
research training pipeline in Python/PyTorch, and a separate, minimal,
dependency-light, low-latency Rust service that only needs to load one
`.onnx` file to run the policy, with no PyTorch/CUDA dependency at
inference time at all.

## 2. Repo layout

```text
dynhand/
├── agents/                      # this planning spec (read-only reference)
├── references/                  # cloned third-party repos, see 03_EXISTING_REPOS_TO_CLONE.md
├── configs/
│   ├── base.yaml                # shared physics/control constants
│   ├── tier_a_relocate.yaml     # Gymnasium-Robotics benchmark run
│   ├── tier_b_pickup.yaml       # our Allegro pickup env
│   └── ablation_{a,b,c,d,e}.yaml
├── data/
│   ├── raw/                     # untouched dataset downloads (ARCTIC/DexYCB/GRAB)
│   ├── processed/               # smoothed/differentiated trajectories, NPZ
│   └── demonstrations/          # retargeted robot-joint-space demos, ready for BC/replay-seeding
├── envs/
│   └── hamer/                   # isolated venv/conda env, HaMeR only, never imported elsewhere
├── python/
│   ├── data/                    # dataset loaders (ARCTIC/DexYCB/GRAB wrappers)
│   ├── retargeting/             # thin wrapper calling dex-retargeting
│   ├── envs/                    # Tier A wrapper + Tier B Allegro env (gymnasium.Env)
│   ├── dynamics/                # residual/black-box dynamics ensemble (04_MATH_SPEC.md §5)
│   ├── rl/                      # sac.py (forked from CleanRL) + bc.py + mbpo.py extensions
│   ├── export/                  # torch->ONNX export + parity check
│   └── evaluation/              # metrics, plotting
├── rust/
│   ├── trajectory/               # Tier 2 preprocessing crate
│   ├── env_stepper/               # Tier 2 stretch, mujoco-rs based
│   └── inference_server/          # Tier 3 production ONNX server (ort crate)
├── experiments/
│   ├── reach/ dynamics/ grasp/ ablations/
├── tests/
│   ├── python/                   # pytest
│   └── rust/                     # cargo test
└── results/
    ├── configs/ metrics/ checkpoints/ plots/
```

## 3. Data flow (text diagram)

**Tier A (validation):**
```text
gymnasium-robotics AdroitHandRelocate-v1  ─┐
minari D4RL/relocate/{human,expert}-v2  ───┼─► python/rl/sac.py (Condition A/B) ─► results/
                                            └─► (optional) stable-baselines3 SAC cross-check
```

**Tier B (the contribution):**
```text
ARCTIC / DexYCB / GRAB (raw, cached HaMeR outputs if using own video)
        │
        ▼  python/data/  (load + smooth + differentiate)
processed trajectories (q_human, qdot, qddot, object pose) [NPZ]
        │
        ▼  python/retargeting/ (calls dex-retargeting)
retargeted Allegro joint trajectories  [data/demonstrations/]
        │
        ├─► BC pretraining (python/rl/bc.py)
        ├─► replay-buffer seeding (replayed through python/envs/allegro_env.py)
        │
        ▼
python/rl/sac.py  (Conditions A–E, python/dynamics/ ensemble when C/D/E)
        │
        ▼
results/checkpoints/*.pt  ─►  python/export/  ─►  policy.onnx
                                                        │
                                                        ▼
                                        rust/inference_server/ (ort crate) ─► control API
```

## 4. Module responsibility boundaries (keep these crisp)

- `python/envs/` owns the MDP definition (state/action/reward). Nothing
  outside it should compute reward or read `mjData` directly.
- `python/dynamics/` owns the learned/residual dynamics ensemble and the
  MBPO-style synthetic rollout generator. It never talks to the real
  optimizer; it only produces transitions that `python/rl/sac.py`
  consumes identically to real ones.
- `python/rl/` owns the actor/critic, replay buffer, and the training
  loop. It is config-driven for all five ablation conditions — no
  condition-specific code branches outside of what's toggled by config
  flags (`05_RL_ALGORITHM_SPEC.md` §4).
- `rust/inference_server/` never imports PyTorch or talks to `python/`
  at runtime — its only contract with the training side is the `.onnx`
  file and the documented input/output tensor shapes
  (`11_PRODUCTIONIZATION.md`).
