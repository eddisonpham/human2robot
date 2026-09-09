# 02 — Tech Stack (exact, verified)

Every item below was checked against current documentation/registries
during planning (Sept 2026). Re-check versions at Phase 0 since this moves
fast; treat pinned numbers as "known-good as of writing," not eternal.

## 0. Hardware-driven ground rules

- GPU: RTX 5060, **8GB VRAM**. All RL/dynamics networks are small MLPs
  (≤256 hidden units). GPU compute is *never* the bottleneck in this
  project — CPU-bound MuJoCo stepping is. Do not over-invest in GPU-side
  optimization (mixed precision, `torch.compile`) before profiling shows
  it matters; do invest in CPU parallelism.
- CPU: 24 cores available, **cap workers at 12–16**, leave headroom for the
  OS, the main training process, and (later) HaMeR preprocessing running
  alongside.
- RAM: 32GB. Replay buffer of 1M transitions at our state/action
  dimensionality is a few hundred MB — keep it in CPU RAM, never on GPU.
- Single machine, single GPU, no multi-node/multi-GPU code paths anywhere.

## 1. Critical hardware gotcha — read this before installing PyTorch

**The RTX 5060 is an NVIDIA Blackwell-architecture GPU (compute capability
`sm_120`).** Verified: official *stable* PyTorch support for `sm_120`
shipped starting in **PyTorch 2.7.0** with CUDA 12.8 wheels. Anything older
will either fail outright (`CUDA error: no kernel image is available for
execution on the device`) or silently fall back to slow PTX JIT
compilation. Concretely:

```bash
# install a stable PyTorch build with CUDA 12.8 (or newer, e.g. 12.9/13.0)
# wheels — check pytorch.org's current selector at Phase 0, but do not
# accept anything older than 2.7.0 for this GPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_device_capability())"  # expect (12, 0)
```

Verify this in Phase 0 before writing a single line of training code — a
GPU that "works" but silently JIT-compiles every kernel will make later
performance numbers meaningless.

## 2. Languages and package managers

| Layer | Choice | Why |
|---|---|---|
| Python | **3.11** | Broadest current compatibility across `mujoco`, `gymnasium`, `dex-retargeting`, PyTorch cu128 wheels |
| Python packaging | **uv** (fallback: plain venv + pip) | Fast, reproducible, single lockfile; `pip install --break-system-packages` only if working outside a venv |
| Rust | stable channel, 2021 edition, MSRV ≥1.82 | Matches current crate requirements (see below) |
| Rust packaging | **cargo workspace** | Standard |
| Rust↔Python bridge | **PyO3 + maturin** | Standard, mature, used for the Tier-2 perf layer |

HaMeR (Section 5 of the original plan) has its own heavy, partially
conflicting dependency set (`detectron2`, `ViTPose`, specific `torch`
CUDA build). **Give it its own isolated virtual environment
(`envs/hamer/`)**, entirely separate from the main training environment.
It is only ever run offline, once, to produce cached NPZ files — it never
needs to share a process or dependency tree with the RL trainer.

## 3. Core Python libraries

| Purpose | Library | Notes |
|---|---|---|
| Simulation | `mujoco` (official DeepMind PyPI package) | ≥3.3.x. CPU physics engine, primary source of all training transitions |
| RL env API | `gymnasium` | Farama's maintained fork of OpenAI Gym; required by Gymnasium-Robotics/Minari |
| Tier A benchmark env | `gymnasium-robotics` | Provides `AdroitHandRelocate-v1` |
| Tier A demos | `minari` | Provides `D4RL/relocate/human-v2`, `D4RL/relocate/expert-v2` |
| Retargeting | `dex-retargeting` (`pip install dex_retargeting`) | Pulls in `pinocchio` for FK/Jacobians internally — do not add a separate Pinocchio dependency by hand |
| Deep learning | `torch` ≥2.7.0, cu128 build | See §1 |
| RL algorithm base | vendored/forked from `cleanrl`'s `sac_continuous_action.py` | Not installed as a library — CleanRL is explicitly not meant to be imported; fork the single file and own it |
| Config | `pydantic` + YAML (`pyyaml`) | Schema-validated configs, fail fast on typos |
| Data pipeline (optional, fast tabular ops) | `polars` | Rust-backed dataframe library; faster than pandas for the trajectory NPZ→table bookkeeping in `08_DATA_AND_RETARGETING_PIPELINE.md` |
| Logging | `tensorboard` | Local, no cloud dependency required |
| Export | `onnx`, `onnxruntime` | Export + parity-check the trained actor before handing to Rust |
| Testing | `pytest` | |

## 4. Core Rust crates

| Purpose | Crate | Notes |
|---|---|---|
| MuJoCo bindings (Tier-2 perf stretch only) | `mujoco-rs` (tracks MuJoCo 3.3.7, RAII-safe high-level wrapper with a native viewer) | Verified actively maintained; alternative is `rusty_mujoco` (lower-level, also current). Do **not** use the old `TheButlah/mujoco-rs` / `mujoco-sys` crates — archived since 2021. |
| ONNX inference (production server) | `ort` (`pykeio/ort`) | Verified actively maintained, widely deployed (used by Twitter, Supabase, etc.), CPU and GPU execution providers |
| Python bridge | `pyo3`, `maturin` | Standard |
| HTTP/serving (production server) | `axum` (or `actix-web`) | Standard, async, lightweight |
| Fast dataframes (optional, trajectory preprocessing) | `polars` (Rust crate directly, if the preprocessing step is moved fully into Rust) | Same engine as the Python `polars` package |
| Serialization | `serde`, `serde_json` | Standard |
| Numerics | `ndarray` | Standard, used only where Rust does its own trajectory-processing math (Phase 4/5) |

## 5. What we deliberately do NOT add to the stack, and why

- **JAX / MJX / MuJoCo Warp** — MJX targets GPU-batched simulation via
  JAX. Running JAX and PyTorch on the *same* 8GB card simultaneously
  invites VRAM contention and CUDA-context overhead for no benefit here
  (our env count is CPU-parallel-friendly, not GPU-batch-friendly at this
  scale). Read MJX in `references/` for ideas only, per the original
  plan's own instruction; do not adopt it as the runtime.
- **Isaac Gym / Isaac Lab** — designed around larger-VRAM GPUs and a much
  heavier Isaac Sim install; not a good fit for an 8GB single-GPU box.
  MuJoCo CPU-parallel is the right-sized choice here.
- **Stable-Baselines3 as the primary trainer** — kept only as an optional,
  secondary "known-good" implementation to sanity-check that an
  environment bug isn't masquerading as an SAC bug (see
  `09_PHASE_PLAN.md`, Phase 1). The primary trainer is our own
  CleanRL-derived, fully-understood SAC implementation, because
  demonstrating you can implement and modify SAC correctly is part of the
  point of a resume project.
- **A custom Rust rigid-body-dynamics engine** — MuJoCo's own `mj_fullM`,
  `mj_rne`, `mj_jac*`, `mj_inverse` already give exact, simulator-consistent
  mass matrix, bias forces, Jacobians, and inverse dynamics. Re-deriving
  RNEA/CRBA by hand in Rust adds large implementation/debugging risk for
  zero scientific benefit (see `04_MATH_SPEC.md` §3).
