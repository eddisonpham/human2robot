# 03 — Existing Repos To Clone (verified)

**Rule for the coding agent: before implementing any subsystem, check this
table. If a verified repo already does it, clone it into `references/`
and adapt it — do not re-derive it from scratch.** Every entry below was
checked (not assumed) during planning; "verified" means we confirmed the
repo currently exists, is installable/clonable, and does what we say it
does. Clone all of these in Phase 0 regardless of when they're first used.

## Tier 0 — Read for architecture, don't run as-is

### DexMV (`yzqin/dexmv-sim` + `yzqin/dexmv-learn`)
```bash
git clone https://github.com/yzqin/dexmv-sim.git references/dexmv-sim
git clone https://github.com/yzqin/dexmv-learn.git references/dexmv-learn
```
**This is the closest existing system to our full pipeline**: human video →
3D hand/object pose → demonstration translation (retargeting) → simulated
multi-finger hand (Adroit) → imitation/RL learning (DAPG among others),
Qin et al., ECCV 2022. **Read it for pipeline architecture and for how they
structure the demonstration-translation step.** Do **not** adopt its
Adroit-specific retargeting script or its `mjrl`-based DAPG trainer as a
production dependency — it targets an older hand model and an on-policy
(NPG) algorithm, whereas we standardize on the newer, more broadly-hand-
compatible `dex-retargeting` library and our own SAC-based trainer (see
`05_RL_ALGORITHM_SPEC.md` for why NPG-based literal-DAPG is not what we
implement).

### PDDM (`google-research/pddm`)
```bash
git clone https://github.com/google-research/pddm.git references/pddm
```
Nagabandi et al., "Deep Dynamics Models for Learning Dexterous
Manipulation." Model-based RL via learned dynamics + MPC. **Archived**
(read-only reference, not a dependency). Read for the data→model→
rollout→MPC→action structure that informs our MBPO-style dynamics
conditions (`05_RL_ALGORITHM_SPEC.md`).

## Tier A — Direct existing solution to the RL+demos benchmark

### Gymnasium-Robotics (`Farama-Foundation/Gymnasium-Robotics`)
```bash
pip install gymnasium-robotics
```
MIT-licensed, actively maintained by the Farama Foundation. Ships
`AdroitHandRelocate-v1` / `AdroitHandRelocateSparse-v1`: a 24-DoF Adroit
hand must pick up a ball and move it to a target — this *is* the
"reach → grasp → lift/relocate" task from the original plan §20, already
built, tested, and using current DeepMind `mujoco` bindings. **Use this
directly as the Tier A validation environment (see `01_MISSION_AND_SCOPE.md`).
No need to build this ourselves.**

### Minari (`Farama-Foundation/Minari`)
```bash
pip install minari
```
```python
import minari

ds = minari.load_dataset("D4RL/relocate/human-v2")  # verify exact id at Phase 0
env = ds.recover_environment()
```
Standardized offline-RL/demonstration dataset library. Hosts the *original*
DAPG (Rajeswaran et al.) human (CyberGlove-collected) and expert
demonstrations, regenerated against current Gymnasium-Robotics envs.
**Use directly as Tier A demonstrations — do not hand-collect or retarget
anything for Tier A.** Confirm the exact relocate dataset ids and license
terms at Phase 0 (`minari.list_remote_datasets()`); the pen/hammer variants
are confirmed to exist under `D4RL/pen/*`, `D4RL/hammer/*` — relocate
follows the same naming convention but verify before relying on it.

## Tier B — Perception, retargeting, robot assets

### dex-retargeting (`dexsuite/dex-retargeting`)
```bash
git clone https://github.com/dexsuite/dex-retargeting.git references/dex-retargeting
pip install dex_retargeting
```
MIT license. From the AnyTeleop project (Qin et al., RSS 2023). Actively
maintained (1,100+ stars, commits through mid-2025 at minimum). Implements
exactly the weighted-least-squares retargeting objective in
`04_MATH_SPEC.md` §2 (position/vector retargeting with temporal smoothing),
using Pinocchio internally for robot FK/Jacobians, with a **built-in
tutorial for retargeting from a hand-object pose dataset (their example is
DexYCB) into robot joint trajectories — precisely our Phase 4 use case.**
Robot URDFs come from the companion `dexsuite/dex-urdf` repo. **Do not
write a custom retargeting optimizer — call this library.** Confirm at
Phase 0 that `dex-urdf` includes the Allegro/Shadow hand URDFs matching our
MuJoCo Menagerie assets (very likely, given the ecosystem, but verify
before depending on it — if a matching URDF is missing, add one by
converting the Menagerie MJCF, don't reroute the whole retargeting
approach).

### MuJoCo Menagerie (`google-deepmind/mujoco_menagerie`)
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git references/mujoco_menagerie
```
DeepMind-curated, high-quality MJCF robot models. Confirmed contents we
use: `wonik_allegro/` (Allegro Hand V3, 16 DoF, BSD-2-Clause) and
`shadow_hand/` (Shadow Hand E3M5, 24 DoF, Apache-2.0), both already fitted
with position-controlled actuators and sane contact parameters. **Use
these MJCF files directly as the base robot model** (see
`06_ROBOT_AND_SIMULATION_SPEC.md` for the floating-base modification we
apply on top).

### HaMeR (`geopavlakos/hamer`)
```bash
git clone --recursive https://github.com/geopavlakos/hamer.git references/hamer
```
Verified actively maintained (updates as recent as Feb 2026). Transformer-
based monocular 3D hand pose/mesh estimator (Pavlakos et al., CVPR 2024).
**Runs offline only, in its own isolated environment, to produce cached
NPZ outputs — never inside the training loop** (per hardware constraints
in `02_TECH_STACK.md`). For the MVP we primarily use dataset-provided 3D
annotations (ARCTIC/DexYCB/GRAB) and only invoke HaMeR for the stretch-goal
"retarget my own webcam video" demo.

## Datasets (human hand/object trajectories)

### DexYCB (`NVlabs/dex-ycb-toolkit`)
```bash
git clone https://github.com/NVlabs/dex-ycb-toolkit.git references/dex-ycb-toolkit
```
Official evaluation/visualization toolkit for DexYCB (Chao et al., CVPR
2021). Dataset itself is **CC BY-NC 4.0 (non-commercial)** — note this in
`13_REPRODUCIBILITY_AND_CONVENTIONS.md`'s licensing section; fine for a
portfolio project, would need a different data source for a commercial
deployment. Includes a real "safe human-to-robot object handover" task,
directly relevant to our pickup objective.

### ARCTIC (`zc-alexfan/arctic`)
```bash
git clone https://github.com/zc-alexfan/arctic.git references/arctic
```
Official repo (Fan et al., CVPR 2023). Bimanual hand-object manipulation,
2.1M frames with dense contact info. Check the repo's `LICENSE` file for
exact commercial-use terms (research-only is typical for this class of
dataset) before any productionization step.

### GRAB (`otaheri/GRAB`)
```bash
git clone https://github.com/otaheri/GRAB.git references/grab
```
Official repo (Taheri et al., ECCV 2020). Whole-body human grasping,
useful for trajectory diversity.

## RL algorithm reference

### CleanRL (`vwxyzjn/cleanrl`)
```bash
git clone https://github.com/vwxyzjn/cleanrl.git references/cleanrl
```
MIT license, JMLR-documented, benchmarked. `cleanrl/sac_continuous_action.py`
is a correct, single-file, from-first-principles SAC implementation.
**Fork this one file** (CleanRL is explicitly designed to be forked, not
imported) as the base of `python/rl/sac.py`, then extend it with BC-init,
demo-buffer-seeding, and MBPO-style rollout augmentation per
`05_RL_ALGORITHM_SPEC.md`. Owning and modifying a correct reference
implementation, rather than calling an opaque library's `.train()`, is
deliberate — it's both lower-risk (you can read every line when something
looks wrong) and better resume signal.

### Stable-Baselines3 (`DLR-RM/stable-baselines3`) — secondary, optional
```bash
pip install stable-baselines3
```
Used only as an independent, "known-good" SAC implementation to sanity-
check a new environment (Phase 1 acceptance test): if SB3's SAC also fails
to learn on our env, the bug is in the environment, not our trainer. Not
the primary training path.

## Production inference

### ort (`pykeio/ort`)
```toml
# rust/inference_server/Cargo.toml
ort = "2"  # verify current major version at Phase 0
```
Actively maintained Rust ONNX Runtime wrapper (MIT/Apache-2.0 dual, used in
production by Twitter, Supabase, and others). See
`11_PRODUCTIONIZATION.md`.

## Summary table

| Repo | License | Role | Use as-is or adapt? |
|---|---|---|---|
| Gymnasium-Robotics | MIT | Tier A env | As-is |
| Minari | MIT | Tier A demos | As-is |
| dex-retargeting | MIT | Retargeting engine | As-is (call its API) |
| mujoco_menagerie | BSD-2 / Apache-2.0 (per-model) | Robot MJCF assets | Adapt (add floating base) |
| HaMeR | check repo license | Offline hand pose estimator | As-is, isolated env |
| dex-ycb-toolkit | check repo license, data CC BY-NC-4.0 | Dataset loader | As-is |
| ARCTIC | check repo license | Dataset loader | As-is |
| GRAB | check repo license | Dataset loader | As-is |
| CleanRL | MIT | SAC reference | Fork and extend |
| DexMV | check repo license | Architecture reference | Read only |
| PDDM | Apache-2.0 (archived) | MBPO/MPC reference | Read only |
| Stable-Baselines3 | MIT | Debugging cross-check | As-is, optional |
| ort | MIT/Apache-2.0 | Rust inference | As-is |
| mujoco-rs / rusty_mujoco | Apache-2.0 (verify per-crate) | Tier-2 perf stretch | As-is |
