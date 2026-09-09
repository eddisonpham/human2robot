# 08 — Data and Retargeting Pipeline

## 1. Data sources, in priority order

1. **Dataset-provided 3D annotations first.** DexYCB (grasping, single
   hand, YCB objects — closest to our pickup task) and, for diversity,
   ARCTIC/GRAB. Use their ground-truth/MANO-fit hand and object poses
   directly. **Do not run HaMeR on these** — that would throw away
   accurate annotations to re-estimate noisier ones from rendered images,
   which is strictly worse and slower. HaMeR is for *new* video without
   annotations.
2. **HaMeR, for the stretch "own video" demo only.** Record or source a
   short clip of a human hand performing a pickup, run it through
   `references/hamer/` (its own isolated environment,
   `07_SYSTEM_ARCHITECTURE.md` §1), cache the output. Pipeline:
   $$ I_t \xrightarrow{\text{HaMeR}} (\theta_t,\beta_t,T_t) \xrightarrow{\text{MANO}} M_t \rightarrow X_t $$
   producing exactly the same 3D-keypoint format as the dataset path, so
   everything downstream is identical either way.

## 2. Trajectory processing (kept from original plan, already standard practice)

```text
q_raw (human keypoints or dataset joint angles)
   -> smoothing (e.g. Savitzky-Golay or a low-pass filter; do not
      differentiate raw unfiltered predictions)
   -> velocity  (central difference: qdot_t ~= (q_{t+1}-q_{t-1}) / (2*dt))
   -> acceleration (qddot_t ~= (q_{t+1}-2*q_t+q_{t-1}) / dt^2)
```

Store per trajectory: `timestamp, q, qdot, qddot, object_pose,
object_velocity, contact_state, trajectory_id, task_id`, in NPZ (small,
simple, no external dependency) or Parquet if the `polars`-based pipeline
is used (`02_TECH_STACK.md`). This step is a good candidate for the Rust
`trajectory` crate (`07_SYSTEM_ARCHITECTURE.md` Tier 2) since it's
CPU-bound batch numeric work with no ML framework dependency — but a
correct Python/numpy version should exist and pass first (Phase 4), with
Rust as an optional, benchmarked speed-up afterward, not a blocking
dependency.

## 3. Retargeting — call `dex-retargeting`, don't rederive it

`dex-retargeting`'s `SeqRetargeting` (see `03_EXISTING_REPOS_TO_CLONE.md`
and `04_MATH_SPEC.md` §2) already implements the exact weighted-IK-with-
smoothing objective this project needs, and ships a tutorial specifically
for "retarget from hand-object pose dataset" using DexYCB as the worked
example — this is almost exactly our Phase 4 task.

```python
# python/retargeting/retarget_dataset.py — sketch, not final code
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

config = RetargetingConfig.load_from_file(
    "references/dex-retargeting/example/.../allegro_hand_right.yml"  # confirm exact path/robot at Phase 0
)
retargeting: SeqRetargeting = config.build()

for human_keypoints_t in dataset_trajectory:  # from python/data/
    q_t = retargeting.retarget(human_keypoints_t)  # -> robot joint positions
    # q_t indices are in dex-retargeting's own joint order — remap to the
    # MuJoCo Menagerie Allegro joint order via joint NAMES, never by
    # positional index (dex-retargeting's own README explicitly warns
    # about this — different URDF/MJCF parsers order joints differently).
```

**Phase 0 checklist item:** confirm `references/dex-urdf` (the companion
asset repo) has an Allegro-hand URDF whose joint set matches our modified
floating-base Menagerie MJCF (`06_ROBOT_AND_SIMULATION_SPEC.md`). If the
floating 6-DoF base isn't representable in `dex-retargeting`'s config
format directly, retarget the 16 finger DoF with `dex-retargeting` and
solve the 6 wrist DoF with a simple separate rigid transform (wrist pose =
retargeted/recorded human wrist pose directly, since it's a free 6-DoF
body, not a kinematic chain requiring IK) — do not force the whole
22-DoF problem through a single IK solve if the library isn't set up for
that; keep it simple.

Object-relative alignment for manipulation (not just hand-pose copying),
$L_{obj} = \lVert p_{hand}(q) - p_{obj}\rVert^2$, is configured via
`dex-retargeting`'s object-aware retargeting mode (used in its DexYCB
tutorial) — again, configuration, not new code.

## 4. Behavior-cloning action labels

Per `04_MATH_SPEC.md` §4 and `05_RL_ALGORITHM_SPEC.md` §2: $a_t^{demo}$ is
the retargeted target joint position (normalized delta), **not** an
inverse-dynamics torque. Store it alongside the retargeted trajectory so
`python/rl/bc.py` and the replay-buffer seeding step can consume it
directly with no further processing.

## 5. Storage schema

```text
data/demonstrations/<dataset>_<trajectory_id>.npz
  q            : (T, 22)   retargeted joint positions
  qdot         : (T, 22)
  a_demo       : (T, 22)   normalized action labels for BC (see above)
  object_pose  : (T, 7)    position (3) + quaternion (4)
  object_vel   : (T, 6)
  contact      : (T, K)    binary per-fingertip
  trajectory_id: str
  task_id      : str
  source       : {"dexycb","arctic","grab","hamer_custom"}
```

## 6. Train/test split — no leakage (kept from original plan §41, already correct)

Split by manipulation sequence and **object identity** first (never by
random frame). For the generalization experiment
(`10_EXPERIMENTS_AND_EVALUATION.md`), hold out entire object identities:
train on object set $A$, evaluate zero-shot on object set $B$, with
$D_{train} \cap D_{test} = \varnothing$ at the object level.

## 7. Licensing note (see also `03_EXISTING_REPOS_TO_CLONE.md`)

DexYCB is CC BY-NC 4.0. ARCTIC/GRAB carry their own research-use terms —
read each `LICENSE` file before assuming anything about redistribution or
commercial use. For a portfolio project this is a non-issue; if this
pipeline were ever repurposed for a commercial deployment, the
demonstration data source would need to change (e.g. first-party
teleoperation capture), while the retargeting/RL code itself has no such
restriction.
