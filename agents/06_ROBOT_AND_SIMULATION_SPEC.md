# 06 — Robot and Simulation Spec

## 1. Robot choice: floating Allegro Hand (MVP), Shadow Hand (stretch)

**MVP robot: Wonik Allegro Hand V3, floating base.** Base asset:
`references/mujoco_menagerie/wonik_allegro/` (16 actuated finger DoF,
BSD-2-Clause, position actuators already configured). We add a **6-DoF
floating joint** (3 translation + 3 rotation) at the palm body in place of
the default fixed weld, giving **22 total controllable DoF** (6 wrist + 16
fingers).

This "floating hand, no arm" setup is not an ad hoc simplification we
invented — it is the standard simplification used in the dexterous-
manipulation literature specifically to decouple hand dexterity research
from arm-IK/reachability concerns (e.g. BODex, Cai et al. 2024/2025,
explicitly evaluates "using a floating hand without a robot arm" with
Shadow/Allegro assets sourced from MuJoCo Menagerie — see
`papers/` if a note is added, or cite directly in the report). It also
gives us the "reach" phase (original plan §15: move the wrist to a target)
for free, since the floating joint *is* the thing that reaches.

**Stretch robot: Shadow Hand E3M5**, same treatment
(`references/mujoco_menagerie/shadow_hand/`, 24 finger DoF, Apache-2.0,
30 DoF total floating). Only attempted after the Allegro pipeline is fully
validated (Research version complete, per `01_MISSION_AND_SCOPE.md`).

**Explicitly rejected:** building a custom hand model from scratch, or
using a full arm+hand (e.g. UR5e/Panda + Allegro) for the MVP — an arm
adds inverse-kinematics/reachability engineering that is orthogonal to the
manipulation-RL research question and is pure schedule risk for this
project's scope.

## 2. Control interface

- Action space: $u_t \in [-1,1]^{22}$, mapped to each position actuator's
  control range (the Menagerie XML's own `ctrlrange`), applied as a
  **target position delta from the previous action**, clipped to a
  per-step max delta to avoid teleporting the wrist (standard practice in
  dexterous-hand RL, e.g. OpenAI's "Learning Dexterous In-Hand
  Manipulation" uses this exact relative-target convention).
- Physics timestep: `dt = 0.002s` (500 Hz) — MuJoCo's own recommended
  default for contact-rich hand/object simulation.
- Control decimation: 10 physics substeps per control step → **50 Hz
  control rate**. Configurable in `configs/base.yaml`; validate in Phase 1
  that this doesn't cause instability before locking it in.
- Episode horizon: 400–600 control steps (8–12 seconds of simulated time).

## 3. Objects (MVP)

Cube, cylinder, sphere primitives (kept from original plan §20 — already a
reasonable, standard, incrementally-harder object set). Randomized per
episode within documented ranges (size, mass, friction) for the domain-
randomization/generalization experiments in `10_EXPERIMENTS_AND_EVALUATION.md`.
Do **not** start with articulated objects (kept from original plan's own
correct instruction).

## 4. Domain randomization ranges (defaults, tune in Phase 11)

| Parameter | Range |
|---|---|
| Object mass | 0.05–0.3 kg |
| Sliding friction | 0.3–1.2 |
| Object size (cube half-extent / cylinder radius / sphere radius) | 0.02–0.045 m |
| Initial object pose | within the hand's reachable workspace, randomized $(x,y)$, fixed table height |
| Actuator gain (for Condition D's sim-to-real-style gap, see `05_RL_ALGORITHM_SPEC.md` §3) | ±20% of nominal |
| Sensor noise (optional, later phase) | small Gaussian noise on $q,\dot q$ observations |

## 5. Contact/observation features

Per-fingertip binary contact flags read from MuJoCo touch sensors (add
`<touch>` sensors to the fingertip geoms in the modified MJCF — a few
lines of XML, no custom contact model). Do not attempt full learned
contact mechanics (original plan §12's own instruction, kept).

## 6. Tier A benchmark environment (no modification needed)

`AdroitHandRelocate-v1` / `AdroitHandRelocateSparse-v1` from
`gymnasium-robotics`, used as-is (24-DoF Adroit hand, ball relocate task,
dense or sparse reward variant — see `03_EXISTING_REPOS_TO_CLONE.md`). No
XML editing, no custom reward — the point of Tier A is to validate against
an unmodified, external, standard benchmark.

## 7. Validation checklist for the modified Allegro MJCF (Phase 1)

1. Loads without XML errors; `mujoco.MjModel.from_xml_path` succeeds.
2. Runs $10^4$+ steps with random actions without NaN/divergence.
3. Same seed → bit-identical (or float-tolerance-identical) trajectory on
   repeated runs (determinism check).
4. Object falls under gravity when unsupported (sanity check on inertial
   properties after the floating-joint edit).
5. Joint limits and actuator `ctrlrange` are respected (no silent clipping
   bugs).
6. Floating base doesn't introduce degenerate contacts with itself (check
   `<contact><exclude>` clauses carried over correctly from the original
   Menagerie XML after the palm-weld → floating-joint edit).
