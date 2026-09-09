# MuJoCo — A Physics Engine for Model-Based Control

**Citation:** Todorov, E., Erez, T., Tassa, Y. "MuJoCo: A Physics Engine
for Model-Based Control." IROS 2012.

**Link:** https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf

**Related repo:** `references/mujoco` (`google-deepmind/mujoco`),
Apache-2.0.

## Summary

MuJoCo is a generalized-coordinate rigid-body physics engine designed
specifically for model-based control and RL research: fast, accurate
contact modeling, and — critically for this project — first-class support
for both **forward** and **inverse** dynamics, computing

$$
M(q)\dot v + c(q,v) = \tau + J^T f
$$

and exposing the mass matrix, bias-force, Jacobian, and inverse-dynamics
computations as individual, callable functions rather than only as a
black-box stepper.

## What we borrow

Everything. MuJoCo is the simulator for this entire project (see
`06_ROBOT_AND_SIMULATION_SPEC.md`), and — per the biggest simplification
in `04_MATH_SPEC.md` §3 — we use its own `mj_fullM`, `mj_rne`,
`mj_jacSite`/`mj_jacBody`, and `mj_inverse` functions directly instead of
re-deriving analytical dynamics by hand, guaranteeing exact consistency
between "the analytical model" and "the simulator," and eliminating a
large, error-prone reimplementation task.

## Key API surface used in this project

| Function | Used for |
|---|---|
| `mj_step` | the actual RL transition, `mj_step(model, data)` |
| `mj_fullM` | dense mass matrix $M(q)$, Phase 3 validation |
| `mj_rne` | bias forces $c(q,\dot q)$ (Coriolis/centrifugal/gravity), Phase 3 |
| `mj_jacSite` / `mj_jacBody` | Jacobians for end-effector/fingertip kinematics |
| `mj_inverse` | full inverse dynamics $\tau_{ID}$, Phase 3 baseline controller and force/energy analysis |
| `mujoco.rollout` (module) | official batched multi-`mjData`, shared-`mjModel` rollout utility — reference design for our Tier 1.5 threaded parallelism attempt, `07_SYSTEM_ARCHITECTURE.md` |

We explicitly do **not** use MJX (the JAX implementation) or MuJoCo Warp
(the GPU-accelerated implementation) as the training-time runtime — see
`12_NON_GOALS_AND_CUT_SCOPE.md` §6 for why. Both remain read-only
references per the original plan's own instruction.
