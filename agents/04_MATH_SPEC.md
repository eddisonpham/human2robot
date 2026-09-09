# 04 — Math Specification (standard, discrete-time only)

Everything in this file is textbook material: a discrete-time MDP, a
weighted least-squares retargeting objective, and MuJoCo's own rigid-body
dynamics equations. If you find yourself reaching for a PDE, a stochastic
differential equation, or a Hamiltonian, stop — that's out of scope, see
`12_NON_GOALS_AND_CUT_SCOPE.md`.

## 1. State, action, and the MDP

We formalize manipulation as a standard finite-horizon discrete-time MDP
$(\mathcal S, \mathcal A, P, r, \gamma, T)$.

**State.** For the robot hand with $N$ actuated joints plus a floating
6-DoF wrist (see `06_ROBOT_AND_SIMULATION_SPEC.md`), and a manipulated
rigid object:

$$
x_t = \big[\, q_t,\ \dot q_t,\ p_t^{o},\ r_t^{o},\ v_t^{o},\ \omega_t^{o} \,\big]
$$

where $q_t \in \mathbb R^N$ is joint position, $\dot q_t \in \mathbb R^N$
joint velocity, $p_t^o \in \mathbb R^3$ object position, $r_t^o$ a 3D
rotation-vector representation of object orientation (never raw unbounded
Euler angles — this was already correct in the original plan, keep it),
$v_t^o, \omega_t^o \in \mathbb R^3$ object linear/angular velocity. Add a
small set of scalar contact features $c_t \in \{0,1\}^K$ (per-fingertip
contact booleans, read directly from MuJoCo contact/touch sensors — no
custom contact model needed) and, for goal-conditioned tasks, a target
position $p^{target}$.

**Action.** $u_t \in [-1, 1]^N$, a normalized command to the robot's
**position actuators** (see `06_ROBOT_AND_SIMULATION_SPEC.md` — the
Menagerie hand assets ship with position-controlled actuators, so this is
the physically-correct action space for these specific assets, not an
arbitrary choice). $u_t$ is affine-mapped to each actuator's control range
by the simulator; we do not additionally model torque as the action.

**Transition.** $P(x_{t+1} \mid x_t, u_t)$ is implicit: it *is* one MuJoCo
`mj_step` call (with a fixed number of physics substeps per control step,
"decimation" — see `06_ROBOT_AND_SIMULATION_SPEC.md`). We never need a
closed-form transition density; SAC and MBPO are both compatible with a
black-box simulator/model transition.

**Reward.** A standard shaped, decomposed reward (kept from the original
plan §21, it was already reasonable):

$$
r_t = r_{approach} + r_{grasp} + r_{lift} - r_{energy} - r_{collision}
$$

$$
r_{approach} = -\alpha \lVert p_{hand} - p_o \rVert^2, \quad
r_{grasp} = \beta \cdot \mathbf 1[\text{stable multi-finger contact}], \quad
r_{lift} = \gamma \max(0,\ z_o - z_{init})
$$

$$
r_{energy} = \eta \lVert \tau_t \rVert^2, \qquad
\tau_t := \text{`actuator\_force`, read directly from MuJoCo (works for position actuators too)}
$$

Success: $z_o - z_{init} > \Delta z$ maintained for $K$ consecutive frames.
All constants ($\alpha,\beta,\gamma,\eta,\Delta z,K$) live in
`configs/*.yaml`, tuned empirically, logged per experiment
(`13_REPRODUCIBILITY_AND_CONVENTIONS.md`).

**Objective.** Standard discounted return:

$$
\pi^\star = \arg\max_\pi \ \mathbb E_{\tau \sim \pi} \Big[ \sum_{t=0}^{T-1} \gamma^t r_t \Big], \qquad \gamma = 0.99
$$

This — not the continuous-time HJB formulation in the original plan §25–28
— is the actual objective we optimize. SAC (`05_RL_ALGORITHM_SPEC.md`)
solves exactly this, correctly, with well-understood convergence behavior.

## 2. Retargeting objective (implemented by `dex-retargeting`, don't rederive)

Given human hand keypoints $X_t^H \in \mathbb R^{K\times 3}$ and robot
forward kinematics $X^R(q) = [x_1(q), \dots, x_K(q)]$ for corresponding
robot keypoints, retargeting solves, per frame, a weighted least-squares
problem with temporal smoothing:

$$
q_t^\star = \arg\min_q \ \lVert X^R(q) - X_t^H \rVert_W^2 \ +\ \lambda \lVert q - q_{t-1}^\star \rVert^2
\quad \text{s.t.}\quad q_{\min} \le q \le q_{\max}
$$

This is a standard bounded weighted IK problem, solved by nonlinear
least-squares (SLSQP or similar) using analytic Jacobians from Pinocchio.
**This is exactly what `dex-retargeting`'s `SeqRetargeting` class does.**
We call it; we do not reimplement it. See `08_DATA_AND_RETARGETING_PIPELINE.md`
for the calling convention and how object-relative alignment (for
manipulation, not just pose-copying) is configured.

## 3. Rigid-body dynamics — use the simulator's own functions

The general floating-base manipulator equation, in the simulator's own
convention:

$$
M(q)\dot v + c(q, v) = Bu + J_c(q)^T \lambda
$$

**Decision (this is the single biggest simplification versus the original
plan):** we do **not** hand-derive or reimplement $M(q)$, $c(q,\dot q)$,
$J(q)$ in Rust or Python. MuJoCo's C API already computes all of these,
exactly consistent with what the simulator itself integrates, via:

| Quantity | MuJoCo function | Exposed in `mujoco` Python as |
|---|---|---|
| $M(q)$ (dense mass matrix) | `mj_fullM` | `mujoco.mj_fullM` |
| $c(q,\dot q)$ (bias forces: Coriolis+centrifugal+gravity) | `mj_rne` (with acceleration=0) | `mujoco.mj_rne` |
| $J(q)$ (site/body Jacobians) | `mj_jacSite` / `mj_jacBody` | `mujoco.mj_jacSite` etc. |
| $\tau_{ID}$ (full inverse dynamics) | `mj_inverse` | `mujoco.mj_inverse` |

Re-deriving these by hand invites a subtle mismatch with whatever the
simulator actually integrates (different Coriolis conventions, different
handling of `mjOption` flags, etc.) for no scientific upside — "use the
exact simulator convention and document it," which the original plan
already said, is best satisfied by literally calling the simulator's own
functions. **Phase 2's acceptance test (`09_PHASE_PLAN.md`) becomes: call
`mj_inverse` on a known trajectory and confirm the returned generalized
force reproduces the applied control to within solver tolerance — this
validates our understanding of the model, not a from-scratch dynamics
implementation.**

## 4. Inverse dynamics and computed-torque control — validation tool, not the action interface

$$
\tau_{ID} = M(q)\ddot q_d + C(q,\dot q)\dot q + g(q) - J_c^T\lambda
$$

with a standard PD-tracking target:

$$
\ddot q_d = \ddot q_{ref} + K_D(\dot q_{ref} - \dot q) + K_P(q_{ref} - q)
$$

Because our actual robot assets (Menagerie) ship with **position**
actuators, `mj_inverse`/computed-torque control is used for two specific,
bounded purposes — not as the RL action interface:

1. **Phase 3 classical-control baseline.** On a torque-actuated variant of
   the model, verify computed-torque tracking against a reference
   trajectory (RMSE metric, original plan §11) as a controls sanity check
   and a "here is a classical baseline" comparison point for the write-up.
2. **Analysis.** Offline force/energy accounting, and (optionally) as a
   feasibility check when generating synthetic demonstration trajectories.

**Behavior-cloning demonstration actions are *not* $\tau_{ID}$.** Since our
robot is position-controlled, the correct, standard, and far simpler choice
is: $a_t^{demo}$ = the retargeted/dataset target joint position (delta from
current), normalized into $[-1,1]^N$ exactly like the policy's own action
space. This removes an unnecessary and error-prone dependency (turning
kinematic demonstrations into torques) from the main learning pipeline —
see `05_RL_ALGORITHM_SPEC.md` §2.

## 5. Residual/learned dynamics — precisely scoped

Kept from the original plan, but precisely operationalized (see
`05_RL_ALGORITHM_SPEC.md` §4 for exactly how this plugs into training —
this section only fixes the equation and the network):

$$
f_{true}(x,u) = f_{physics}(x,u) + r_\theta(x,u), \qquad \ddot q = \ddot q_{physics} + r_\theta(q,\dot q,u)
$$

Network (kept from original plan, it was already a reasonable, small,
standard MLP):

```text
input -> Linear(256) -> SiLU -> Linear(256) -> SiLU -> Linear(N)
```

trained with:

$$
L_{dyn} = \frac{1}{B}\sum_{i=1}^{B} \big\lVert \hat{\Delta \ddot q}_i - \Delta \ddot q_i \big\rVert_2^2
$$

evaluated at multi-step rollout horizons $k \in \{1,5,10,20,50\}$ via
$E_k = \frac1T \sum_t \lVert \hat x_{t+k} - x_{t+k}\rVert_2$ — unchanged
from the original plan, this evaluation protocol was already correct
practice.

## 6. What is explicitly NOT in this spec

Continuous-time stochastic dynamics ($dx = f\,dt + \Sigma\,dW$), the HJB
PDE, Hamiltonian/port-Hamiltonian neural dynamics, and differentiable
physics trajectory optimization are all **removed from the core math
spec.** See `12_NON_GOALS_AND_CUT_SCOPE.md` for the reasoning. Everything
this project trains, evaluates, and reports on uses only §1–5 above.
