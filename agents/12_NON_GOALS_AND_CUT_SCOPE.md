# 12 — Non-Goals and Cut Scope

This file exists so that "should we build X" has a written answer instead
of being re-litigated mid-project. Everything here was in the original
brain-dump plan. Each item is cut **from the core deliverable**, with the
specific reason, and a note on if/how it could ever come back.

## 1. Continuous-time stochastic dynamics and the HJB equation (original §24–28, §46)

$$ dx_t = f_\theta(x_t,u_t)\,dt + \Sigma_\theta(x_t,u_t)\,dW_t, \qquad -\partial_t V = \min_u[\cdots] $$

**Cut entirely**, not deferred as a "phase," from both the math spec
(`04_MATH_SPEC.md`) and the RL spec (`05_RL_ALGORITHM_SPEC.md`). Reasons:

- It is not the standard formulation for this problem. SAC, DAPG, MBPO,
  and every other paper this plan actually cites operate in discrete
  time. Introducing continuous-time HJB machinery on top adds an entire
  second, harder theory (stochastic optimal control, PDE residual
  learning) without changing what the resulting controller could do.
- PDE-residual value-function learning is a niche technique with a track
  record mostly on very low-dimensional toy systems (e.g. Doya 2000's own
  experiments); there is no strong evidence it scales cleanly to a
  22-DoF contact-rich manipulation problem, and debugging a failed HJB
  residual is materially harder than debugging a failed SAC run.
- Given the hardware and time budget (`02_TECH_STACK.md`), this is pure
  schedule risk with no expected payoff for the stated research question,
  which is fully answerable in the standard discrete-time MDP framing.

**If ever revisited:** as a clearly-labeled, separate research spike, only
after the discrete-time system is fully working, and only first validated
on a toy 2–4 dimensional system (e.g. a pendulum) before ever touching the
22-DoF hand.

## 2. Hamiltonian neural networks / port-Hamiltonian dynamics (original §29–30)

**Cut.** These are legitimate, interesting techniques for learning
energy-conserving dynamics, but they are an alternative *dynamics-model
architecture*, not something the core research question needs — the
question is about demonstrations and physics-structured dynamics helping
RL, and the standard residual-MLP dynamics model in `04_MATH_SPEC.md` §5
already tests that. Swapping in a Hamiltonian architecture is an
orthogonal ablation, not a requirement, and adds Lie-group/energy-
conservation machinery this project doesn't need to answer its question.

## 3. Differentiable physics / trajectory optimization branch (original §31, Phase 16)

**Cut.** Comparing "gradient-based physics optimization vs. RL" is a
genuinely interesting question, but it's a *different* research question
from the one this project asks, and it requires either MJX/JAX (rejected
in `02_TECH_STACK.md` for VRAM-contention reasons on an 8GB single-GPU
box) or a from-scratch differentiable simulator. Not attempted here.

## 4. PETS / probabilistic ensembles with trajectory sampling (original §33)

**Cut**, in favor of MBPO's simpler deterministic/short-rollout ensemble
approach (`05_RL_ALGORITHM_SPEC.md` §3), which already gives us a
standard, citable way to use a learned dynamics model to accelerate an
off-policy actor-critic. PETS-style CEM/trajectory-sampling MPC is a
reasonable alternative but doubles the model-based machinery we'd need to
implement and debug for a benefit this project doesn't need to
demonstrate its point.

## 5. Full custom Rust rigid-body dynamics engine (original §9, Phase 2's original framing)

**Cut**, replaced by calling MuJoCo's own `mj_fullM`/`mj_rne`/`mj_jac*`/
`mj_inverse` (`04_MATH_SPEC.md` §3). Reimplementing RNEA/CRBA by hand is a
substantial, error-prone undertaking whose only purpose in the original
plan was "physics-structured dynamics" — which the simulator's own,
exactly-consistent functions already provide, with far less risk.

## 6. MJX / MuJoCo Warp / Isaac Gym / Isaac Lab as the runtime simulator

**Rejected** (not part of the original plan, but a natural question a
reviewer might ask, so answered here explicitly). MJX and Isaac Lab both
target GPU-batched simulation, which competes with PyTorch for the same
8GB of VRAM and adds a second heavy simulation stack (JAX, or the full
Isaac Sim install) to an already CPU-parallel-friendly problem size (our
env count and model size don't need GPU batching to hit good throughput).
CPU-parallel MuJoCo (`07_SYSTEM_ARCHITECTURE.md` Tier 1) is the correctly
-sized choice. MJX remains a read-only reference per the original plan's
own instruction, never a runtime dependency.

## 7. Literal DAPG (on-policy NPG) as the RL algorithm

**Not implemented as such.** The original plan cited DAPG (an NPG-based,
on-policy algorithm) but wanted it combined with SAC (off-policy) — a real
inconsistency. We keep DAPG as a citation for *why* demonstrations help,
and implement the standard, correct off-policy analog instead
(Večerík et al. 2017 / Nair et al. 2018 style BC-init + demo-buffer-
seeding, `05_RL_ALGORITHM_SPEC.md` §2). See `papers/dapg_rajeswaran_2018.md`
for the precise distinction.

## 8. Sim-to-real / physical robot deployment

**Out of scope** for this project entirely (see `11_PRODUCTIONIZATION.md`
§5). No physical Allegro/Shadow hand is assumed available; "production-
ready" here means "a real controller could call this served model,"
demonstrated by a tested API contract, not an actual hardware integration.

## 9. Standing rule for anything not on this list either

If a technique isn't in `04_MATH_SPEC.md`, `05_RL_ALGORITHM_SPEC.md`, or
named as a stretch item in `01_MISSION_AND_SCOPE.md` §3, and it isn't
explicitly cut here, raise it as a documented open question before
implementing it — don't silently add scope. The single biggest risk to
this project succeeding is scope creep back toward the original
ten-subsystem plan; this file is the guardrail against that.
