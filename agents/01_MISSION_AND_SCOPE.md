# 01 — Mission and Scope

## 0. What stays the same

The core research question from the original plan is good and stays:

> **Do human manipulation demonstrations plus physics-structured dynamics
> models improve RL sample efficiency, stability, and generalization on
> dexterous manipulation, relative to RL from scratch, RL+demos alone, and
> RL+black-box learned dynamics alone?**

That is a real, standard, answerable RL research question. Everything in
this revision exists to make it *answerable in finite time on one desktop
GPU*, not to water down the question.

## 1. What changed, in one paragraph

The original plan asked for ten different subsystems (perception,
retargeting, analytical dynamics, residual dynamics, inverse dynamics,
demonstration-guided RL, parallel sim, continuous-time SDEs, HJB neural PDE
solving, Hamiltonian/port-Hamiltonian dynamics, differentiable physics) to
all ship in "version 1." That's a multi-year lab agenda, not a solo
project. This revision keeps the RL science (SAC, demonstrations, learned
dynamics) as standard, well-cited, discrete-time methods; replaces
custom-built subsystems with verified existing libraries wherever one
exists (retargeting, rigid-body dynamics, the RL algorithm skeleton, and
even a whole benchmark environment + demo dataset); and moves everything
that is genuinely research-frontier (HJB/SDE, Hamiltonian NNs,
differentiable physics, PETS) to an explicit non-goals list so it never
silently eats the schedule.

## 2. Two-tier project structure (new — this is the main structural change)

While verifying which pieces of this problem already have maintained
open-source solutions, we found that **the RL side of this problem
(dexterous hand + demonstrations + SAC/DAPG) already has a maintained
benchmark environment and a demonstration dataset**, from the exact paper
(Rajeswaran et al., DAPG) the original plan cites:
`Farama-Foundation/Gymnasium-Robotics`'s `AdroitHandRelocate-v1` env, with
human and expert demonstrations available through `Farama-Foundation/Minari`
(`D4RL/relocate/human-v2`, `D4RL/relocate/expert-v2`). See
`03_EXISTING_REPOS_TO_CLONE.md` for details. That changes how we should
sequence the project:

### Tier A — Correctness validation (do this first, ~1–2 weeks)

Train our own SAC + demo-integration + MBPO-style dynamics-augmentation
implementation on `AdroitHandRelocate-v1` using the off-the-shelf Minari
demonstrations. This is not the research contribution — it's a **known
benchmark with a known task** (pick up a ball, move it to a target) used to
prove our RL code is *correct* before we trust it on a novel environment.
If our SAC implementation can't get a reasonable success rate on a task the
field has solved many times over, nothing downstream is trustworthy. This
mirrors how a competent RL engineer actually de-risks a project.

### Tier B — The actual contribution (the "DynHand" pipeline)

Build our own environment (a floating **Allegro Hand** in MuJoCo, see
`06_ROBOT_AND_SIMULATION_SPEC.md`) and our own demonstration pipeline
(monocular/dataset human hand video → HaMeR or dataset annotations →
`dex-retargeting` → Allegro joint trajectories, see
`08_DATA_AND_RETARGETING_PIPELINE.md`), then run the **same, validated**
SAC/demo/dynamics trainer from Tier A on it. This is where the project's
actual novelty lives: applying a validated demo-guided, physics-structured
RL recipe to a *video-driven* retargeting pipeline onto a *modern*
(Menagerie) hand, which is not something any single existing repo does
end-to-end. Tier B is also where the ablation study (Section 22 of the
original plan, revised in `05_RL_ALGORITHM_SPEC.md`) is run, since we
control the environment fully (domain randomization, object variety,
held-out generalization splits) which the fixed Adroit benchmark doesn't
give us.

Tier A and Tier B share one trainer codebase, switched by config — this is
itself a design decision, not incidental: it means Tier A's validation
result is evidence about the exact code that later produces the Tier B
result.

## 3. MVP / Research / Stretch definitions (revised from original §44)

**MVP is complete when:**
```text
Tier A: SAC (+ optional BC-init) solves AdroitHandRelocate-v1 reproducibly
        (success rate and learning curve broadly consistent with published
        DAPG-era numbers on this task).
Tier B: ARCTIC/DexYCB/GRAB clip -> dex-retargeting -> Allegro joint
        trajectory -> replayed open-loop in MuJoCo without exploding, and
        SAC (from scratch) solves a single fixed cube pickup on the
        floating Allegro hand.
```

**Research version is complete when:**
```text
The 5-condition ablation matrix (A–E, see 05_RL_ALGORITHM_SPEC.md) has
been run on Tier B with matched compute budgets, sample-efficiency curves
plotted, and a defensible answer to the research question — even if the
answer is "no measurable improvement," which is a valid, reportable
result.
```

**Stretch (optional, only after Research version is done and only if time
remains):**
```text
- Shadow Hand (24 DoF) instead of/alongside Allegro.
- Live HaMeR demo: record your own webcam video, retarget it, replay it.
- Rust-native parallel environment stepper (Tier 2 perf layer).
- Rust ONNX inference server as a deployable artifact (11_PRODUCTIONIZATION.md).
- Generalization study: train on object set A, evaluate on held-out set B.
```

Nothing outside these three tiers ships in this project. If it's not in
`09_PHASE_PLAN.md`, don't build it — raise it as a documented future-work
item instead (see `12_NON_GOALS_AND_CUT_SCOPE.md` for the standing list).

## 4. Definition of done for the whole project

A reviewer should be able to: clone the repo, run one setup script, run one
command to reproduce the Tier A validation curve, run one command to
reproduce the Tier B ablation curves, and load the final exported ONNX
policy into the Rust inference server and get a control output back. That
end-to-end reproducibility is the actual bar, not any individual metric
value.
