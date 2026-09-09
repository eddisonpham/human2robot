# DynHand — Agent Specification Index

This folder is the complete, standalone build specification for **DynHand**: a
physics-grounded, demonstration-guided RL system that retargets human hand
video into manipulation skills for a simulated dexterous robot hand.

It is a **revision** of an earlier, much more ambitious brain-dump. The
revision's job is to make the project (a) buildable solo, locally, on one
RTX 5060 (8GB VRAM) + 24 CPU cores, (b) grounded in **standard, textbook RL
and rigid-body-dynamics math** (no PDE/HJB/SDE machinery), (c) built on
**verified, actively-maintained existing repositories wherever one already
solves a piece of the problem**, and (d) engineered to a bar that would
survive a "could we actually ship this" review at a company, using Rust
where Rust genuinely earns its place and Python/PyTorch everywhere else.

If you are a coding agent: **read files in the order below before writing
any code.** Each file is self-contained but they cross-reference each other;
do not skip `12_NON_GOALS_AND_CUT_SCOPE.md` — it tells you what *not* to
build, which is as load-bearing as what to build.

## Reading order

| # | File | What it locks down |
|---|------|---------------------|
| 1 | `01_MISSION_AND_SCOPE.md` | Research question, MVP/Research/Stretch tiers, Tier A vs Tier B split |
| 2 | `02_TECH_STACK.md` | Exact languages, libraries, versions, hardware gotchas |
| 3 | `03_EXISTING_REPOS_TO_CLONE.md` | Every third-party repo we build on, verified, with license + role |
| 4 | `04_MATH_SPEC.md` | The standard MDP, retargeting, and dynamics equations we actually implement |
| 5 | `05_RL_ALGORITHM_SPEC.md` | SAC spec, demo integration, model-based augmentation, the 5 ablation conditions |
| 6 | `06_ROBOT_AND_SIMULATION_SPEC.md` | Robot(s), MuJoCo assets, control interface, domain randomization |
| 7 | `07_SYSTEM_ARCHITECTURE.md` | Rust/Python split, repo layout, data flow |
| 8 | `08_DATA_AND_RETARGETING_PIPELINE.md` | Datasets, HaMeR, retargeting, trajectory storage |
| 9 | `09_PHASE_PLAN.md` | Sequential build phases with acceptance criteria |
| 10 | `10_EXPERIMENTS_AND_EVALUATION.md` | Metrics, ablations, plots, reproducibility manifest |
| 11 | `11_PRODUCTIONIZATION.md` | ONNX export, Rust inference server, serving contract |
| 12 | `12_NON_GOALS_AND_CUT_SCOPE.md` | Everything cut from the original plan, and why |
| 13 | `13_REPRODUCIBILITY_AND_CONVENTIONS.md` | Config schema, seeds, testing, git conventions |
| — | `papers/INDEX.md` | Every paper we implement or cite, pre-digested for the coding agent |

## Non-negotiable constraints (repeated everywhere on purpose)

1. **Hardware**: RTX 5060, 8GB VRAM, 24 CPU cores, 32GB RAM, single Linux
   machine, everything trains locally. Never assume cloud/multi-GPU.
2. **Math must be standard.** Discrete-time MDP + SAC. No continuous-time
   HJB/SDE, no Hamiltonian neural nets, no differentiable-physics branch in
   the core deliverable. See `12_NON_GOALS_AND_CUT_SCOPE.md`.
3. **Prioritize existing, verified repos over writing new code.** Before
   implementing anything, check `03_EXISTING_REPOS_TO_CLONE.md` — if it's
   there, clone and adapt it; do not re-derive it from scratch.
4. **Tech stack is Rust + Python/PyTorch**, scoped precisely (see
   `02_TECH_STACK.md` and `07_SYSTEM_ARCHITECTURE.md`) — Rust is not
   sprinkled everywhere, it is used where it has a clear, justified job.
5. **Every phase has a measurable acceptance test.** No phase is "done"
   because code runs without crashing; it's done when its acceptance
   criterion in `09_PHASE_PLAN.md` passes.
6. **This is a portfolio/production-quality project.** Code should be
   something you'd hand to a new hire, with tests, configs, logging, and a
   servable model artifact at the end — see `11_PRODUCTIONIZATION.md`.

## Folder layout this spec assumes

```text
agents/                      <- you are here (planning docs, read-only reference)
references/                  <- cloned third-party repos (see file 03)
  ├── mujoco_menagerie/
  ├── dex-retargeting/
  ├── dexmv-sim/ dexmv-learn/
  ├── cleanrl/
  ├── hamer/
  ├── dex-ycb-toolkit/ arctic/ grab/
  └── NOTES.md              <- Phase 0 deliverable, one section per reference
data/
  ├── raw/ processed/ demonstrations/
configs/
python/                      <- training, retargeting glue, evaluation
rust/                        <- trajectory pipeline, (stretch) env stepper, inference server
experiments/
results/
tests/
```
