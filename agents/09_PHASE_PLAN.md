# 09 — Phase Plan (sequential, each phase has a measurable acceptance test)

Execute in order. Do not start phase $N+1$ before phase $N$'s acceptance
criterion passes. Every phase names which existing repo (if any) it
leans on — check `03_EXISTING_REPOS_TO_CLONE.md` before writing new code.

## Phase 0 — Research setup and environment verification

**Tasks**
1. Clone every repo in `03_EXISTING_REPOS_TO_CLONE.md` into `references/`.
2. Verify the PyTorch/CUDA Blackwell setup (`02_TECH_STACK.md` §1):
   `torch.cuda.get_device_capability()` reports `(12, 0)`, a trivial MLP
   forward/backward runs on GPU without kernel-image errors.
3. Create the isolated HaMeR environment (`envs/hamer/`) and confirm it
   installs and runs its own demo script — but do this in parallel/last,
   it is not on the MVP critical path.
4. Confirm `dex-retargeting` installs and its bundled example runs against
   a sample DexYCB clip.
5. Confirm `minari.load_dataset("D4RL/relocate/human-v2")` (or the
   correctly-named current dataset id — verify exact id, it may differ)
   loads and `recover_environment()` produces a working
   `AdroitHandRelocate` env.
6. Write `references/NOTES.md`: one section per major reference, recording
   the exact algorithmic idea borrowed from each and any version/API
   deltas found versus what this spec assumed. Do not copy code blindly
   into it — summarize and cite.

**Acceptance:** `references/NOTES.md` exists with one section per
reference in `03_EXISTING_REPOS_TO_CLONE.md`; steps 2, 4, 5 above each
produce a passing smoke-test script committed under `tests/python/test_phase0_*.py`.

## Phase 1 — Tier A: reproduce a known RL+demos result (correctness gate)

**Tasks**
1. Fork CleanRL's `sac_continuous_action.py` into `python/rl/sac.py`
   (`03_EXISTING_REPOS_TO_CLONE.md`), adapt to `AdroitHandRelocateSparse-v1`.
2. Run Condition A (plain SAC) and Condition B (BC-init + demo-buffer-
   seeding from Minari demos, `05_RL_ALGORITHM_SPEC.md` §2) on this env.
3. Cross-check Condition A against Stable-Baselines3's SAC on the same
   env, same seed budget.

**Acceptance:** our SAC and SB3's SAC produce learning curves within a
reasonable band of each other on the same task (no order-of-magnitude
divergence); Condition B measurably improves sample efficiency over
Condition A on this benchmark, broadly consistent with DAPG-era published
numbers on this task (we are not trying to match numbers exactly — the
literature uses a different algorithm — but a completely absent
improvement here means something is wrong with the demo-integration code,
not a valid research finding). **This phase's purpose is to prove the RL
code is correct before it's ever trusted on Tier B.**

## Phase 2 — Tier B simulator setup: modified Allegro MJCF

**Tasks:** apply the floating-base modification to
`references/mujoco_menagerie/wonik_allegro/` (`06_ROBOT_AND_SIMULATION_SPEC.md`
§1), add touch sensors, add primitive objects to the scene.

**Acceptance:** all six checks in `06_ROBOT_AND_SIMULATION_SPEC.md` §7
pass, committed as `tests/python/test_allegro_env_sanity.py`.

## Phase 3 — Analytical/inverse dynamics validation (not implementation)

**Tasks:** call `mj_fullM`, `mj_rne`, `mj_jacSite`, `mj_inverse` on the
modified model (`04_MATH_SPEC.md` §3–4); build the torque-actuated
computed-torque baseline controller for the classical-control comparison.

**Acceptance:** on random feasible states, the relative residual
$\lVert \tau_{ID} - \tau_{applied}\rVert / (\lVert \tau_{applied}\rVert +
\epsilon)$ is below a small tolerance (this validates *our understanding
of the model*, since $\tau_{ID}$ and $\tau_{applied}$ both come from
MuJoCo — a large residual means an XML/units/convention bug, not a
dynamics-modeling error, and must be found before proceeding); computed-
torque tracking on a reference trajectory achieves low RMSE with no
joint-limit violations across multiple seeds.

## Phase 4 — Demonstration pipeline (Tier B)

**Tasks:** implement `python/data/` loaders for DexYCB (primary) and
ARCTIC/GRAB (diversity); implement the smoothing/differentiation pipeline
(`08_DATA_AND_RETARGETING_PIPELINE.md` §2); wire up `dex-retargeting`
calls (§3); generate visualizations (human trajectory vs. retargeted
Allegro trajectory vs. object trajectory vs. contact points).

**Acceptance:** at least 100 valid trajectories processed automatically
end-to-end (dataset → retargeted `.npz` demo file); retargeted
trajectories replay open-loop in the Phase 2 environment without
joint-limit violations or divergence for at least 90% of processed clips.

## Phase 5 — Demonstration action labels

**Tasks:** compute `a_demo` (normalized target-position deltas, not
$\tau_{ID}$, per `04_MATH_SPEC.md` §4) for every retargeted trajectory;
store per the schema in `08_DATA_AND_RETARGETING_PIPELINE.md` §5.

**Acceptance:** replaying a demo's `a_demo` sequence open-loop through the
Phase 2 environment reproduces the retargeted `q` trajectory within a
small tracking error (this validates the action-label conversion is
self-consistent before it's used for BC).

## Phase 6 — Dynamics ensemble (black-box and residual)

**Tasks:** implement `python/dynamics/` per `04_MATH_SPEC.md` §5 and
`05_RL_ALGORITHM_SPEC.md` §3; collect randomized-rollout training data
(randomize mass/friction/actuator gain/external perturbations, per
`06_ROBOT_AND_SIMULATION_SPEC.md` §4).

**Acceptance:** one-step prediction beats a zero-residual baseline;
multi-step rollout ($k \in \{1,5,10,20,50\}$) errors are reported and
don't diverge catastrophically at $k=50$; the residual variant's error is
lower than the black-box variant's under the domain-randomized evaluation
condition (this is the first quantitative signal that "physics structure"
is doing something, ahead of the full RL ablation).

## Phase 7 — Tier B RL, Condition A (reach, then pickup, from scratch)

**Tasks:** implement `python/envs/allegro_env.py` fully (reward per
`04_MATH_SPEC.md` §1); first validate a simple reach sub-task (wrist to
target, $SR>90\%$, matching the spirit of the original plan's staged
difficulty), then the full pickup task, both with plain SAC (Condition A).

**Acceptance:** reach task $SR>90\%$ before attempting pickup (kept from
original plan §7's staged-difficulty instruction); pickup Condition A
achieves a non-trivial, reproducible success rate across ≥3 seeds — even a
modest one is fine, since Condition A is the *baseline* the other
conditions are compared against.

## Phase 8 — Tier B RL, Conditions B–E (the ablation matrix)

**Tasks:** run all five conditions (`05_RL_ALGORITHM_SPEC.md` §4) with
matched compute budgets (same total env steps, same seed count ≥3–5
per condition), on the Phase-4/5 retargeted demonstrations.

**Acceptance:** sample-efficiency curves (success rate vs. environment
transitions) for all five conditions, plotted with confidence intervals,
per `10_EXPERIMENTS_AND_EVALUATION.md`. This is the primary experimental
deliverable of the project.

## Phase 9 — Robustness and generalization

**Tasks:** domain-randomized evaluation (`06_ROBOT_AND_SIMULATION_SPEC.md`
§4); object-level held-out generalization split
(`08_DATA_AND_RETARGETING_PIPELINE.md` §6).

**Acceptance:** robustness and generalization plots per
`10_EXPERIMENTS_AND_EVALUATION.md`, for at least the best-performing and
the from-scratch (Condition A) policies.

## Phase 10 — Export and production serving

**Tasks:** ONNX export + parity check (`python/export/`); Rust inference
server (`rust/inference_server/`), per `11_PRODUCTIONIZATION.md`.

**Acceptance:** the Rust server, given no PyTorch/CUDA dependency at all,
loads `policy.onnx` and returns a control output for a sample state within
the latency budget in `11_PRODUCTIONIZATION.md`, matching the Python-side
ONNX Runtime output to numerical tolerance.

## Phase 11 (stretch) — Shadow Hand, live HaMeR demo, Rust env stepper

Only attempted after Phase 9 is complete, in this priority order:
1. Live HaMeR demo (record your own hand, retarget, replay) — best
   resume/demo value for the least engineering risk.
2. Rust-native parallel env stepper (Tier 2, `07_SYSTEM_ARCHITECTURE.md`),
   only if Phase 7/8's Python-parallelism throughput was profiled as a
   real bottleneck.
3. Shadow Hand (24 DoF) repeat of Phases 2–8 — the most expensive stretch
   item, attempt last.
