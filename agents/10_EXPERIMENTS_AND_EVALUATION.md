# 10 — Experiments and Evaluation

## 1. Metrics (kept from original plan §23, already standard — do not just report final reward)

| Metric | Definition |
|---|---|
| Success rate | $SR = \#\text{successful episodes} / \#\text{episodes}$ |
| Sample efficiency | $SR$ vs. environment transitions (the primary plot for the ablation) |
| Tracking error | $RMSE_q = \sqrt{\frac1{TN}\sum_t \lVert e_q(t)\rVert^2}$, $e_q = q_{ref}-q$ (Phase 3 controller baseline) |
| Object position error | $E_o = \lVert p_o - p_{target}\rVert$ |
| Energy | $E_\tau = \sum_t \lVert \tau_t\rVert^2 \Delta t$ (reads `actuator_force`, works under position control) |
| Dynamics multi-step error | $E_k = \frac1T\sum_t \lVert \hat x_{t+k}-x_{t+k}\rVert_2$, $k\in\{1,5,10,20,50\}$ |
| Robustness | $SR$ under domain randomization (`06_ROBOT_AND_SIMULATION_SPEC.md` §4) |
| Generalization | $SR$ on held-out object identities |

## 2. The primary experiment: the five-condition ablation (Phase 8)

Report, for Conditions A–E (`05_RL_ALGORITHM_SPEC.md` §4), on the *same*
Tier B pickup task, with matched compute budgets and ≥3–5 seeds each:

- Sample-efficiency curves ($SR$ vs. transitions), mean ± confidence band
  across seeds — this is the figure that answers the core research
  question from `01_MISSION_AND_SCOPE.md`.
- Final $SR$, $E_\tau$, and wall-clock training time, in one summary table.
- A one-paragraph, honest interpretation. **If the ablation shows no
  measurable benefit from demonstrations or physics-structured dynamics
  on this task, report that.** A clean negative or null result, correctly
  measured across matched seeds, is a legitimate and defensible finding
  for a portfolio project — it is much better than a confounded
  "positive" result from mismatched compute budgets between conditions.

## 3. Secondary experiments

- **Robustness (Phase 9):** $SR$ under domain randomization, for
  Condition A vs. the best-performing condition from Phase 8.
- **Generalization (Phase 9):** $SR$ on held-out object identities,
  same comparison.
- **Dynamics-model quality (Phase 6):** multi-step rollout error,
  black-box vs. physics+residual, standalone (not policy-mediated) —
  this is useful as an early, cheap signal before running the full,
  expensive RL ablation.

## 4. Required visualizations (kept from original plan §42, already a good standard list)

1. Human trajectory vs. retargeted robot trajectory.
2. Joint / velocity / acceleration trajectories.
3. Inverse-dynamics torques (Phase 3 baseline).
4. Residual dynamics error, by horizon $k$.
5. RL learning curves, all five conditions overlaid.
6. Success rate vs. environment steps (the primary ablation figure).
7. Energy vs. success (does the trained policy find an efficient or a
   wasteful grasp strategy?).
8. Pickup trajectory in 3D.
9. Contact events over time.
10. Robustness and generalization bar/line plots.

All plots carry units and, where multiple seeds exist, confidence
intervals — no single-seed learning curve is reported as if it were a
stable estimate.

## 5. Reproducibility manifest — every experiment logs

```text
seed, git_commit, config.yaml, dataset_split, robot_model,
simulation_timestep, control_decimation, number_of_parallel_envs,
optimizer, learning_rate, batch_size, training_steps, hardware,
wall_clock_time
```

saved as `results/<experiment_id>/{config.yaml, metrics.json,
checkpoint.pt, git_commit.txt, system_info.json}`. Every experiment is
rerunnable with one command:
`python -m python.rl.train --config configs/ablation_e.yaml --seed 3`.
See `13_REPRODUCIBILITY_AND_CONVENTIONS.md` for the full schema.

## 6. Report structure (kept from original plan §43 — already a standard, correct research-report skeleton)

Abstract → Introduction (problem statement, the boxed research question
from `01_MISSION_AND_SCOPE.md`) → Method (§ retargeting, § MDP/reward, §
SAC, § demo integration, § MBPO-style dynamics augmentation — cite
`04_MATH_SPEC.md`/`05_RL_ALGORITHM_SPEC.md` sections directly) →
Experiments (the ablation design, §2 above) → Results → Limitations
(simulator fidelity, retargeting quality, imperfect contact labels, model
bias, no physical-robot validation — unchanged from original plan, still
the right list) → Future Work (point to
`12_NON_GOALS_AND_CUT_SCOPE.md` — HJB/SDE, Hamiltonian dynamics,
differentiable physics, Shadow Hand, sim-to-real).
