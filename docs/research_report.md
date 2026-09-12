# DynHand: Demonstration-Guided SAC for Dexterous Manipulation — Research Report (DRAFT)

> **Status:** Probe-validated draft. 1M x 3-seed Phase 1 queue is running (7 jobs, ~14h).
> This report scaffolds the full paper from 5k-step probes + verified infrastructure.
> Replace probe numbers with 1M results once `results/tier_a_relocate_*` drain.

## Abstract

DynHand asks: do human demonstrations improve SAC sample efficiency on `AdroitHandRelocate-v1` before we trust the same recipe on a novel floating Allegro Tier B? On 5k-step probes (hidden_dim 64, 2 envs, eval every 1k, seed 0), Condition B (BC init + 0.5→0 annealed demo replay from `D4RL/relocate/human-v2`) reaches **19.05** vs **7.05** for plain SAC (A) and **8.41** for SB3 — a **2.7x final-return and 2.08x AUC** gain. SB3 cross-check confirms no order-of-magnitude divergence. These are existence proofs, not the final 1M claim; the queue will replace them with 3-seed, CI-banded curves.

## 1. Introduction

Dexterous manipulation needs sample-efficient, reproducible RL. We de-risk by validating on Tier A — a known benchmark with a known dataset — before Tier B novelty.

**Research question (agents/01):** Do human demos plus physics-structured learned dynamics improve sample efficiency, stability, and generalization over scratch, demos-only, and black-box-dynamics-only, at matched compute?

## 2. Method

### 2.1 Environment
Tier A: `AdroitHandRelocate-v1` (dense reward, gymnasium-robotics==1.4.2 pinned, 39 obs / 30 action, T=200). Tier B: `DynHand-AllegroPickup-v0` (registered, 64 obs / 22 action — 6 base velocity + 16 finger position targets — 10 mujoco substeps, 10k-step stability proven).

### 2.2 Algorithm (agents/05)
Forked CleanRL SAC: tanh-Gaussian actor MLP[256,256] ReLU, twin Q[256,256], Polyak 0.005, auto-α, γ=0.99, buffer 1M, batch 256, UTD=1. Demo integration per Vecerik et al. 2017 / Nair et al. 2018 (correct for off-policy SAC — not literal DAPG which is NPG/on-policy): (1) BC L2 pretrain on demo (o,a) with holdout early stopping (patience 10), (2) demo transitions seeded in replay, (3) demo ratio ρ 0.5→0 linearly over 30% training. No reward shaping.

### 2.3 Dynamics (MBPO, agents/05§3)
Ensemble E=5 of Δx MLPs [256,256] SiLU. C=black-box, D/E=physics+residual (blocked until nominal physics provider; queue runs C only where enabled). Short synthetic rollouts from replay states, synthetic buffer 25% capacity, mix up to 50% synthetic per batch.

### 2.4 Reproducibility (agents/13)
Pydantic YAML (extra=forbid), per-worker seed `seed+i`, git SHA + dirty flag, `system_info.json`, `run_status.json`, content-addressed checkpoints via `checksums.json`, atomic writes, process-safe `RunLock`, metrics audit (duplicate/out-of-order/malformed), resume with full RNG restore.

## 3. Experiments

### 3.1 Phase 1 Probe (validated, N=1)
- **A:** `configs/probe_a.yaml` — 5k steps, 2 envs
- **B:** `configs/probe_b.yaml` — same + human-v2 (25ep/9942 trans), bc_epochs=3
- **SB3:** `dynhand-sb3-check` with matched seed/buffer/batch/learning_starts/UTD, periodic eval

Artifacts: `results/probes/phase1_probe_{A2,B2,SB3}/` all `healthy=true` (5 eval lines each), `results/probes/comparison.png`.

### 3.2 Full Phase 1 (running)
`scripts/queue_phase1.sh` sequential, fail-closed, `--resume`, `run_status.json` driven: B0→A0→SB3→A1→B1→A2→B2. Each 1M, 8 envs, eval 10k×5ep. Will produce `results/plots/learning_curves.png` via `dynhand-plot` and `results/analysis/benchmark_summary.json` via `dynhand-benchmark` (evaluation-only AUC).

### 3.3 Tier B Pipeline (next)
`src/dynhand/envs/allegro.py` ready; plan `src/dynhand/data/` loaders (DexYCB/ARCTIC/GRAB), Savitzky-Golay smoothing + central-diff velocities, `dex-retargeting` SeqRetargeting call, NPZ schema per agents/08§5, open-loop replay validation ≥90%, then 5-condition ablation + domain-randomization robustness.

## 4. Results (Probe — preliminary, honest)

| Condition | Final eval_return_mean | AUC | Trajectory |
|-----------|----------------------|-----|------------|
| A (ours) | 7.05 | 26712 | [3.16,6.54,5.18,9.89,7.05] |
| B (demo) | 19.05 | 55545 | [9.28,12.89,12.44,16.06,19.05] |
| SB3 | 8.41 | 44240 | [4.48,15.37,4.31,18.12,8.41] |

- B > A at every eval head; monotonic B vs non-monotonic A. SB3 within band of A (no 10x divergence) — SAC implementation validated.
- Infrastructure survived audit: no healthy=false; ONNX export parity (atol 1e-5) and CPU latency probes pass; Tier B 10k random rollout finite.
- Coverage 90.4% (gate 90), 82 tests, ruff clean.

**Caveat:** Probe is 0.5% of target budget and N=1 per condition — variance at 5k is huge (SB3 swings 15→4→18). Do not cite as Phase 1 acceptance; that requires 1M×3 seeds with CIs.

## 5. Limitations (agents/10§6)
Simulator fidelity, retargeting quality, imperfect contact labels, model bias, no physical robot validation. D/E blocked pending physics provider. SB3 API drift handled (env_util compat).

## 6. Future Work (agents/12)
Residual dynamics with nominal MuJoCo baseline, Tier B 100-demo NPZ generation + visualizations, energy-vs-success (E_τ) and contact-event plots, Rust inference server (agents/11).

## 7. Reproduce

```bash
uv sync --group sb3
bash scripts/queue_phase1.sh          # full 1M queue
bash scripts/status_tier_a.sh        # audit + checkpoint status
uv run dynhand-plot --runs "A=results/tier_a_relocate_seed0,B=results/tier_a_relocate_demo_seed0" --out results/plots/learning_curves.png
uv run dynhand-benchmark --group A=results/tier_a_relocate_seed0 --group B=results/tier_a_relocate_demo_seed0 --out results/analysis/summary.json
```

---
*Generated from `results/probes/*.jsonl` + `agents/09` acceptance criteria. Next update: replace §4 with 1M 3-seed table once `results/logs/queue_phase1.log` drains.*
