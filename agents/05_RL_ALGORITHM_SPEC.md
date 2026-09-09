# 05 — RL Algorithm Specification

One trainer, one algorithm family (SAC), extended by two orthogonal,
independently-toggleable feature flags (demonstrations, model-based
rollout augmentation). This file is the algorithmic core of the whole
project — read it fully before writing `python/rl/`.

## 1. Base algorithm: Soft Actor-Critic (standard, off-policy)

Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
with a Stochastic Actor" (2018) — see `papers/sac_haarnoja_2018.md`. Fork
`references/cleanrl/cleanrl/sac_continuous_action.py` as
`python/rl/sac.py` and keep it recognizably SAC: don't invent variants.

**Networks**
- Actor: tanh-squashed diagonal Gaussian policy, MLP `[256, 256]`, ReLU.
  Outputs $(\mu(x), \log\sigma(x))$ per action dimension.
- Critic: twin Q-networks (double-Q, standard SAC), each MLP `[256, 256]`,
  ReLU, both take $(x, u)$ concatenated as input.
- Target critics: Polyak-averaged copies, $\tau = 0.005$.

**Hyperparameters (defaults — override per-experiment in
`configs/*.yaml`, never hardcode)**

| Hyperparameter | Default | Notes |
|---|---|---|
| Learning rate (actor, critic, $\alpha$) | 3e-4, Adam | standard |
| Discount $\gamma$ | 0.99 | |
| Polyak $\tau$ | 0.005 | |
| Replay buffer size | 1,000,000 | ~a few hundred MB at our state dim — CPU RAM only |
| Batch size | 256 | |
| Warmup / random-exploration steps | 5,000–10,000 | uniform random actions before policy-driven collection starts |
| Target entropy | $-|\mathcal A|$ | standard heuristic; automatic temperature tuning, not a fixed $\alpha$ |
| Gradient (update-to-data, UTD) steps per env step | 1 (default), tunable | higher UTD is a documented, optional sample-efficiency lever, not required for the MVP |
| Episode horizon $T$ | 400–600 control steps | see `06_ROBOT_AND_SIMULATION_SPEC.md` for control rate |

Automatic entropy-temperature tuning (the standard modern SAC variant,
Haarnoja et al.'s follow-up), not a fixed manually-tuned $\alpha$ — this
avoids a fragile, task-specific hyperparameter and is what CleanRL's
reference implementation already does.

## 2. Demonstration integration — standard, and *not* literal DAPG

**Important correction versus the original plan.** The original plan cited
DAPG (Rajeswaran et al. 2018) and then said "pretrain with BC, then
continue with SAC" — but DAPG's actual algorithm augments **Natural Policy
Gradient** (on-policy) with a demonstration-augmented loss term; it is not
built on an off-policy actor-critic like SAC. Mixing the two without care
would be a real math error. Instead, we implement the standard, correct
way to combine demonstrations with an **off-policy** actor-critic,
following Večerík et al. 2017 ("Leveraging Demonstrations for Deep RL on
Robotics Problems with Sparse Rewards") and Nair et al. 2018 ("Overcoming
Exploration in RL from Demonstrations") — see
`papers/demos_sparse_reward_vecerik_2017.md`. DAPG (and DexMV, which uses
it) remains the correct citation for *why* demonstrations help dexterous
manipulation and for architectural inspiration; it is not the literal
algorithm we run.

**The algorithm, precisely:**

1. **BC pretraining.** Before any RL, fit the actor's mean head by
   supervised regression on demonstration (state, action) pairs:
   $$L_{BC} = \frac1T \sum_t \lVert \pi_\phi(o_t) - a_t^{demo} \rVert^2$$
   where $a_t^{demo}$ is the retargeted/dataset target joint position,
   normalized into $[-1,1]^N$ (see `04_MATH_SPEC.md` §4 — **not**
   $\tau_{ID}$). Train for a fixed number of epochs (default 100, tune per
   run) with early stopping on a held-out demo split.
2. **Replay-buffer seeding.** Replay the demonstration trajectories
   through the *actual* environment (not just stored as static data) to
   generate real $(x_t, a_t, x_{t+1}, r_t)$ transitions, and insert these
   into the SAC replay buffer before online collection starts.
3. **Demo-ratio sampling with annealing.** Each SAC minibatch samples a
   fraction $\rho_t$ from the demo transitions and $1-\rho_t$ from online
   experience; $\rho_t$ starts at 0.5 and linearly anneals to 0 over the
   first ~30% of training. This, not a reward-shaping term, is the
   mechanism that "guides exploration without forcing exact imitation" —
   the original plan's intent in §19 is preserved, just via the standard
   off-policy mechanism instead of an ad hoc extra reward term. (An
   optional trajectory-prior reward term, $r_{demo} = -\lambda_d\lVert
   q_t - q_t^{demo}\rVert^2$, may still be added as a documented ablation
   knob — keep $\lambda_d$ small and log it, per the original plan's own
   warning not to let it dominate the task reward.)

Condition **B** in the ablation matrix (§5 below) = steps 1–3 on top of
plain SAC. Condition **A** = plain SAC, steps 1–3 disabled.

## 3. Model-based dynamics augmentation — MBPO, not ad hoc

**Second correction versus the original plan.** "SAC + learned dynamics"
and "SAC + physics + residual dynamics" (original §22, conditions C/D)
were left algorithmically underspecified. The standard, well-cited,
correct way to inject a learned/residual dynamics model into an
**off-policy** actor-critic is **Model-Based Policy Optimization (MBPO)**,
Janner et al., NeurIPS 2019 — see `papers/mbpo_janner_2019.md`. We adopt
it directly, simplified in scale (not in mechanism) for our compute
budget:

1. Maintain an ensemble of $E=3$–$5$ dynamics models
   $\{\hat f_\theta^{(i)}\}$, each the architecture in
   `04_MATH_SPEC.md` §5, predicting $\Delta x$ (state delta) from
   $(x, u)$.
   - **Condition C (black-box):** trained purely on interaction data, no
     physics prior.
   - **Condition D (physics + residual):** $\hat f = f_{physics} +
     r_\theta$ — i.e. step the real MuJoCo model to get
     $f_{physics}(x,u)$, and only learn the residual correction
     $r_\theta$. To make the residual non-trivial (otherwise there is
     nothing to correct, since our "physics" model *is* the real
     simulator), Condition D is evaluated under **domain-randomized
     dynamics** at rollout time (randomized object mass/friction and
     actuator gains per episode, `06_ROBOT_AND_SIMULATION_SPEC.md`),
     while the *nominal* physics model used inside $f_{physics}$ uses
     fixed default parameters — so $r_\theta$ is literally learning to
     correct a sim-to-real-style gap from interaction data. This is the
     precise, standard framing of Johannink et al.'s residual-RL idea
     (`papers/residual_rl_johannink_2019.md`), not a vague "physics +
     neural net" mixture.
2. Retrain the ensemble on all real data collected so far every ~250 real
   env steps (standard MBPO cadence, tune per compute budget).
3. At each policy-update round, generate short synthetic rollouts (default
   horizon $k=1$, documented as an easy extension to the paper's
   epoch-scheduled longer horizon) by branching from real states sampled
   from the replay buffer, using a randomly-selected ensemble member.
   Insert synthetic transitions into a short-lived model buffer.
4. SAC minibatches draw a configurable mix of real and synthetic
   transitions (default: up to 50% synthetic — deliberately more
   conservative than MBPO's own ≥90% figure, because our ensemble and
   compute budget are smaller; this is a documented, honest scaling-down,
   not a reproduction of the paper's exact numbers, and should be
   validated empirically, not assumed).

## 4. The five ablation conditions (original plan §22, now precisely defined)

All five conditions run the **same** trainer (`python/rl/sac.py`) with
different config flags — this is a deliberate software-engineering choice
so that differences in results are attributable to the ablated factor, not
to different code paths.

| Condition | BC-init + demo buffer (§2) | Dynamics-model rollout augmentation (§3) |
|---|---|---|
| A — RL from scratch | off | off |
| B — Demo-guided RL | **on** | off |
| C — Black-box learned dynamics | off | **on**, black-box ensemble |
| D — Physics + residual dynamics | off | **on**, physics+residual ensemble |
| E — Full system | **on** | **on**, physics+residual ensemble |

This is a clean 2-factor (demos × dynamics-model-type) design with a
shared no-augmentation baseline (A) and a "kitchen sink" condition (E),
which directly answers the original research question without inventing a
combinatorially larger, harder-to-interpret experiment matrix.

## 5. Debugging / correctness practice

- Run Condition A on **Tier A** (`AdroitHandRelocate-v1`) first, cross-
  checked against Stable-Baselines3's SAC on the same env (see
  `03_EXISTING_REPOS_TO_CLONE.md`). If ours and SB3's learning curves
  diverge wildly, the bug is in our SAC code, not the environment.
- Only after Condition A validates on Tier A do we move to Tier B and the
  full A–E matrix, per `09_PHASE_PLAN.md`.
- Log every hyperparameter, git commit, and seed per run — see
  `13_REPRODUCIBILITY_AND_CONVENTIONS.md`.
