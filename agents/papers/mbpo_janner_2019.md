# MBPO — Model-Based Policy Optimization

**Citation:** Janner, M., Fu, J., Zhang, M., Levine, S. "When to Trust
Your Model: Model-Based Policy Optimization." NeurIPS 2019.

**Link:** https://arxiv.org/abs/1906.08253

## Summary

MBPO trains an **ensemble** of learned dynamics models on real
environment data, then uses that ensemble to generate **short, branched
synthetic rollouts** starting from real states sampled out of the replay
buffer. These synthetic transitions are added to the same replay buffer
used by an off-policy actor-critic (the paper uses SAC directly), so the
policy/critic effectively gets many more (model-generated) gradient
updates per real environment step than a purely model-free method would.
Keeping the synthetic rollout horizon short is the paper's key insight for
controlling compounding model error — it's explicitly *not* full-episode
imagined rollouts (that would be closer to Dreamer-style world-model RL,
a different and heavier technique we don't use).

## What we borrow

This is the precise algorithmic basis for `05_RL_ALGORITHM_SPEC.md` §3
(Conditions C and D): a small ensemble of MLP dynamics models
(architecture from `04_MATH_SPEC.md` §5), periodic retraining on all real
data, short-horizon branched synthetic rollouts inserted into the SAC
replay buffer, and a real/synthetic mixing ratio for each training
minibatch. Our ensemble size (3–5) and synthetic ratio (up to 50%) are
**deliberately scaled down** from the paper's own figures (which use
larger ensembles and much higher synthetic ratios, tuned for larger-scale
benchmarks) to fit this project's compute budget — this is a documented,
honest simplification, not a claim of reproducing the paper's exact
numbers; the mixing ratio should be tuned empirically in Phase 8, not
assumed.

## Key mechanism

$$
\hat f_\theta^{(i)}(x,u) \to \Delta x, \quad i=1,\dots,E \qquad \text{(ensemble, retrained every ~250 real steps)}
$$
$$
x_{t+1}^{synthetic} = x_t^{real,sampled} + \hat f_\theta^{(i)}(x_t, \pi_\phi(x_t)), \quad k \text{ steps}, \ k \text{ small}
$$
$$
\mathcal B_{SAC} = (1-\beta)\cdot \mathcal B_{real} + \beta \cdot \mathcal B_{synthetic}
$$

## How Condition D (residual) differs from Condition C (black-box)

Condition C's $\hat f_\theta$ is a pure black-box MLP with no physics
prior. Condition D's $\hat f_\theta = f_{physics} + r_\theta$
(`04_MATH_SPEC.md` §5) — the nominal step comes from the real MuJoCo
model, and only the residual is learned. See
`residual_rl_johannink_2019.md` for the precise motivation and framing of
that residual term.
