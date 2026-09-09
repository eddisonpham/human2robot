# Soft Actor-Critic (SAC)

**Citation:** Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. "Soft
Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a
Stochastic Actor." ICML 2018.

**Link:** https://arxiv.org/abs/1801.01290

**Related repo:** `references/cleanrl/cleanrl/sac_continuous_action.py`
(forked as `python/rl/sac.py`, see `03_EXISTING_REPOS_TO_CLONE.md`).

## Summary

SAC is an off-policy actor-critic algorithm for continuous action spaces
that maximizes a maximum-entropy objective
$\mathbb E[\sum_t r_t + \alpha \mathcal H(\pi(\cdot|s_t))]$, using twin
Q-function critics (to reduce overestimation bias, following the Double-Q
idea) and a stochastic, tanh-squashed Gaussian policy. A widely-used later
refinement (also implemented in CleanRL and adopted here) replaces the
fixed entropy temperature $\alpha$ with automatic dual-gradient tuning
against a target entropy, removing a fragile manual hyperparameter.

## What we borrow

The entire base algorithm, verbatim in spirit: twin critics, target
networks with Polyak averaging, automatic temperature tuning, replay
buffer, off-policy updates. See `05_RL_ALGORITHM_SPEC.md` §1 for our exact
hyperparameters (mostly the standard defaults from the paper and from
CleanRL's tuned reference implementation).

## Key equations

Soft Q-function target:
$$
y = r + \gamma\, \mathbb E_{a'\sim\pi}\big[\min(Q_1,Q_2)(s',a') - \alpha \log \pi(a'|s')\big]
$$

Policy objective:
$$
J(\pi) = \mathbb E_{s\sim\mathcal D,\, a\sim\pi}\big[\alpha\log\pi(a|s) - \min(Q_1,Q_2)(s,a)\big]
$$

Temperature objective (automatic tuning):
$$
J(\alpha) = \mathbb E_{a\sim\pi}\big[-\alpha(\log\pi(a|s) + \bar{\mathcal H})\big], \qquad \bar{\mathcal H} = -|\mathcal A|
$$

## Why SAC specifically (not PPO/TD3/on-policy methods)

Off-policy sample efficiency matters here because environment steps are
CPU-simulation-bound (`02_TECH_STACK.md`), and SAC's replay buffer is what
makes the demonstration-integration mechanism in
`05_RL_ALGORITHM_SPEC.md` §2 (seeding + ratio-sampling from a buffer)
and the MBPO-style synthetic-rollout injection in §3 both natural,
standard fits — both techniques are specified against an off-policy
actor-critic in their original papers.
