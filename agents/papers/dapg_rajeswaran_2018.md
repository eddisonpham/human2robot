# DAPG — Demo Augmented Policy Gradient

**Citation:** Rajeswaran, A., Kumar, V., Gupta, A., Vezzani, G., Schulman,
J., Todorov, E., Levine, S. "Learning Complex Dexterous Manipulation with
Deep Reinforcement Learning and Demonstrations." RSS 2018.

**Link:** https://arxiv.org/abs/1709.10087

**Related repo:** original codebase is `aravindr93/hand_dapg` /
`aravindr93/mjrl` (Adroit-hand environments and demos from this codebase
are what Gymnasium-Robotics/Minari now serve in maintained form — see
`03_EXISTING_REPOS_TO_CLONE.md`, Tier A).

## Summary

DAPG shows that a small number of human demonstrations, combined with RL,
can solve dexterous manipulation tasks (object relocation, opening a door,
using a hammer, twirling a pen) with a 24–30 DoF Adroit hand — tasks that
RL from scratch struggles with. The algorithm has two pieces: (1) behavior
cloning pretraining of the policy on demonstrations, and (2) an
augmented **Natural Policy Gradient (NPG)** update where the standard
policy-gradient term is summed with a decaying, demonstration-weighted
imitation term.

## Important correction for this project

**DAPG's core algorithm is on-policy (NPG), not built on an off-policy
actor-critic like SAC.** The original brain-dump plan for this project
asked for "DAPG, then continue with SAC," which conflates two different
algorithm families' demo-integration mechanisms. We do not implement
literal DAPG. We cite it for **why** demonstrations help dexterous
manipulation (strong empirical motivation, same problem class as this
project) and implement the standard, correct off-policy analog instead —
see `demos_sparse_reward_vecerik_2017.md` and `05_RL_ALGORITHM_SPEC.md` §2.
See also `12_NON_GOALS_AND_CUT_SCOPE.md` §7.

## Key equation (NPG + demo augmentation, for reference only — not implemented as-is)

$$
g_{aug} = \sum_{(s,a)\in\rho_{RL}} \nabla_\theta \log \pi_\theta(a|s) A^\pi(s,a)
\;+\; \lambda_0 \lambda_1^{k} \sum_{(s,a)\in\rho_{demo}} \nabla_\theta \log \pi_\theta(a|s)
$$

with $\lambda_1^k$ decaying the demonstration term's weight over training
iteration $k$ — the same qualitative idea (demonstrations guide early
exploration, task reward dominates later) that our annealed demo-ratio
sampling (`05_RL_ALGORITHM_SPEC.md` §2, step 3) implements via a different,
off-policy-appropriate mechanism.
