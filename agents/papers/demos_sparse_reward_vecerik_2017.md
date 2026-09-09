# Leveraging Demonstrations for Deep RL on Robotics Problems with Sparse Rewards

**Citation:** Večerík, M., Hester, T., Scholz, J., Wang, F., Pietquin, O.,
Piot, B., Heess, N., Rothörl, T., Lampe, T., Riedmiller, M. "Leveraging
Demonstrations for Deep Reinforcement Learning on Robotics Problems with
Sparse Rewards." arXiv 2017.

**Link:** https://arxiv.org/abs/1707.08817

**See also:** Nair, A., McGrew, B., Andrychowicz, M., Zaremba, W., Abbeel,
P. "Overcoming Exploration in Reinforcement Learning with Demonstrations."
ICRA 2018. https://arxiv.org/abs/1709.10089 — same family of technique,
applied to DDPG; corroborates the approach below.

## Summary

Both papers combine a small set of demonstrations with an **off-policy**
actor-critic (DDPG in both cases; the mechanism transfers directly to SAC,
which this project uses) by (1) seeding the replay buffer with
demonstration transitions before online training starts, and (2) during
training, sampling each minibatch from a mix of demonstration and
online-collected transitions, rather than treating demonstrations as a
one-time pretraining signal only. This is the correct, standard way to
inject demonstrations into an off-policy method — as opposed to DAPG's
on-policy NPG-specific mechanism (`dapg_rajeswaran_2018.md`).

## What we borrow

This is the precise algorithmic basis for `05_RL_ALGORITHM_SPEC.md` §2:
BC pretraining of the actor, replaying demonstrations through the real
environment to generate genuine transitions for buffer-seeding, and
minibatch sampling with a demonstration ratio that anneals over training
rather than staying fixed (our own addition, motivated by the same
"guide early exploration, don't force exact imitation forever" intent the
original brain-dump plan stated, but implemented correctly for an
off-policy algorithm).

## Key mechanism

At each SAC update, draw a minibatch as
$$
\mathcal B = \mathcal B_{demo} \cup \mathcal B_{online}, \qquad |\mathcal B_{demo}| = \rho_t \cdot |\mathcal B|
$$
with $\rho_t$ annealed from an initial value toward zero
(`05_RL_ALGORITHM_SPEC.md` §2, step 3) rather than fixed for the whole
run, since a fixed nonzero demo ratio would bias the policy away from
optimizing the true task reward indefinitely.
