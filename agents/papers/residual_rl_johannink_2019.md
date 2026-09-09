# Residual Reinforcement Learning for Robot Control

**Citation:** Johannink, T., Bahl, S., Nair, A., Luo, J., Kumar, A.,
Loskyll, M., Ojea, J. A., Solowjow, E., Levine, S. "Residual
Reinforcement Learning for Robot Control." ICRA 2019.

**Link:** https://arxiv.org/abs/1812.03201

## Summary

The paper's core idea: combine a conventional model/controller with a
small learned residual that only has to correct what the conventional
model gets wrong (contact/friction effects the analytic model can't
capture), rather than learning the full control/dynamics problem from
scratch. This gives the learned component a much smaller, better-shaped
problem to solve, and inherits the conventional model's stability and
sample efficiency where it's already accurate.

## What we borrow

The **framing**, applied to dynamics modeling rather than control
directly: $\hat f = f_{physics} + r_\theta$
(`04_MATH_SPEC.md` §5). $f_{physics}$ is MuJoCo's own step function (exact,
free, no learning needed for the nominal dynamics); $r_\theta$ only has to
learn the *gap* between the nominal simulator and whatever the "true"
dynamics are assumed to be for that experiment.

## Why this needs a deliberately introduced gap (important implementation note)

If $f_{physics}$ is literally the same simulator used to evaluate the
policy, there is nothing for $r_\theta$ to correct — the residual would
learn to be near-zero, and Condition D would collapse into "Condition C
with extra steps." To make the residual meaningful and the ablation
informative, **Condition D is evaluated under domain-randomized dynamics**
(randomized object mass/friction/actuator gain per episode,
`06_ROBOT_AND_SIMULATION_SPEC.md` §4) while $f_{physics}$ inside the
dynamics-model ensemble uses fixed nominal parameters. $r_\theta$ then has
a genuine, non-trivial job: learning a sim-to-real-style correction from
interaction data, exactly the situation this paper's residual-RL framing
was designed for. This detail is specified precisely in
`05_RL_ALGORITHM_SPEC.md` §3 — implement it exactly as described there,
not as a same-parameters residual, or the ablation won't measure anything.
