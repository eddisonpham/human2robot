# Executability fails on the thumb, and the cause is upstream of the optimizer

## The observation

Only **34 of 100** optimized subject-01 trajectories are executable when
replayed open-loop in the simulated Allegro hand
(`results/trajectory_optimization/feasibility_dexycb.json`). That number has no
remedy attached to it, and the obvious one, moving more slowly, barely helps.

## What it is not

Each probe below rules out one explanation. All are measurements, not
reasoning, and the scripts are short enough to re-run.

**Not velocity or control lag.** Replaying the same path at 0.15x speed (6.7x
slower) raises executability from 34/100 to 48/100 and drops mean drift only
from 0.398 to 0.355 rad. A tracking-lag problem would be strongly
speed-dependent. It is not.

| Speed | Executable | Mean drift | Mean max velocity |
| --- | --- | --- | --- |
| 1.00x | 34/100 | 0.398 | 2.00 |
| 0.70x | 38/100 | 0.383 | 1.52 |
| 0.50x | 41/100 | 0.370 | 1.12 |
| 0.35x | 41/100 | 0.372 | 0.79 |
| 0.25x | 41/100 | 0.367 | 0.58 |
| 0.15x | 48/100 | 0.355 | 0.35 |

**Not the three long fingers.** Freezing one finger at its start pose and
replaying the rest:

| Frozen | Executable | Mean drift |
| --- | --- | --- |
| nothing (baseline) | 34/100 | 0.398 |
| **thumb** | **63/100** | **0.300** |
| index | 34/100 | 0.414 |
| middle | 29/100 | 0.410 |
| ring | 38/100 | 0.387 |

**Not collision between the thumb and the other fingers.** Freezing the thumb
collapses thumb drift from 0.264 to 0.051 rad and leaves the other twelve
joints untouched at 0.180 to 0.177 rad. If the thumb were shoving the fingers
aside, their drift would fall too. It does not.

So the thumb fails on its own, slowly, and independently.

## What it is

**The commanded trajectories saturate both joint limits on all sixteen
joints.** Measured across all 100 optimized subject-01 sequences, every joint's
commanded minimum equals the actuator lower bound and every maximum equals the
upper bound, to three decimal places:

| Joint | Range | Commanded span |
| --- | --- | --- |
| index/middle/ring 1-4 | -0.470 to 1.709 | spans the full range |
| thumb 1 | 0.263 to 1.396 | spans the full range |
| thumb 2-4 | -0.105 to 1.719 | spans the full range |

The retargeting stage is asking for configurations the hand cannot reach, and
something downstream clips them. The clipping is invisible to the pipeline's own
metrics: `sustained_limit_violation` is **0.0000** on every arm of every data
set, because by the time it is measured the values are already in bounds. The
metric confirms compliance and certifies the clipping.

## Why the optimizer cannot fix it

`TrajectoryOptimizer` receives a trajectory that has already been clipped, and
smooths it. It cannot recover the motion the clipper discarded, because that
information is gone. This is consistent with the numbers already published: jerk
falls 16.8 percent and smoothness 30.4 percent, both real improvements, while
executability stays at 34/100. The optimizer is doing its job on an input whose
amplitude is wrong.

It is also consistent with the optimizer's configuration, which sets
`collision=0.0` and weights tracking lightly against smoothness. None of that
is the proximate cause here, but it means nothing upstream or downstream of the
optimizer is currently checking whether the *command* is reachable.

## The prediction this makes

A falsifiable one, which is the point of writing this down:

- If thumb saturation is the cause, **fixing the retargeting for the thumb
  should move executability well past 63/100**, and the residual failure should
  become speed-dependent, so that slowing down starts to help. It currently
  does not.
- If executability stays near 34/100 after a correct thumb retargeting, this
  explanation is wrong and the cause is elsewhere in the retargeting or in the
  hand model.

## What is not established

**The mechanism by which the thumb saturates is not yet diagnosed.** Candidates
that remain open, in rough order of prior probability:

1. The human thumb opposes across the palm; the Allegro thumb largely does not,
   and `thb1` cannot go below 0.263 rad. A human opposition motion maps to an
   unreachable request and gets clipped.
2. The IK stage produces thumb targets that are clipped by the retargeter
   rather than by the hand's range, which would put the fault even further
   upstream.
3. The floating hand model used for replay is a crude position-servo
   approximation (sixteen actuators, unit gain, three identical fingers) and its
   thumb may simply be weaker than the real device.

Distinguishing these needs a per-joint, per-subject count of how often each
joint reaches a bound, which has not been run.

## Why this is worth fixing rather than documenting

The current pipeline reports limit violation as perfect while silently
discarding human motion at the joint range. Any executability number computed
downstream of that clipping is a measurement of the clipper. That is the same
class of defect as the earlier ones in this project, where a metric faithfully
reported the wrong thing.
