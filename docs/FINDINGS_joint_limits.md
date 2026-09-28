# The joint limits were wrong in four places at once

## What was wrong

Every path that needed the Allegro's joint limits had its own hand-written
copy: `optimization/pipeline.py`, `data/dexycb.py`, `scripts/retarget_dexycb_ik.py`,
and `compare_dexycb_synthetic.py`. Four copies, no shared definition, and no
test comparing any of them to the robot. They had drifted, and they had
drifted away from the MuJoCo model itself.

**15 of the 16 finger entries were incorrect, in both bounds at once.** The
copied values assumed a uniform `[-0.196, 0.196]` range per proximal joint and
placed the thumb at `[-0.8, 0]`. The model's `actuator_ctrlrange` says:

| Joint | Copied lower | Copied upper | Real lower | Real upper |
| --- | --- | --- | --- | --- |
| index proximal | -0.196 | 0.196 | -0.47 | 0.47 |
| index middle | -0.196 | 0.196 | -0.196 | 1.61 |
| index distal | -0.196 | 0.196 | -0.174 | 1.709 |
| middle proximal | -0.196 | 0.196 | -0.227 | 1.618 |
| thumb joints 1-4 | -0.8 | 0.0 | 0.263 / -0.105 / -0.189 / -0.162 | 1.396 / 1.163 / 1.644 / 1.719 |

The copied thumb upper bound was 0.0 against a real 1.719. Every trajectory the
optimizer produced held three thumb joints at or below zero, while the real
thumb needs them up to 1.72 rad. Measured against the model, optimized
trajectories violated the real limits by **0.75 radians**.

The second error is related and just as consequential. **The model has 16
actuators, not 22.** The 6 leading coordinates in the demo schema are base
motion, and nothing in this pipeline fills them, so they are pinned at zero. The
code treated all 22 as if they were actuated, which meant the optimizer was
configured with 6 degrees of freedom that could never move and whose bounds
`[0, 0]` silently pinned the base.

## Why it survived

Because every number it produced looked reasonable. A velocity table does not
reveal that a third of the joints were being held at zero, and a smoothness
cost is lower when motion is suppressed, so the defect made the optimizer look
*more* effective than it was: it was smoothing by clamping, not by optimizing.

The same blind spot is why the correction was found by hand rather than by any
metric. No experiment in this repository compares its own configuration against
the simulator it claims to target.

## The fix

`src/human2robot/data/limits.py` is now the single definition, a MuJoCo-free
mirror of the model's `actuator_ctrlrange` so retargeting can still run where
MuJoCo is not installed. `tests/python/test_invariants.py` pins the module
against a live `AllegroPickupEnv` and fails if the two ever disagree, which is
the test whose absence let four copies drift for as long as they did.

`OptimizerConfig` now also validates that its limit arrays are the right length
and that no lower bound exceeds its upper. `lower == upper` remains legal,
because the 6 pinned base coordinates depend on it.

## What it changed

Every figure below is from `results/trajectory_optimization/`, and the two
columns are the same pipeline before and after the limits were corrected.

| Quantity | Before | After |
| --- | --- | --- |
| max jerk reduction, subject-01 | -9.1% | **-16.8%** |
| max jerk reduction, subject-02 | -4.6% | **-12.1%** |
| cross-subject descent rate | 99/100 | **100/100** |
| median cost reduction, subject-02 | 37.3% | **43.0%** |
| **executable in simulation, subject-01** | **6/100** | **34/100** |
| executable in simulation, subject-02 | 0/100 | **8/100** |
| limit violation against the real model | 0.75 rad | **0.0000 rad** |

Note the direction. Correcting the limits made the headline jerk result
**worse**, from -9.1% to -16.8%, because the optimizer was previously benefiting
from motion it was achieving by clamping. Two of the three rows moved the
pipeline forward and one moved a published number backward, and the backward one
is the one that was inflated.

## The generalizable lesson

This is the second time in this project that a duplicated constant was the
defect, after the 1-DOF retargeter. In both cases the duplicated value was
*plausible*, and in both cases the symptom was a number that looked better than
it should. The check that catches this class of defect is not a metric, it is a
comparison between the configuration a pipeline is using and the system it is
configured for, and it has to be a test rather than a review because four
reviewers reading four files do not diff them against each other.
