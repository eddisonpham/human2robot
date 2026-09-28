# Finding: the published results were measured on a 1-DOF signal

## Summary

Every trajectory result this repository published was measured on data that was
not a retargeting. The retargeter in `src/human2robot/data/dexycb.py` summed
MANO joint magnitudes into a single per-finger curl scalar and broadcast that
scalar to all four joints of a finger with a 0.15 spread term. The result was a
hand-curl signal carrying **90 percent of its variance in 1 dimension of 16**.

A real retargeting, `scripts/retarget_dexycb_ik.py` (DexPilot vector
retargeting against MANO keypoints), was implemented, tested, and never used by
any experiment. Its 100 demos sat in `data/demonstrations_dexycb_ik/` for the
whole life of the project while every published number came from the other
directory.

The fix changed the project's headline in both directions. Some numbers went up
and some went down, and the downs were the informative ones.

## How the degenerate path produced plausible numbers

The measurement is the SVD rank of the finger-position matrix (16 columns,
temporal standard deviation removed):

| Data set | dims for 90% of variance | \|J2−J1\| | \|J3−J2\| |
| --- | --- | --- | --- |
| `demonstrations_dexycb` (published) | **1** | 0.047 rad | 0.052 rad |
| `demonstrations_dexycb_ik` (DexPilot) | 8 | 0.767 rad | 0.401 rad |

The two paths differ by a median **1.83 rad per sequence**, essentially the
entire joint range. A real finger curls with the distal joints flexing
differently from the proximal ones; the published path moved adjacent joints by
0.05 rad, which is a single scalar plus noise.

Note the Allegro's distal joints move *less* than its proximal ones under
normal flexion, so `|J3−J2| < |J2−J1|` is correct behavior and not itself a
defect. The discriminator is magnitude, and the invariant test is written on
magnitude.

## What the numbers look like on real retargeted data

Both subjects, 100 sequences each, hyperparameters selected on subject-01 only
(`step_size = 0.35`, chosen from the first 50 subject-01 sequences).

| Quantity | Published (1-DOF) | Subject-01 (IK) | Subject-02 (IK, held out) |
| --- | --- | --- | --- |
| search improved its starting point | 47/50 | 50/50 | **99/100** |
| median cost reduction | 18.0% | 33.0% | 37.3% |
| max jerk | **-41.9%** | **-9.1%** | **-4.6%** |
| smoothness cost | -64.7% | -30.0% | -28.1% |
| max velocity | -57.2% | -71.7% | -72.2% |
| max acceleration | -27.6% | -18.0% | -13.9% |

**The jerk result was almost entirely an artifact, and that is the finding.**
-41.9% became -9.1%. The optimizer was reporting credit for smoothing away a
signal that had no structure to preserve: a one-dimensional curl has nothing but
jitter to remove, so removing it looks like a large win. Real hand motion is
already smooth, so there is almost nothing left to take (4.6% on subject-02).
The smoothness figure moves the same way, -64.7% to -30.0%.

The velocity figure moves the *other* way, -57.2% to -72.2%, because the
degenerate path never drove joints anywhere near the limit and the real
retargeting does.

## What the BC comparison looks like now

The published mean-error result had been demoted for leaking adjacent
transitions. On real retargeted data it no longer needs demoting.

| Data set | random transitions | whole trajectories | tail extrapolation |
| --- | --- | --- | --- |
| Subject-01 | +45.0% (sd 2.1) | **+49.7% (sd 10.7, all seeds positive)** | +6.1% |
| Subject-02 | +41.1% (sd 2.9) | **+51.4% (sd 6.6, all seeds positive)** | -24.6% |
| Synthetic | +56.1% (sd 0.5) | +50.4% (sd 1.0) | +40.3% |

On the 1-DOF path the whole-trajectory split was the *weakest* test at +21%
with a standard deviation of 18 and a sign that flipped between seeds. On
real retargeted data it is the *strongest* test at ~50% with every seed
positive. A degenerate signal that is trivially predictable is exactly the
signal whose advantage should evaporate under a harder split, and it did.

Worst-case error against the resampled control: -56.5% on subject-01 and
-55.1% on subject-02.

The tail-extrapolation split remains weak: +6.1% on subject-01 and -24.6% on
subject-02, against +40.3% on synthetic. That gap is unexplained and is stated
as an open question rather than resolved.

## Why it was missed

Five prior findings in this project were artifacts, and all five were found by
checking a control. This one survived because every check that had been run was
a *relative* comparison. The control arm, the held-out split, the
cross-subject split, and the seed variance all compared the optimizer against
itself on the same degenerate input, so they were all correctly reporting a
real effect measured on the wrong data.

What was missing was a check on the data itself rather than on the measurement.
`tests/python/test_invariants.py` now pins the rank of the demo sets, the
magnitude of within-finger articulation, and the historical degeneracy of the
removed path so it cannot be reintroduced silently.

## What changed in the code

- `src/human2robot/data/dexycb.py`: the basis retargeter raises
  `NotImplementedError`. The unused `_MANO_BASIS` constant that was the tell is
  gone. Sequence loading and manifest writing remain, so a caller cannot
  produce degenerate demos by accident.
- `scripts/retarget_dexycb_ik.py`: positional arguments became flags, and
  `--output-dir` is now required for any subject other than subject-01. The old
  positional form had a hardcoded output directory, so retargeting subject-02
  would have silently replaced subject-01's 100 demos.
- `src/human2robot/data/allegro_demos.py`: the MuJoCo environment import is
  deferred into the synthetic generator. It was module-scope, which made the
  whole `data` package unimportable in a MuJoCo-free environment and blocked
  the WSL retargeting venv.
- `scripts/compare_dexycb_synthetic.py`, `scripts/run_downstream_bc.py`,
  `scripts/validate_optimizer_split.py`: read the DexPilot IK directories.

## Why the BC metric is still not the right metric

Unchanged by this finding, and worth restating. "Is the optimized signal easy
to regress" is a proxy for "is this a useful robot demonstration", and the two
came apart here: the degenerate path scored *worse* on the hard split, so the
metric was at least sensitive to the defect. But nothing in it tests whether the
Allegro can execute the trajectory or whether the object is grasped. A
feasibility score against the simulated hand is the metric that answers the
project's actual question, and it is not implemented.
