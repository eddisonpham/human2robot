# Human2Robot - Status and Results (as of 2026-09-27, full 5-condition matrix)

## Project identity

- Repo: eddisonpham/human2robot (local path `human2robot/`)
- Tier A env: `AdroitHandRelocateSparse-v1` / `-v1`
- Tier B env: `Human2Robot-AllegroPickup-v0`
- Runner: `uv run human2robot-train --config <yaml> --seed <n> [--run-name <name>] [--results-dir <dir>] [--resume]`
- Configs: `configs/{tier_a_relocate.yaml, tier_b_pickup.yaml, tier_b_cond_{b,c,d,e}.yaml}`
- Tables below are regenerated from disk by `uv run python scripts/summarize_ablation.py --markdown`

## Headline

All five conditions are now complete on the Tier B pickup task at 2,000,004 steps
each, and the Q-function is numerically stable. The headline finding changed
once the critic bug was found, and the change is the important part of this
update:

> **The previous "BC-init beats from-scratch SAC by 48 percent" result did not
> survive the fix for the critic bug.** After the fix, Conditions A and B are
> statistically indistinguishable (A -48.9 +/- 5.3, B -51.0 +/- 1.6, over three
> seeds each). The 48 percent gap was an artifact of a broken observation
> pipeline, not evidence for demonstration-guided initialization.

The single most useful condition is **C (blackbox dynamics augmentation) at
-43.9**, but that is one seed and should be read as a lead, not a result.

## The bug that invalidated the previous results

`_rotation_vector` (three duplicated copies: `envs/allegro.py`,
`dynamics/nominal_physics.py`, `rl/_local_demos.py`) recovered a rotation axis by
dividing the skew part of the rotation matrix by `2 * sin(angle)`. That
expression is singular at angle = pi, and a free box landing on the floor hits
180 degrees routinely. Observed magnitudes reached **6.35e8** (p99.99 = 458,837)
in observation dimensions 47-48 (object rotation vector) and 53-55 (object
angular velocity).

Those dimensions are inputs to the critic, so every critic target was poisoned.
The very first offline update logged `qf_loss = 9.9e11`; full runs reached
1e15-1.9e15 with hundreds to a thousand spikes above 1000 per run.

The fix is a shared `utils/rotation.py` providing `rotation_vector(matrix)`,
which converts through a normalized unit quaternion using the
largest-diagonal branch, canonicalized to non-negative scalar part, with
`angle = 2 * atan2(norm, w)`. It is bounded by pi at all angles. All three
duplicated copies are deleted in favor of the import.

Verified effect:

| Measurement | Before | After |
|-------------|--------|-------|
| max abs observation over 60k env steps | 6.35e8 | 71-84 |
| median qf_loss over 390 updates | - | 12.77 |
| max qf_loss over 390 updates | 1.9e15 | 351 |
| spikes above 1000 | hundreds to 1000 | 0 |

`tests/python/test_rotation.py` (22 tests) includes a pinned reference copy of
the old singular implementation asserting that it still diverges, so the
regression cannot silently return.

Commit `4b89a8c`.

## Second fix: dynamics augmentation retune

`retrain_every_steps: 250 -> 2500` in `configs/tier_b_cond_{c,d,e}.yaml`.
Refitting 5 ensemble members every 250 env steps dominated wall clock while the
replay buffer had gained only a few hundred transitions. Commit `619af06`.

`synthetic_ratio` stayed at `0.5` deliberately. A 40k-step Condition C probe with
the old ratio and the new retrain interval gave max q_loss 305 and zero spikes,
which establishes the retune as a compute fix rather than a stability crutch.
The instability was the rotation bug, not the augmentation ratio.

## Third fix: the residual physics delta did not match the simulator

Found while chasing the coverage gap, not by any training symptom.

`dynamics/nominal_physics.py` had **13 percent test coverage** and three real
bugs underneath it:

1. `compute_obs_delta` and `compute_physics_deltas` built a `ctrl` vector and
   then never assigned it to `data.ctrl`. The simulated step ran with zero
   actuator force.
2. Neither function applied the env's base-qvel override. `AllegroPickupEnv.step`
   treats `action[:6]` as a direct free-joint velocity command, not an actuator
   input; the nominal step ignored it.
3. `_rotvec_to_quat` returned `[x, y, z, w]` where MuJoCo's `qpos` expects
   `[w, x, y, z]`, so reconstructed states were misoriented. The same ordering
   mistake appeared in `_quat_delta_to_rotvec_delta`, which read the vector part
   from columns `0:3` (that is `w, x, y`) instead of `1:4`.

Measured against real `AllegroPickupEnv` steps, the delta was wrong by a median
of **12.9 per step** on a scale where the deltas themselves are order 1. The
consequence: **Conditions D and E were training their residual ensembles
against physics that did not describe the simulator they were modeling.** Their
reported results are not evidence about residual dynamics augmentation.

After the fix, `compute_obs_delta` reproduces a real env step to float32
precision (max error 1.7e-6 over 80 random steps). `train.py` now calls it
directly instead of routing through the lossy qpos/qvel compatibility chain.
`tests/python/test_nominal_physics.py` (23 tests) pins the equivalence against
the live environment, so this cannot silently regress.

Commit `87bd57b`. The D/E seed 0 runs were moved to
`results/_prefix_archive/tier_b_pickup_cond_{d,e}_s0_badphysics` and D/E are
being re-run from scratch on the corrected delta.

## Results, all conditions

Conditions D and E below are from the **quarantined** runs described above and
are retained only to show what the broken physics produced. Do not read them as
results for residual augmentation; the replacement runs are in flight. A, B, and
C are unaffected, since C is blackbox and never calls the physics delta path.

Trajectory summary (mean/median/best/worst/final over all evals in the run):

| Condition | Seeds | Mean eval | Median | Best | Worst | Final |
|-----------|-------|-----------|--------|------|-------|-------|
| A (from-scratch SAC) | 3 | -48.9 | -9.5 | -2.4 | -495.1 | -5.2 |
| B (BC-init + demo replay) | 3 | -51.0 | -14.6 | -1.9 | -577.8 | -12.5 |
| C (blackbox dynamics aug) | 1 | -43.9 | -9.8 | -2.9 | -423.6 | -17.0 |
| D (residual dynamics aug) | 0 valid | QUARANTINED | - | - | - | - | - |
| E (demos + residual dynamics aug) | 0 valid | QUARANTINED | - | - | - | - | - |

Per run, with critic health:

| Condition | Seed | Status | Step | Evals | Mean | qf median | qf max | >1e3 | >1e6 |
|-----------|------|--------|------|-------|------|-----------|--------|------|------|
| A | s0 | completed | 2000004 | 103 | -44.9 | 0.04411 | 385.2 | 0 | 0 |
| A | s1 | completed | 2000004 | 100 | -46.8 | 0.06769 | 678 | 0 | 0 |
| A | s2 | completed | 2000004 | 100 | -54.9 | 0.05263 | 323.8 | 0 | 0 |
| B | s0 | completed | 2000004 | 100 | -49.1 | 0.1689 | 1844 | 16 | 0 |
| B | s1 | completed | 2000004 | 100 | -51.6 | 0.1337 | 1398 | 16 | 0 |
| B | s2 | completed | 2000004 | 100 | -52.2 | 0.122 | 1369 | 15 | 0 |
| C | s0 | completed | 2000004 | 100 | -43.9 | 0.05564 | 467.6 | 0 | 0 |
| D | s0 | quarantined, re-running | - | - | - | - | - | - | - |
| E | s0 | quarantined, re-running | - | - | - | - | - | - | - |

What this table says:

- **No 1e6-scale explosions anywhere.** The failure mode is gone.
- **A vs B is a null result.** A -48.9 +/- 5.3, B -51.0 +/- 1.6. B is
  marginally *worse* and the gap is well inside seed noise. Three seeds each.
- **B still shows mild 1e3-scale spikes** (15-16 per run, max 1.8e3). A and C
  show none. Demo-seeded replay is the common factor in B and the quarantined
  E. Not urgent, but it is the one remaining critic-health difference between
  the conditions that have valid data.
- **C has one valid seed.** Its -43.9 is a lead, not a finding. A vs B remains
  the only comparison with enough seeds to mean anything.
- Absolute returns are all negative because the reward is shaped around object
  lift plus a success bonus and the agent rarely picks the object up. No run
  reaches a positive return.

## Process defect found and worked around, not fixed

Four runs wedged reproducibly: A s0 at 1.60M and 1.765M steps, A s1 at
1.914M, A s2 at 30k. Symptom was ~59 CPU-seconds/minute with no metric writes
for 25-45 minutes, always late in training, and never during a checkpoint write.
Memory was fine (12GB free), so contention is not a confirmed cause.

Each occurrence was recovered with `taskkill //F` plus `--resume`. The
checkpoint fixes from the previous round (trimmed `state_dict`, retention of the
newest 3) are what made every resume succeed: 608MB checkpoints loaded cleanly
at 1.55M, 1.75M, and 1.9M states. A s2 was relaunched from scratch. **This hang
is undiagnosed.**

## What was built this session

- `src/human2robot/utils/rotation.py` (new): `rotation_vector`,
  `_matrix_to_quaternion`. The fix described above. 98% covered.
- `src/human2robot/envs/allegro.py`, `dynamics/nominal_physics.py`,
  `rl/_local_demos.py`: local `_rotation_vector` copies deleted, shared import.
- `tests/python/test_rotation.py` (new): 22 tests, includes the pinned
  old-implementation divergence check.
- `tests/python/test_tier_b_configs.py` (new): 13 tests pinning every Tier B
  config to `Human2Robot-AllegroPickup-v0`, 2,000,000 steps, 12 envs, and
  consistent dynamics hyperparameters across the residual conditions. Added
  after a launch error sent A s1/s2 to `AdroitHandRelocate-v1` under Tier B run
  names; both were killed, deleted, and relaunched correctly.
- `scripts/summarize_ablation.py` (new): regenerates every table above from
  `results/tier_b_pickup_cond_<letter>_s<seed>/metrics.jsonl` and
  `run_status.json`, so this document cannot drift from the metrics.
- `configs/tier_b_cond_{c,d,e}.yaml`: `retrain_every_steps` 250 -> 2500.
- `src/human2robot/export/onnx.py`: `export_actor` now creates the parent
  directory of an explicitly requested manifest path, which previously raised
  `FileNotFoundError` for any nested manifest.

Pre-fix runs are quarantined in `results/_prefix_archive/*_prefix` so they
cannot be mistaken for the corrected results, and the bad-physics D/E runs in
`results/_prefix_archive/*_badphysics`.

## Coverage tests added

- `tests/python/test_nominal_physics.py` (new, 23 tests): pins the nominal
  physics delta to real environment steps, plus the ctrl mapping, state
  reconstruction, quaternion conventions, and the unwired ensemble wrapper.
- `tests/python/test_eval_cli_coverage.py` (new, 10 tests): the
  `domain_randomized_eval` sweep, the plot CLI, and the benchmark CLI including
  its malformed-group rejection.
- `tests/python/test_onnx_export.py` (+6 tests): `_build_sac`, checkpoint
  export with parity validation, explicit manifest paths, and the CLI.
- `tests/python/test_sb3_check.py` (+3 tests): the SB3 CLI's config-to-kwargs
  translation, timestep override, and output-dir defaulting.

## Open issues

1. **The task is barely learned.** No run gets a positive return. The Allegro
   fingers curl inward and cannot reach below the palm, so grasping an object at
   z=0.08 needs the hand to descend and the fingers to extend almost
   simultaneously. Random exploration for 10k steps essentially never finds
   that. This is the underlying difficulty and it limits what any of these
   conditions can show.
2. **C needs seeds 1 and 2; D and E need to be re-established entirely.** C is
   the most promising lead at -43.9 and its seed 1 is in flight. D and E have no
   valid data at all until the corrected re-runs finish, so the residual
   dynamics conditions are currently untested rather than merely underpowered.
3. **Undiagnosed training hang** at 1.6M-1.9M steps, four occurrences.
   Mitigated by resume, not fixed.
4. **B and E retain 1e3-scale qf spikes.** Small, but the only remaining
   critic-health difference between conditions.
5. **Coverage is fixed**: 95.25 percent, above the 90 percent gate, with 246
   tests passing. The gate had been failing at 85.62 percent. The gap was not
   mainly in the previously identified modules; `nominal_physics.py` at 13
   percent was the single largest hole and is where the physics bug was hiding.
   `onnx.py` is now 99%, `robustness.py` 100%, `sb3_check.py` 98%,
   `plot_cli.py` 96%, `benchmark.py` 94%. `train.py` remains at 87% and
   `demo.py` at 83%.
6. **Push**: not attempted. No remote access configured in this environment.

## How to read this as a human

The defensible claim today: **after fixing a singularity in the observation
rotation conversion that had been poisoning every critic target, SAC trains
stably on the Tier B pickup task, and demonstration-guided initialization (B)
shows no benefit over from-scratch SAC (A) across three seeds each.** The
earlier 48 percent advantage for B was an artifact of the bug and does not
reproduce.

Blackbox dynamics augmentation (C) is the most promising direction at -43.9 on
one seed. **Residual augmentation (D, E) has no valid result yet**: the physics
delta those conditions depend on did not match the simulator, so their earlier
numbers are quarantined and being re-run.

The pattern across this session is worth stating plainly: twice now, an
apparently empirical finding (BC-init's 48 percent advantage, then residual
augmentation's apparent weakness) turned out to be an artifact of an
unexercised code path rather than a property of the method. Neither had test
coverage that would have caught it. The task itself is also unsolved by every
condition, so the absolute numbers mostly measure how slowly each run
accumulates the dense shaping terms rather than task competence.
