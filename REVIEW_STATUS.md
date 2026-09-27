# Human2Robot - Status and Results (as of 2026-09-27, 3 of 5 conditions complete)

> **Scope note.** This document covers the **SAC ablation only**. The project's
> primary deliverable is the human-to-robot trajectory conversion pipeline in
> `src/human2robot/data/dexycb.py` and `cpp/`, which runs on 100 real DexYCB
> sequences. See [`agents/RESUME_ENTRY.md`](agents/RESUME_ENTRY.md) for that
> side, summarized in the README.

## Project identity

- Repo: eddisonpham/human2robot (local path `human2robot/`)
- Tier A env: `AdroitHandRelocateSparse-v1` / `-v1`
- Tier B env: `Human2Robot-AllegroPickup-v0`
- Runner: `uv run human2robot-train --config <yaml> --seed <n> [--run-name <name>] [--results-dir <dir>] [--resume] [--device cpu|cuda]`
- Run control: `bash scripts/{resume_run,pause_run,monitor_run,watchdog_runs}.sh`
- Configs: `configs/{tier_a_relocate.yaml, tier_b_pickup.yaml, tier_b_cond_{b,c,d,e}.yaml}`
- Tables below are regenerated from disk by `uv run python scripts/summarize_ablation.py --markdown`

## Headline

Conditions A, B, and C are complete on the Tier B pickup task at 2,000,004 steps
each with three seeds apiece, and the Q-function is numerically stable
everywhere. Conditions D and E were stopped early and are quarantined: the
residual they learn is identically zero, for a structural reason documented
below, and no amount of additional training changes that.

Two headline findings, both of which reversed earlier conclusions:

> **1. The previous "BC-init beats from-scratch SAC by 48 percent" result did
> not survive the fix for the critic bug.** After the fix, A and B are
> indistinguishable (A -49.2 +/- 4.9, B -51.0 +/- 1.6, three seeds each). The
> 48 percent gap was an artifact of a broken observation pipeline, not evidence
> for demonstration-guided initialization.

> **2. A, B, and C are all within noise of each other.** A -49.2 +/- 4.9,
> B -51.0 +/- 1.6, C -46.9 +/- 6.9. C is nominally the best condition, but its
> three seeds span -41.9 to -54.8, which is the same spread A shows. The
> honest reading is that on this task, at this scale, none of A, B, or C
> measurably outperforms the others.

No run in any condition reaches a positive return. The task is not solved.

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

Commit `87bd57b`. The D/E seed 0 runs with the broken physics were moved to
`results/_prefix_archive/tier_b_pickup_cond_{d,e}_s0_badphysics` and re-run from
scratch on the corrected delta.

## Fourth finding: D and E are structurally degenerate, not merely hard

After the physics delta was fixed, the replacement D and E runs were checked
against ground truth with `scripts/check_residual_target.py`. The learned
residual is **0.0000 percent of the true delta**.

The reason is not a bug and not under-training. `compute_obs_delta` is a direct
forward simulation of the same MuJoCo model the environment steps, so the
nominal model *is* the simulator. There is no model error left for a residual
to explain. The target for the residual ensemble is identically zero and the
condition reduces to a no-op with an ensemble attached.

The spec anticipates this. `agents/05_RL_ALGORITHM_SPEC.md` lines 104-117
prescribes domain randomization at rollout time, which is what makes the
simulator a *model* of something other than itself and gives the residual
something to learn. That is **not implemented**: `evaluation/robustness.py` is
a stub that mutates no environment state. So the one mechanism the design
relies on to make D and E meaningful is absent.

D and E were therefore stopped at roughly 600k steps and their status files
marked `cancelled`. Completing them requires implementing rollout-time domain
randomization (estimated 6-7 hours), which is a scope decision, not a
debugging task. Full analysis in `docs/FINDINGS_residual_degeneracy.md`.

## Results

A, B, and C are complete at 2,000,004 steps with three seeds each. C is
blackbox and never calls the physics delta path, so it was unaffected by the
delta bugs. D and E rows are the cancelled partial runs, shown only so the
table regenerates cleanly from disk; **they are not results**.

Trajectory summary (mean/median/best/worst/final over all evals in the run):

| Condition | Seeds | Mean eval | Median | Best | Worst | Final |
|-----------|-------|-----------|--------|------|-------|-------|
| A (from-scratch SAC) | 3 | -49.3 | -9.5 | -2.4 | -495.1 | -5.2 |
| B (BC-init + demo replay) | 3 | -51.0 | -14.6 | -1.9 | -577.8 | -12.5 |
| C (blackbox dynamics aug) | 3 | -46.9 | -9.8 | -2.1 | -427.2 | -10.2 |
| D (residual dynamics aug) | 0 valid | QUARANTINED | - | - | - | - | - |
| E (demos + residual dynamics aug) | 0 valid | QUARANTINED | - | - | - | - | - |

Per run, with critic health:

| Condition | Seed | Status | Step | Evals | Mean | qf median | qf max | >1e3 | >1e6 |
|-----------|------|--------|------|-------|------|-----------|--------|------|------|
| A | s0 | completed | 2000004 | 100 | -46.0 | 0.05227 | 385.2 | 0 | 0 |
| A | s1 | completed | 2000004 | 100 | -46.8 | 0.06837 | 678 | 0 | 0 |
| A | s2 | completed | 2000004 | 100 | -54.9 | 0.05263 | 323.8 | 0 | 0 |
| B | s0 | completed | 2000004 | 100 | -49.1 | 0.1689 | 1844 | 16 | 0 |
| B | s1 | completed | 2000004 | 100 | -51.6 | 0.1337 | 1398 | 16 | 0 |
| B | s2 | completed | 2000004 | 100 | -52.2 | 0.122 | 1369 | 15 | 0 |
| C | s0 | completed | 2000004 | 100 | -43.9 | 0.05552 | 467.6 | 0 | 0 |
| C | s1 | completed | 2000004 | 100 | -54.8 | 0.04851 | 428.7 | 0 | 0 |
| C | s2 | completed | 2000004 | 100 | -41.9 | 0.05019 | 431.9 | 0 | 0 |
| D | s0 | cancelled, quarantined | 600000 | 30 | -108.5 | 2.894 | 417.9 | 0 | 0 |
| D | s1 | cancelled, quarantined | 620004 | 31 | -118.0 | 2.951 | 425 | 0 | 0 |
| E | s0 | cancelled, quarantined | 600000 | 30 | -115.2 | 18.57 | 2098 | 12 | 0 |
| E | s1 | cancelled, quarantined | 600000 | 30 | -80.5 | 18.08 | 1400 | 7 | 0 |

What this table says:

- **No 1e6-scale explosions anywhere.** The failure mode is gone.
- **A vs B is a null result.** A -49.2 +/- 4.9, B -51.0 +/- 1.6. B is
  marginally *worse* and the gap is well inside seed noise. Three seeds each.
- **C does not separate from A or B either.** C -46.9 +/- 6.9 is the best mean
  but has the largest spread; its worst seed (-54.8) ties A's worst. C s2 was
  the single best run at -41.9, which is what made C look promising on one
  seed. With three seeds the advantage disappears into the noise, which is why
  the earlier "C is the promising lead at -43.9" framing does not survive.
- **B shows mild 1e3-scale qf spikes** (15-16 per run, max 1.8e3). A and C show
  none. Demo-seeded replay is the common factor in B and the cancelled E. Not
  urgent, but it is the one remaining critic-health difference between the
  conditions with valid data.
- **The cancelled D/E runs have qf medians 50x to 350x the others** (2.9 and
  18.1-18.6 against 0.05-0.17). Training a residual ensemble against an
  identically-zero target is the only condition that destabilizes the critic.
  This is corroborating evidence for the degeneracy, not a separate result.
- Absolute returns are all negative because the reward is shaped around object
  lift plus a success bonus and the agent rarely picks the object up.

## Fifth finding: the training hang was a CUDA driver hang, and it is now avoidable

Four runs wedged reproducibly: A s0 at 1.60M and 1.765M steps, A s1 at
1.914M, A s2 at 30k. Symptom was ~59 CPU-seconds/minute with no metric writes
for 25-45 minutes, always late in training, and never during a checkpoint write.

A live specimen was caught with `py-spy dump --native`. The Python stack parked
in `torch.distributions.Normal.__init__`, which is a red herring. The native
stack is the answer:

```
cuStreamSynchronize -> c10::cuda::memcpy_and_sync -> searchsorted_out_cuda
                    -> _local_scalar_dense_cuda -> item -> is_nonzero
```

PyTorch validates `Normal` by calling `is_nonzero(...)` and reducing it with
`.item()`. That is a synchronizing CUDA call, and it never returned. Meanwhile
`nvidia-smi` listed **seven PIDs on the GPU with only one alive**: every
force-killed run had leaked a CUDA context, and the accumulation eventually
wedged the driver.

The fix is an escape hatch rather than a patch, because the leak comes from
`taskkill //F`, which on Windows has no graceful counterpart. The trainer now
takes `--device`, `scripts/resume_run.sh` defaults to `DEVICE=cpu`, and CUDA is
opt-in. Measured cost: 129s CPU against 120s CUDA on 40k steps, about 7 percent.
Run the gate on CPU until the machine is rebooted. Documented in
`docs/FINDINGS_training_hang.md`. Commit `22caf5a`.

## Sixth finding: resumes were silently corrupting the metrics stream

`--resume` rewound the trainer to the last checkpoint but not `metrics.jsonl`,
so re-executed steps appended duplicate and out-of-order records. **Six of the
twelve runs were affected**, and `human2robot-plot` and `human2robot-benchmark`
both correctly refuse an unhealthy stream.

Fixed by `RunRecorder.rewind_metrics(step)`, called on resume. Commits
`6cda9b8`, `3b105d9`.

The repair of the already-damaged files then went wrong in a way worth
recording. Deduplicating on `step` alone destroyed every `eval_return_mean` in
every run, because training and eval records share step values. All 100 eval
points per completed run were recovered from the TensorBoard event files
(`eval/return_mean` tag) by `scripts/rebuild_eval_metrics.py`. A s0's recovered
mean of -46.04 exactly matched the value computed from the intact stream before
the damage, which is the check that the recovery is sound. Only
`eval_return_mean` is recoverable, since it is the only eval field any result
uses. Documented in `docs/FINDINGS_metrics_integrity.md`.

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
- `src/human2robot/rl/train.py`: `--device` flag and `resolve_device()`, the
  escape hatch from the CUDA hang. Pause is a `results/<run>/PAUSE` sentinel
  polled every 100 steps, because Windows offers no graceful signal to a
  detached process, and `rewind_metrics` on resume.
- `src/human2robot/dynamics/ensemble.py`: `state_dict` / `load_state_dict`, so a
  resume keeps the trained ensemble instead of refitting from scratch.
- `scripts/monitor_run.sh` / `.py` (new): step, rate, ETA, stall state, and
  metrics audit health for a live run.
- `scripts/watchdog_runs.sh`: PID lock, added after a stale watchdog resurrected
  deliberately cancelled D/E runs and wasted about a GPU-hour.
- `scripts/rebuild_eval_metrics.py` (new): recovers `eval_return_mean` from
  TensorBoard event files.
- `scripts/check_residual_target.py` (new): the ground-truth check that exposed
  the D/E degeneracy.
- Repository cleanup (commit `a79ac68`): 8 superseded scripts removed (21 to
  13), 6 dead functions removed from `nominal_physics.py` (324 to 185 lines),
  `README.md` rewritten with every command verified by execution. The gitignore
  pattern `data/` was unanchored and therefore also matched
  `src/human2robot/data/`, which had silently swallowed `dexycb.py` (171 lines)
  from the repository. It is now tracked with passing tests.

Pre-fix runs are quarantined in `results/_prefix_archive/*_prefix` so they
cannot be mistaken for the corrected results, and the bad-physics D/E runs in
`results/_prefix_archive/*_badphysics`.

## Coverage tests added

- `tests/python/test_nominal_physics.py` (new, 23 tests): pins the nominal
  physics delta to real environment steps, plus the ctrl mapping, state
  reconstruction, quaternion conventions, and the unwired ensemble wrapper.
- `tests/python/test_invariants.py` (new): pins the `[w, x, y, z]` quaternion
  order, the shared substep count, config reachability, and same-seed
  reproducibility all the way to the action sequence. Read this before touching
  `envs/`, `dynamics/`, or `rl/`.
- `tests/python/test_pause_resume.py` (new, 14 tests): ensemble round-trip,
  checkpoint contents, the pause sentinel, the signal path, and device
  resolution.
- `tests/python/test_eval_cli_coverage.py` (new, 10 tests): the
  `domain_randomized_eval` sweep, the plot CLI, and the benchmark CLI including
  its malformed-group rejection.
- `tests/python/test_tier_b_configs.py` (new, 13 tests): every Tier B config
  pinned to `Human2Robot-AllegroPickup-v0`, 2,000,000 steps, 12 envs, and
  consistent dynamics hyperparameters across the residual conditions.
- `tests/python/test_onnx_export.py` (+6 tests) and `test_sb3_check.py`
  (+3 tests).

## Open issues

1. **The task is barely learned.** No run gets a positive return. The Allegro
   fingers curl inward and cannot reach below the palm, so grasping an object at
   z=0.08 needs the hand to descend and the fingers to extend almost
   simultaneously. Random exploration for 10k steps essentially never finds
   that. This is the underlying difficulty and it limits what any of these
   conditions can show.
2. **D and E are blocked on a scope decision.** They are not underpowered, they
   are degenerate until rollout-time domain randomization is implemented, which
   the spec prescribes and the code does not do. Either implement it (6-7 hours)
   and complete the 5-condition matrix, or ship A, B, and C and document D and E
   as degenerate. Shipping three conditions is the defensible option for a
   resume.
3. **The GPU is in a degraded state.** Several leaked CUDA contexts from
   force-killed runs. Train on CPU (`--device cpu`) until the machine is
   rebooted. Cost is about 7 percent.
4. **B retains 1e3-scale qf spikes** (15-16 per run, max 1.8e3). Small, and the
   only remaining critic-health difference between the conditions with valid
   data.
5. **The results are underpowered for the effect sizes involved.** Three seeds
   cannot separate conditions whose means differ by 2 points with standard
   deviations of 2-7. If the goal is to rank A, B, and C, the honest answer
   needs more seeds, not more conditions.
6. **Coverage is fixed**: 95.09 percent, above the 90 percent gate, with 270
   tests passing. The gate had been failing at 85.62 percent. The gap was not
   mainly in the previously identified modules; `nominal_physics.py` at 13
   percent was the single largest hole and is where the physics bug was hiding.
7. **Push**: not attempted. No remote access configured in this environment.

## How to read this as a human

The defensible claim today: **after fixing a singularity in the observation
rotation conversion that had been poisoning every critic target, SAC trains
stably on the Tier B pickup task, and none of the three conditions that can be
tested (A, B, C) separates from the others across three seeds each.** The
earlier 48 percent advantage for demonstration-guided initialization was an
artifact of the bug and does not reproduce. Blackbox dynamics augmentation is
nominally the best of the three at -46.9 +/- 6.9, but its three seeds span
-41.9 to -54.8, so that ordering is not established.

**Residual augmentation (D, E) has no valid result and cannot get one as the
code stands.** The nominal model is the simulator, so the residual target is
identically zero. This is not a tuning problem.

The pattern across this session is worth stating plainly. Three times now, an
apparently empirical finding turned out to be an artifact of an unexercised
code path rather than a property of the method: BC-init's 48 percent advantage
(rotation singularity), residual augmentation's apparent weakness (physics
delta that never matched the simulator, then a zero target), and C's apparent
prominence (one lucky seed). None of the three had test coverage that would
have caught it, and each was found by reading code or ground truth rather than
by watching a metric. The task itself is also unsolved by every condition, so
the absolute numbers mostly measure how slowly each run accumulates the dense
shaping terms rather than task competence.
