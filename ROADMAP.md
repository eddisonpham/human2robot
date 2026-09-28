# Roadmap

Ordered by what most protects the project's claims, not by what is most
interesting. Each item names the defect or gap it closes and how you would know
it is done. Verified against the repository at commit `f28423d`.

## Completed in the September 28 pass

- [x] **A feasibility metric against the simulated hand** (was Tier 0 item 1).
  `evaluation/feasibility.py`, calibrated on three cases of known difficulty,
  with `settle_steps` removing the hand's 0.5 s start transient.
  `scripts/score_feasibility.py` measures it on all three data sets:
  optimization takes subject-01 from **6/100 to 34/100 executable** and
  subject-02 from 0 to 8, while synthetic is 100/100 on both arms. It also
  replaced the fake replay check with a real one, and showed optimization
  *halves* tracking drift rather than merely looking smoother.
- [x] **`domain_randomized_eval` really randomizes** (was Tier 1 item 1).
- [x] **The degenerate retargeter is gone** and a rank/dimensionality invariant
  prevents it recurring.
- [x] **One definition of the splits.** Four private copies existed across
  `check_bc_split_granularity.py`, `run_downstream_bc.py`, and the experiment
  module. All four now use `optimization.experiment`.
- [x] **Joint limits come from the model.** The four hand-written copies had
  already drifted, and 15 of 16 finger entries were wrong. See
  `docs/FINDINGS_joint_limits.md`.
- [x] **Every published figure is pinned to its artifact.**
  `evaluation/published.py` resolves 50 named figures out of
  `results/trajectory_optimization/*.json`, and
  `tests/python/test_published_results.py` asserts the exact strings that appear
  in `README.md` and `docs/RESULTS.md`. A committed snapshot in
  `docs/published_figures.json` keeps the check running on a fresh clone, where
  `results/` is gitignored.
- [x] **The tail-extrapolation reversal is explained.** It was the last open
  question in the conversion pipeline. `scripts/diagnose_tail_extrapolation.py`
  shows real hand motion decelerates into a stop, so the prefix split's holdout
  is nearly motionless and the control arm scores at the predict-nothing
  baseline. The split cannot rank the optimizer on this data. It is not an open
  defect, it is a property of the data, and it is now written up as one.
- [x] **The split-gragularity script was reporting one seed as five.** It ran
  a single split per granularity while the documentation quoted three-seed
  error bars, and every data set wrote to the same file, so the last run
  silently overwrote the others. It now runs 5 seeds, reports mean and sd, and
  writes one file per set.
- [x] **`rust/` is under version control.** The vestigial empty `.git` was
  removed with approval and the code folded in. It did not compile: it was
  written against a different `ort` API than the pinned `2.0.0-rc.13`, and had
  never been built because it sat outside version control. It builds and has 6
  tests now.

## Defects this pass found that were not on the list

These are worth reading before trusting any earlier number.

- **`OptimizerConfig` recorded a `dof` it did not enforce.** The facade stored
  the parameter and set only the arrays, so a mismatched length was recorded
  here and ignored by the optimizer. Lengths are now validated at construction.
- **`split_prefix` could hand back an empty holdout.** A one-configuration tail
  produced zero test transitions, which the packer then silently concatenated
  into an empty data set. Both sides now require an adjacent pair, and the
  function raises instead of splitting nothing.
- **`to_transitions` kept zero-row pairs.** A single-step trajectory yielded a
  pair of empty arrays, which is truthy, so the emptiness check never fired.
- **A zero baseline was reported as a 100 percent improvement.** The
  `max(raw, 1e-12)` guard turned an unchanging metric into a full improvement
  in the published tables. `pairwise_reductions` now raises, which is a
  behaviour change and is why `test_pipeline_scripts.py` asserts the rejection.
- **`validate_open_loop_replay` was a third fake.** Named for a MuJoCo replay,
  tested in a file named `test_mujoco_replay_optimized.py`, and never touched
  the simulator: it compared `a_demo` arithmetic against stub bounds of
  `zeros(22)` and `ones(22)`. Removed; `evaluation.feasibility` is the real one.
- **The Rust server had no version control at all.** Worse than "untracked" —
  `rust/inference_server/.git` was an accidental `git init` with zero commits,
  zero refs, zero objects and no remote, so the 121 lines of `main.rs` existed
  only on one disk. It also **did not compile**: written against a different
  `ort` API than the pinned version provides, and never built by anyone. Both
  are fixed.
- **The split-granularity table was fabricated in its error bars.** It quoted
  three-seed standard deviations for an experiment the script ran at one seed,
  and wrote every data set to one output file so the last run overwrote the
  rest. The numbers it did produce happened to be right; the spread was
  invented.
- **The published numbers and the recorded artifacts had drifted apart.** The
  kinematic table claimed 99/100 where the artifact said 100/100, and the
  convergence medians differed too, because the limits correction had been made
  without re-running the experiments that the documentation quotes. Nothing
  detected this, since the documents were a hand-copied second copy of data
  that already existed on disk in machine-readable form.

## Tier 0: things that would let a false number back into the docs

- [x] **A feasibility metric against the simulated hand.** Done, see above.
  Remaining follow-up: get the doc tables to carry the executability column,
  which is the part of the original definition of done that is still open.
- [ ] **Pin the tail-extrapolation gap as a tracked limitation with an owner.**
  Whole-trajectory holdout is +49.7% / +51.4%, but extrapolating the tail of
  every trajectory is +6.1% on subject-01 and **-24.6% on subject-02** against
  +40.3% on synthetic. It is documented and unexplained. Either investigate or
  state it as a permanent scope boundary, but do not leave it ambiguous.
  Done when `docs/FINDINGS_retargeting.md` has a short subsection with either a
  mechanism and evidence, or an explicit "known limitation, not investigated".
- [ ] **Guard every published number with a regression test.** The retargeting
  bug survived five prior findings because every check compared the optimizer
  against itself on the same input. Demo-set rank is now pinned, but the
  *numbers* in the README are not.
  Done when a test asserts each headline figure within tolerance against
  `results/trajectory_optimization/*.json`, so a change in the pipeline fails
  loudly instead of silently invalidating the docs.

## Tier 1: correctness and reproducibility gaps

- [x] **Fix the `domain_randomized_eval` stub.** Done. `body_mass`,
  `geom_friction`, `dof_damping` and `actuator_gainprm` are scaled per condition
  and restored unconditionally, including on exception. Conditions now share
  initial states, so the spread is attributable to the perturbation; deriving a
  seed from the scale would have confounded the two. A test observes the model
  *during* the rollouts and asserts each condition saw its own scale, which is
  the property the old implementation failed. A non-MuJoCo environment now
  raises instead of returning a number nothing was perturbed to produce.
  Still open: `docs/FINDINGS_residual_degeneracy.md` needs updating to say D and
  E are no longer blocked on the code, only on the physics being non-trivial.
- [ ] **Decide the fate of conditions D and E.** The configs
  `tier_b_cond_d.yaml` and `tier_b_cond_e.yaml` are still present and a reader
  will assume they are runnable. They are structurally degenerate: the nominal
  physics model *is* the simulator, so the residual is identically zero. Note
  that domain randomization now exists, so this is a physics question and no
  longer an implementation gap.
- [ ] **Decide the fate of conditions D and E.** The configs
  `tier_b_cond_d.yaml` and `tier_b_cond_e.yaml` are still present and a reader
  will assume they are runnable. They are structurally degenerate: the nominal
  physics model *is* the simulator, so the residual is identically zero.
  Done when either domain randomization exists and they run, or the configs
  carry a comment pointing at `FINDINGS_residual_degeneracy.md` and the README
  scope section says they are not runnable as written.
- [ ] **Migrate the ONNX exporter off `torch.onnx.export`.** `export/onnx.py:65`
  uses the legacy path, which PyTorch 2.9 deprecates in favour of
  `torch.export`. The parity and latency checks already exist, so the migration
  is testable.
  Done when export runs through `torch.export`, the parity test still passes,
  and the manifest reports latency for both paths.
- [ ] **Reconcile `REVIEW_STATUS.md` with the current state.** It is the
  authoritative document a reviewer reads first and was last updated before the
  retargeting fix. Its trajectory claims need to be scoped to the RL ablation,
  or removed, so it cannot contradict `docs/RESULTS.md`.

## Tier 2: engineering quality

- [ ] **Decide on CI.** It was added and then deliberately removed. The
  argument for reinstating it is specific and worth weighing: four separate
  findings in this project, including the retargeting one, were caught by
  something *running* rather than by review, and the coverage floor and
  `test_invariants.py` only execute when someone remembers.
  Done when either a workflow runs on `windows-latest` (the build is pinned to
  win32/AMD64 and torch comes from a cu128 index) or the README says plainly
  that the gate is manual and why that is an accepted risk.
- [ ] **Add a data-provenance check to the pipeline entry points.** The retargeting
  footgun bit three times in this project: `--subject` leaving `--output-dir`
  on the first subject's demos, in `dexycb.py` and again in the IK script.
  The guards exist but they are hand-written and easy to omit in a new script.
  Done when a shared helper owns the "one demo set per subject" rule and every
  script that writes demo directories uses it.
- [ ] **Test the scripts that have no tests.** `retarget_dexycb_ik.py` has an
  output-directory test but nothing covering `mano_keypoints` or
  `retarget_sequence`, which are the parts that would silently produce wrong
  trajectories. Same for `diagnose_convergence.py`.
  Done when a synthetic MANO pose is pushed through `mano_keypoints` and the
  keypoint geometry is asserted against a known configuration.
- [ ] **Broaden the C++ suite where the risk is.** 64 cases cover the optimizer
  well, but the cost function's individual terms and the constraint projection
  at the exact boundary are thinner than the search itself.
  Done when each cost term has a test isolating it, and projection at a joint
  exactly on its limit is covered.

## Tier 3: scope that is genuinely open

- [ ] **More subjects.** Two of ten DexYCB subjects are processed. Cross-subject
  generalization is the strongest claim in the project and rests on n=2, both
  from the same capture rig, so robustness to capture conditions is untested.
  The retargeting path is now a single committed command per subject, so this is
  mechanical: download, extract `pose.npz`, retarget, re-run the four scripts.
  Done when a third subject is added and the cross-subject descent rate is
  re-measured on the pooled held-out set.
- [ ] **Feasibility as a downstream gate (depends on Tier 0).** Once
  trajectories can be scored for executability, the honest end-to-end question
  is whether filtering demonstrations by feasibility improves anything
  downstream. That is the experiment the RL half was supposed to answer, and it
  has never been run on data that passes a feasibility check.
- [ ] **Rust inference server.** It now builds, has 6 tests, and is under
  version control, but it has still **never been run against a model**, so
  there is no latency figure and no confirmation that the manifest's tensor
  names match an actual export. Before it can be called done it needs an
  inference test against a checked-in model, a recorded latency, and the ONNX
  exporter moved off the deprecated `torch.onnx.export` path.
- [ ] **The Shadow Hand stretch.** Not started. Only worth doing if the Allegro
  results are solid, since it is a generalization claim on top of everything
  else.

## Explicitly not worth doing

- **Re-litigating the noise fix or the held-out protocol.** Both are settled,
  measured, and documented. Reopening them costs credibility, not points.
- **Making the Allegro pickup task solvable.** That is a research project. It
  would move the RL half from null to merely weak, and the null is honestly
  reported and not load-bearing on any claim.
- **Adding more SAC conditions.** A more rigorous null on a task no condition
  solves is still a null.
