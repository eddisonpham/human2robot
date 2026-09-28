# Roadmap

Ordered by what most protects the project's claims, not by what is most
interesting. Each item names the defect or gap it closes and how you would know
it is done. Verified against the repository at commit `f28423d`.

## Tier 0: things that would let a false number back into the docs

- [ ] **A feasibility metric against the simulated hand.** Nothing in the repo
  currently tests whether the Allegro can *execute* an optimized trajectory.
  Every quality claim rests on a behavior-cloning proxy that measures
  regressibility, which the retargeting finding showed will happily reward a
  degenerate signal. This is the single highest-value remaining item: it is
  the metric the project's stated purpose actually needs.
  Done when a committed script reports, per trajectory, whether the
  constraint-projected trajectory is dynamically executable in `AllegroPickupEnv`
  under open-loop replay, and the doc tables carry that column.
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

- [ ] **Fix the `domain_randomized_eval` stub.** It is 31 lines whose docstring
  says "no env mutation": it loops over scales and re-evaluates the *same*
  deterministic environment, so every scale returns the same number. Conditions
  D and E are blocked precisely because this was never implemented, and the
  function currently looks like working code.
  Done when mass, friction, and actuator-gain randomization actually mutate the
  MuJoCo model, the function is covered by tests that assert the scales produce
  *different* returns, and `docs/FINDINGS_residual_degeneracy.md` is updated.
- [ ] **Track `rust/` or delete it.** `/rust/` is gitignored with a comment
  claiming the Phase 10 work is "unstarted", but `rust/inference_server/src/
  main.rs` is 121 lines of working axum + ONNX Runtime code. A whole subsystem
  is invisible to version control and the comment actively misdescribes it.
  Done when the source and `Cargo.toml` are tracked with `target/` ignored
  narrowly, and `cargo test` passes; or the directory is removed.
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
- [ ] **Rust inference server (depends on tracking it first).** 121 lines exist.
  Finish it or remove it; the current state is the worst of both, since it
  looks unimplemented in the docs and is invisible in git.
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
