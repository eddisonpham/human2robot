# Human2Robot handoff (superseded)

This file previously carried a phase-by-phase status table. It was retired on
2026-09-27 because it had drifted badly out of date: it still reported Tier B
conditions A-E as "not started", quoted pre-fix BC-init numbers that no longer
reproduce, and described a coverage gap that has since been closed.

Maintaining a second status document alongside `REVIEW_STATUS.md` guarantees
the two will disagree, so the table now lives in one place.

## Where things actually stand

- **`REVIEW_STATUS.md`** (repo root) is the authoritative status: the current
  five-condition ablation results, the root causes of the two invalidating bugs
  found so far, and the open issues.
- **`docs/FINDINGS_residual_degeneracy.md`** explains why Conditions D and E
  cannot currently produce an informative result.
- **`docs/FINDINGS_training_hang.md`** records how far the intermittent
  single-threaded stall has been narrowed.
- **`docs/FINDINGS_metrics_integrity.md`** records the resume-induced metrics
  duplication, the data loss it caused, and the TensorBoard recovery path.
- **`agents/09_PHASE_PLAN.md`** remains the source of truth for what each phase
  requires and its acceptance test.
- **`agents/RESUME_ENTRY.md`** is the entry point for picking the work back up.

## Still-true notes carried over from the old file

These parts were accurate and are preserved:

- The repo was renamed `dynhand` -> `human2robot` (package, entry points, env id
  `Human2Robot-AllegroPickup-v0`).
- The C++ trajectory subsystem under `cpp/` is built and validated (pybind11
  module `h2r_cpp`), and the Python optimization pipeline runs over synthetic
  demos. See `cpp/README.md` for toolchain specifics.
- DexYCB downloads live under `data/raw/dexycb/` via
  `scripts/download_dexycb.sh` (Google Drive, gdown).
- Results are never committed: `data/` and `results/` are gitignored. Each run
  writes config, metrics, git commit, and system info into
  `results/<experiment_id>/`.

## Stale guidance that should not be followed

- The old table's "Phase 7/8: Tier B RL conditions A-E, not started" is wrong.
  The matrix has been run; see `REVIEW_STATUS.md`.
- The old metrics section quoted Tier A relocate numbers and a `run_status.json`
  last updated 2026-09-12 with a note to treat it as stale. That run is
  superseded by the Tier B matrix.
- The old "Phase 9: helpers scaffolded, runs not done" is still accurate, and
  is now known to be a blocker for Condition D rather than merely future work.
