---
name: dynhand-phase-execution
description: Execute a DynHand build phase from agents/09_PHASE_PLAN.md with its acceptance test as the definition of done.
---

# Phase Execution Workflow

Use this skill when starting or resuming a build phase of the DynHand
project.

## Before writing code

1. Read `agents/00_INDEX.md` through the phase's referenced spec files.
   The phase plan names the files; read them, do not work from memory.
2. Read `agents/12_NON_GOALS_AND_CUT_SCOPE.md` if the phase tempts any
   ambitious addition. If the idea is listed there, do not build it.
3. Check `agents/03_EXISTING_REPOS_TO_CLONE.md`. If a verified repo solves
   the subproblem, clone via `scripts/setup_references.sh` and adapt,
   never re-derive.
4. Confirm the previous phase's acceptance test actually passes by running
   its test file, not by trusting a summary.

## While implementing

- Track the phase as todos; one todo per task in the phase plan.
- Follow `agents/13_REPRODUCIBILITY_AND_CONVENTIONS.md` for config schema,
  seeding, logging, and test placement.
- Keep the five ablation conditions config-only. If you are about to write
  `if condition == "E"`, stop and re-read `agents/05_RL_ALGORITHM_SPEC.md`
  section 4.

## Definition of done

1. The phase's acceptance criterion passes and is encoded as a pytest file
   under `tests/python/` named for the phase.
2. `uv run pytest` and `uv run ruff check .` pass.
3. The smoke run `configs/smoke.yaml` still trains end to end if the
   trainer was touched.
4. Results land in `results/<experiment_id>/` with the manifest files.
5. A short summary states what passed, what deviated from spec, and what
   the next phase is.
