---
name: senior-staff-review
description: Apply senior staff software engineer review standards to any code change in this repo before declaring it done.
---

# Senior Staff Review

Apply this checklist to every change. A change is done when all items pass,
not when the code merely runs.

## Correctness

- The change does exactly what the requesting phase or task requires, no
  more. Anything extra is scope creep: refuse it and note it instead.
- Behavior is deterministic where the spec requires it (seeds, env resets,
  replay sampling). Any nondeterminism must be config-controlled.
- All hyperparameters come from the validated config schema, never literals
  in trainer or environment code.

## Design

- Module boundaries hold: `envs/` owns reward and observation, `dynamics/`
  produces transitions but never touches the optimizer, `rl/` has no
  condition-specific branches. Reject changes that leak across boundaries.
- Public functions have short docstrings; nothing else carries comments.
  No emoji, no em dashes in committed files.
- Type hints on every function signature. `ruff check` must pass.

## Tests

- Every functional change ships unit tests in the same commit. New behavior
  without tests is an incomplete change.
- Run `uv run pytest` (coverage gate at 90 percent) and make it pass.
- Prefer fast, seeded tests: tiny MLPs, tiny buffers, 3-step MuJoCo runs.
  The suite must finish in minutes, not hours.

## Reproducibility

- Any run-producing change writes config, metrics, git commit, and system
  info into `results/<experiment_id>/`.
- Experiment defaults live in `configs/*.yaml`; a config typo must fail at
  load time through the pydantic schema.

## Communication

- Update README when setup, usage, or layout changes. Keep it concise, no
  filler.
- Note any deviation from `agents/` specs in the final summary so the user
  can accept or reject it explicitly.
