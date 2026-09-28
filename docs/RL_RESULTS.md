# RL ablation: a null result

The SAC half consumes the converted trajectories and did not work. This is
reported as a null, not dropped. The conversion pipeline is the deliverable; see
[RESULTS.md](RESULTS.md).

## Result

Three conditions, three seeds each, 2,000,004 steps per run, on the simulated
Allegro pickup task. Evaluation return, higher is better:

| Condition | Seed means | Mean +/- sd |
| --- | --- | --- |
| A from-scratch SAC | -46.0, -46.8, -54.9 | -49.2 +/- 4.9 |
| B BC-init + demo replay | -49.1, -51.6, -52.2 | -51.0 +/- 1.6 |
| C dynamics augmentation | -43.9, -54.8, -41.9 | -46.9 +/- 6.9 |

**No run reaches a positive return.** The task is unsolved, so the experiment
never had the signal to separate conditions. C is nominally best, but its seeds
span -41.9 to -54.8, the same spread A shows, so that ordering is not
established.

Regenerate this table from the recorded metrics:

```bash
uv run python scripts/summarize_ablation.py --markdown
```

## Two results that reversed on inspection

**"BC-init beats from-scratch by 48 percent" did not survive a bug fix.** The
observation pipeline converted rotations by dividing by `2 * sin(angle)`, which
is singular at 180 degrees, and a free-falling object hits that routinely. The
affected dimensions feed the critic, so every critic target was poisoned.

| Measurement | Before | After |
| --- | --- | --- |
| max abs observation over 60k env steps | 6.35e8 | 71-84 |
| max `qf_loss` in a 2M-step run | 1.9e15 | 1,844 (worst of 9 runs) |

After the fix, A and B are indistinguishable.

**Residual augmentation's apparent weakness was a structurally impossible
target.** The hand-written nominal physics model disagrees with its own
simulator by a median 12.9 per step, in a file with 13 percent coverage. Having
been corrected, the residual turned out to be identically zero, because the
nominal model *is* the simulator. Conditions D and E therefore cannot produce a
result at any training length; see
[FINDINGS_residual_degeneracy.md](FINDINGS_residual_degeneracy.md).

## Reproducing

```bash
uv run human2robot-train --config configs/tier_b_pickup.yaml --seed 0
bash scripts/watchdog_runs.sh          # restart a run that stops writing metrics
uv run human2robot-eval --config <cfg> --checkpoint <ckpt>
```

Each run writes `results/<experiment_id>/` containing its config, metrics stream,
checkpoints, git commit, and system info. Training defaults to CPU deliberately;
see [FINDINGS_training_hang.md](FINDINGS_training_hang.md).

## Open issues

- The task is unsolved, which is the root problem. Making it solvable is a
  research project, and it is the precondition for the ablation meaning anything.
- D and E configs (`configs/tier_b_cond_d.yaml`, `tier_b_cond_e.yaml`) remain in
  the tree. A reader will assume they are runnable; they are structurally
  degenerate.
- Resume used to write duplicate metric records, silently corrupting 6 of 12
  runs. Fixed, and the metrics stream is now audited on load; see
  [FINDINGS_metrics_integrity.md](FINDINGS_metrics_integrity.md).
- The ONNX exporter still uses `torch.onnx.export`, deprecated in torch 2.9.
