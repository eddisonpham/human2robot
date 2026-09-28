# Finding: resume corrupted the metrics streams, twice

Date: 2026-09-27. Found while cleaning the repository, not by a training symptom.

## The defect

Resuming a run rewinds the trainer to a checkpoint, but the metrics file was
never rewound to match. Training then re-executed the steps between the
checkpoint and the last recorded metric, appending a second record for steps
that already had one. The audit tool correctly reported this:

```
AuditReport(eval_lines=103, malformed_lines=0,
            duplicate_steps=[1560000, 1580004, 1760004],
            out_of_order_transitions=1)
```

Six of the twelve Tier B runs were affected, which is every run that had been
restarted at least once. `human2robot-plot` and `human2robot-benchmark` both
refuse an unhealthy stream, so the affected runs could not be plotted or
benchmarked at all.

Any summary computed over the stream was also slightly wrong: Condition A seed 0
reported 103 evaluations where it had 100, because duplicate evaluation steps
were counted twice.

## The fix

`RunRecorder.rewind_metrics(step)` drops records beyond the checkpoint step and
is called during resume. Future resumes produce monotonic, duplicate-free
streams.

## The mistake

Repairing the existing files, I deduplicated on `step` alone. That is wrong:
training metrics and evaluation metrics are logged at the same step values, so
keeping the first record at each step discarded every evaluation record that
shared a step with a training record. All evaluation returns were destroyed
across all twelve runs.

`qf_loss`, `actor_loss`, `alpha`, and `q_mean` survived, since those were the
records being kept. The critic-health columns in `RL_RESULTS.md` were
unaffected.

## The recovery

The training loop writes every evaluation to both `metrics.jsonl` and a
TensorBoard event file, under the tag `eval/return_mean`. That redundancy is
what made the loss recoverable.

`scripts/rebuild_eval_metrics.py` reads every event file in a run, merges the
scalar series, and re-inserts the records, ordering training and evaluation
records independently so they no longer collide.

All 100 evaluation points were recovered for every completed run. The recovered
mean for Condition A seed 0 is -46.04, which matches the value computed from
the intact stream before the damage, confirming the recovery is exact.

**Only `eval_return_mean` is recoverable.** TensorBoard never saw the return
standard deviation or the episode length, so `eval_return_std` and
`eval_episode_steps` are gone from the affected runs. Neither appears in any
reported result.

## Corrected numbers

Deduplicating changes the affected figures slightly:

| Run | Before | After |
| --- | --- | --- |
| A s0 evals | 103 | 100 |
| A s0 mean | -44.9 | -46.0 |
| A condition mean | -48.9 | -49.3 |

The A-versus-B conclusion is unchanged: -49.2 versus -51.0 across three seeds
each, still a null result.

## Lesson

The repo now logs evaluation to two independent sinks precisely because a
single one proved insufficient. Any repair script touching a metrics stream must
key on record type as well as step, and should be written as a tested,
recoverable operation rather than an ad-hoc rewrite of live results.
