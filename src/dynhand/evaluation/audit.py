"""Health audit for recorded metrics files."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AuditReport:
    """Summary of structural checks for one metrics stream."""

    eval_lines: int = 0
    malformed_lines: int = 0
    duplicate_steps: list[int] = field(default_factory=list)
    out_of_order_transitions: int = 0

    @property
    def healthy(self) -> bool:
        """Return whether the stream is safe to use for analysis."""
        return (
            not self.duplicate_steps
            and self.malformed_lines == 0
            and self.out_of_order_transitions == 0
        )


def audit_metrics(path: str | Path, metric: str = "eval_return_mean") -> AuditReport:
    """Audit one JSONL metrics stream for single-writer consistency."""
    eval_steps: list[int] = []
    malformed = 0
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
                if metric in record:
                    eval_steps.append(int(record["step"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                malformed += 1

    counts: dict[int, int] = {}
    for step in eval_steps:
        counts[step] = counts.get(step, 0) + 1
    duplicates = sorted(step for step, count in counts.items() if count > 1)
    out_of_order = sum(
        1
        for previous, current in zip(eval_steps, eval_steps[1:], strict=False)
        if current < previous
    )
    return AuditReport(
        eval_lines=len(eval_steps),
        malformed_lines=malformed,
        duplicate_steps=duplicates,
        out_of_order_transitions=out_of_order,
    )


def summarize(run_dir: str | Path, metric: str = "eval_return_mean") -> AuditReport:
    """Audit the metrics stream inside a run directory."""
    return audit_metrics(Path(run_dir) / "metrics.jsonl", metric=metric)
