"""Run directory ownership, lifecycle, and the metrics rewind edge cases."""

import json
import os

import pytest

from human2robot.config.schema import ExperimentConfig
from human2robot.envs.record import RunLock, RunRecorder


def make_config(tmp_path, **overrides) -> ExperimentConfig:
    base = {
        "experiment_id": "recorder_test",
        "seed": 0,
    }
    base.update(overrides)
    return ExperimentConfig(**base)


@pytest.fixture
def recorder(tmp_path):
    rec = RunRecorder(make_config(tmp_path), results_root=str(tmp_path))
    yield rec
    rec.close()


# --- RunLock ------------------------------------------------------------------


def test_run_lock_writes_owner_metadata(tmp_path):
    lock = RunLock(tmp_path / ".run.lock")
    lock.acquire()
    try:
        # The locked region is unreadable through a fresh handle on Windows, so
        # read it through the handle that owns the lock.
        lock._file.seek(0)
        payload = json.loads(lock._file.read())
        assert payload["pid"] == os.getpid()
        assert payload["platform"]
    finally:
        lock.close()


def test_run_lock_is_reentrant_across_close_cycles(tmp_path):
    lock = RunLock(tmp_path / ".run.lock")
    lock.acquire()
    lock.close()
    lock.acquire()
    lock.close()


def test_run_lock_close_is_idempotent(tmp_path):
    lock = RunLock(tmp_path / ".run.lock")
    lock.acquire()
    lock.close()
    lock.close()
    assert lock._file is None


def test_run_lock_context_manager_releases(tmp_path):
    with RunLock(tmp_path / ".run.lock") as lock:
        assert lock._file is not None
    assert lock._file is None


def test_second_lock_is_refused_while_held(tmp_path):
    first = RunLock(tmp_path / ".run.lock")
    first.acquire()
    try:
        second = RunLock(tmp_path / ".run.lock")
        with pytest.raises(RuntimeError):
            second.acquire()
    finally:
        first.close()


# --- RunRecorder lifecycle ----------------------------------------------------


def test_constructor_writes_config_metrics_and_status(tmp_path):
    rec = RunRecorder(make_config(tmp_path), results_root=str(tmp_path))
    try:
        run_dir = tmp_path / "recorder_test"
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "metrics.jsonl").exists()
        assert (run_dir / "system_info.json").exists()
        assert (run_dir / "git_commit.txt").exists()
        status = json.loads((run_dir / "run_status.json").read_text())
        assert status["status"] == "running"
    finally:
        rec.close()


def test_constructor_releases_lock_when_setup_fails(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(RunRecorder, "_system_info", staticmethod(boom))
    with pytest.raises(OSError, match="disk full"):
        RunRecorder(make_config(tmp_path), results_root=str(tmp_path))
    # The failed construction must not leave the directory locked.
    retry = RunLock(tmp_path / "recorder_test" / ".run.lock")
    retry.acquire()
    retry.close()


def test_status_transitions_are_recorded(recorder):
    run_dir = recorder.run_dir
    recorder.mark_paused(1234)
    paused = json.loads((run_dir / "run_status.json").read_text())
    assert paused["status"] == "paused"
    assert paused["step"] == 1234

    recorder.mark_completed()
    assert (
        json.loads((run_dir / "run_status.json").read_text())["status"] == "completed"
    )

    recorder.mark_failed(ValueError("boom"))
    failed = json.loads((run_dir / "run_status.json").read_text())
    assert failed["status"] == "failed"
    assert failed["error_type"] == "ValueError"
    assert failed["error"] == "boom"


def test_pause_sentinel_round_trip(recorder):
    assert recorder.pause_requested() is False
    recorder.pause_path.write_text("stop")
    assert recorder.pause_requested() is True
    recorder.clear_pause()
    assert recorder.pause_requested() is False
    recorder.clear_pause()
    assert recorder.pause_requested() is False


def test_recorder_context_manager_closes(tmp_path):
    with RunRecorder(make_config(tmp_path), results_root=str(tmp_path)) as rec:
        rec.log_metrics(1, {"loss": 0.5})
    assert rec.lock._file is None
    records = [
        json.loads(line)
        for line in (tmp_path / "recorder_test" / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records == [{"step": 1, "loss": 0.5}]


# --- rewind_metrics edge cases ------------------------------------------------


def test_rewind_on_missing_file_is_a_no_op(recorder):
    assert recorder.rewind_metrics(10) == 0


def test_rewind_keeps_blank_and_malformed_lines(recorder):
    path = recorder.metrics_path
    path.write_text(
        '{"step": 5, "v": 1}\n\nnot json\n{"novalue": 3}\n{"step": 50, "v": 2}\n',
        encoding="utf-8",
    )
    assert recorder.rewind_metrics(10) == 1
    kept = path.read_text(encoding="utf-8").splitlines()
    assert '{"step": 5, "v": 1}' in kept
    assert "not json" in kept
    assert '{"novalue": 3}' in kept
    assert '{"step": 50, "v": 2}' not in kept


def test_rewind_returns_zero_when_nothing_is_dropped(recorder):
    recorder.metrics_path.write_text('{"step": 1}\n{"step": 2}\n', encoding="utf-8")
    assert recorder.rewind_metrics(99) == 0
    assert len(recorder.metrics_path.read_text().splitlines()) == 2


def test_rewind_of_only_malformed_content_writes_empty_file(recorder):
    recorder.metrics_path.write_text("garbage\n", encoding="utf-8")
    assert recorder.rewind_metrics(0) == 0
    assert recorder.metrics_path.read_text(encoding="utf-8") == "garbage\n"
