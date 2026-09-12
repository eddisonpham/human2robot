"""Experiment output recording and process-safe run ownership."""

import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import torch
import yaml

from dynhand.config.schema import ExperimentConfig
from dynhand.utils.git_info import get_git_commit


class RunLock:
    """Hold an operating-system file lock for one experiment directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: IO[str] | None = None

    def acquire(self) -> None:
        """Acquire the lock or raise with the current owner details."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+", encoding="utf-8")
        if self.path.stat().st_size == 0:
            file.write(" ")
            file.flush()
        file.seek(0)
        try:
            self._lock_file(file)
        except (BlockingIOError, OSError) as exc:
            file.close()
            raise RuntimeError("run is already locked by another process") from exc
        file.seek(0)
        file.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "python": sys.executable,
                "platform": platform.platform(),
            },
            file,
        )
        file.flush()
        self._file = file

    def close(self) -> None:
        """Release the lock and close its file handle."""
        if self._file is None:
            return
        try:
            self._unlock_file(self._file)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _lock_file(file: IO[str]) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock_file(file: IO[str]) -> None:
        if os.name == "nt":
            import msvcrt

            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


class RunRecorder:
    """Create a run directory and persist config, metrics, and manifest."""

    def __init__(self, config: ExperimentConfig, results_root: str = "results") -> None:
        self.run_dir = Path(results_root) / config.experiment_id
        self.lock = RunLock(self.run_dir / ".run.lock")
        self.lock.acquire()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(config.model_dump(), f, sort_keys=True)
            self.metrics_path = self.run_dir / "metrics.jsonl"
            self.metrics_path.touch(exist_ok=True)
            with open(self.run_dir / "git_commit.txt", "w", encoding="utf-8") as f:
                f.write(get_git_commit() + "\n")
            with open(self.run_dir / "system_info.json", "w", encoding="utf-8") as f:
                json.dump(self._system_info(), f, indent=2)
            self._write_status("running")
        except Exception:
            self.lock.close()
            raise

    def mark_completed(self) -> None:
        """Record successful completion without releasing run ownership."""
        self._write_status("completed")

    def mark_failed(self, error: BaseException) -> None:
        """Record a failed run and its error type before cleanup."""
        self._write_status(
            "failed",
            {"error_type": type(error).__name__, "error": str(error)},
        )

    def close(self) -> None:
        """Release ownership of the run directory."""
        self.lock.close()

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Append one metrics record while the run lock is held."""
        record = {"step": step, **metrics}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_checkpoint(self, step: int, state: dict) -> Path:
        """Atomically save one checkpoint under the run's checkpoint directory."""
        path = self.checkpoint_dir / f"step_{step}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_path = self.checkpoint_dir / "checksums.json"
        with open(checksum_path, "a", encoding="utf-8") as file:
            record = {"step": step, "file": path.name, "sha256": digest}
            file.write(json.dumps(record) + "\n")
        return path

    def latest_checkpoint(self) -> Path | None:
        """Return the newest checkpoint path, or None if there are none."""

        def step_of(path: Path) -> int:
            try:
                return int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                return -1

        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"), key=step_of)
        return checkpoints[-1] if checkpoints else None

    def _write_status(self, status: str, extra: dict | None = None) -> None:
        """Write the current lifecycle state for operational inspection."""
        payload = {
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
        }
        if extra:
            payload.update(extra)
        temporary = self.run_dir / "run_status.json.tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        temporary.replace(self.run_dir / "run_status.json")

    @staticmethod
    def _system_info() -> dict:
        info = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = str(torch.cuda.get_device_capability(0))
        return info
