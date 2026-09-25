"""Capture git provenance for the experiment manifest."""

import subprocess
from pathlib import Path


def get_git_commit(repo_dir: str | Path = ".") -> str:
    """Return the current git SHA plus a dirty flag, or 'unknown'."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"
