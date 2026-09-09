"""Tests for git provenance capture using a temporary repository."""

import subprocess
from pathlib import Path

from dynhand.utils.git_info import get_git_commit


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_clean_repo_returns_sha(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(
        tmp_path,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "--allow-empty",
        "-qm",
        "init",
    )
    commit = get_git_commit(tmp_path)
    assert commit != "unknown"
    assert "dirty" not in commit
    assert len(commit) == 40


def test_dirty_repo_returns_dirty_flag(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(
        tmp_path,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "--allow-empty",
        "-qm",
        "init",
    )
    (tmp_path / "untracked.txt").write_text("dirty")
    commit = get_git_commit(tmp_path)
    assert commit.endswith("-dirty")
    assert len(commit) == 46
