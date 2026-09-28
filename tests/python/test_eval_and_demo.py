"""Evaluation env construction, demo loading helpers, and the DexYCB CLI."""

import json
from pathlib import Path

import numpy as np
import pytest

from human2robot.data import dexycb as dexycb_mod
from human2robot.evaluation.evaluate import build_single_env, evaluate
from human2robot.rl.demo import _flatten_obs, _load_local_demos, seed_replay_buffer

# --- evaluate.build_single_env ------------------------------------------------


def test_allegro_env_is_constructed_directly():
    from human2robot.envs.allegro import AllegroPickupEnv

    env = build_single_env("Human2Robot-AllegroPickup-v0")
    try:
        assert isinstance(env, AllegroPickupEnv)
    finally:
        env.close()


def test_dict_observation_env_gets_flattened():
    env = build_single_env("Human2Robot-AllegroPickup-v0")
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
    finally:
        env.close()
    assert obs_dim > 0


class _FakeSAC:
    """Minimal SAC stand-in that returns a fixed action."""

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def act(self, obs, deterministic=True):
        self.calls += 1
        return self.action


def test_evaluate_returns_expected_keys_and_counts_steps():
    env = build_single_env("Human2Robot-AllegroPickup-v0")
    try:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
    finally:
        env.close()
    sac = _FakeSAC(action)
    metrics = evaluate(sac, "Human2Robot-AllegroPickup-v0", episodes=2, seed=3)
    assert set(metrics) == {
        "eval_return_mean",
        "eval_return_std",
        "eval_episode_steps",
    }
    assert sac.calls == int(metrics["eval_episode_steps"]) * 2
    assert metrics["eval_return_std"] >= 0.0


# --- demo helpers -------------------------------------------------------------


def test_flatten_obs_passes_arrays_through():
    arr = np.zeros((4, 7), dtype=np.float32)
    assert _flatten_obs(arr) is arr


def test_flatten_obs_concatenates_dict_values():
    obs = {
        "a": np.zeros((4, 3), dtype=np.float32),
        "b": np.ones((4, 2), dtype=np.float32),
    }
    flat = _flatten_obs(obs)
    assert flat.shape == (4, 5)
    np.testing.assert_allclose(flat[:, 3:], 1.0)


def test_flatten_obs_rejects_unknown_type():
    with pytest.raises(TypeError, match="unexpected observation type"):
        _flatten_obs("not an observation")


def test_seed_replay_buffer_delegates_to_add_batch():
    calls = {}

    class _Buffer:
        def add_batch(self, obs, acts, next_obs, rewards, dones):
            calls["n"] = len(obs)
            calls["shapes"] = (acts.shape, next_obs.shape, rewards.shape, dones.shape)

    demos = {
        "obs": np.zeros((9, 22), dtype=np.float32),
        "acts": np.zeros((9, 22), dtype=np.float32),
        "next_obs": np.zeros((9, 22), dtype=np.float32),
        "rewards": np.zeros((9, 1), dtype=np.float32),
        "dones": np.zeros((9, 1), dtype=np.float32),
    }
    assert seed_replay_buffer(_Buffer(), demos) == 9
    assert calls["n"] == 9
    assert calls["shapes"] == ((9, 22), (9, 22), (9, 1), (9, 1))


def test_load_local_demos_reads_demo_directory(tmp_path):
    for i in range(2):
        np.savez_compressed(
            tmp_path / f"demo_{i:04d}.npz",
            q=np.zeros((4, 22), dtype=np.float32),
            qdot=np.zeros((4, 22), dtype=np.float32),
            a_demo=np.zeros((4, 22), dtype=np.float32),
            object_pose=np.zeros((4, 7), dtype=np.float32),
            object_vel=np.zeros((4, 6), dtype=np.float32),
            contact=np.zeros((4, 5), dtype=np.float32),
            trajectory_id=np.array(f"t{i}"),
            task_id=np.array("allegro_pickup"),
            source=np.array("unit-test"),
            schema_version=np.array("1.0"),
        )
    out = _load_local_demos(str(tmp_path))
    assert set(out) == {"obs", "acts", "next_obs", "rewards", "dones"}
    assert len(out["obs"]) > 0


# --- dexycb CLI ---------------------------------------------------------------


def test_dexycb_cli_rejects_missing_subject_dir(tmp_path, capsys):
    rc = dexycb_mod.main(
        ["--subject-dir", str(tmp_path / "absent"), "--output-dir", str(tmp_path / "o")]
    )
    assert rc == 1
    assert "subject dir not found" in capsys.readouterr().err


def test_dexycb_cli_rejects_directory_without_sequences(tmp_path, capsys):
    subject = tmp_path / "subject"
    subject.mkdir()
    rc = dexycb_mod.main(
        ["--subject-dir", str(subject), "--output-dir", str(tmp_path / "o")]
    )
    assert rc == 1
    assert "no pose.npz sequences" in capsys.readouterr().err


def test_dexycb_cli_defaults_subject_dir_under_project_root(capsys):
    """With no arguments the subject path must resolve under the repo root.

    Asserted through the missing-directory error rather than a successful run,
    so the test does not depend on the DexYCB download being present.
    """
    project_root = Path(dexycb_mod.__file__).resolve().parents[3]
    assert dexycb_mod.main(["--subject", "no-such-subject"]) == 1
    err = capsys.readouterr().err
    assert str(project_root / "data" / "raw" / "dexycb" / "no-such-subject") in err


def test_dexycb_cli_defaults_output_dir_under_project_root(monkeypatch, tmp_path):
    captured = {}
    subject = tmp_path / "sub"
    subject.mkdir()
    monkeypatch.setattr(dexycb_mod, "discover_sequences", lambda d: [Path("x")])
    monkeypatch.setattr(
        dexycb_mod,
        "build_subject_demos",
        lambda s, o, seed=0, max_count=None: (
            captured.update(
                subject=Path(s), out=Path(o), seed=seed, max_count=max_count
            )
            or []
        ),
    )
    rc = dexycb_mod.main(
        ["--subject-dir", str(subject), "--seed", "5", "--max-count", "2"]
    )
    assert rc == 0
    assert captured["subject"] == subject
    assert captured["out"].as_posix().endswith("data/demonstrations_dexycb")
    assert captured["seed"] == 5
    assert captured["max_count"] == 2


def test_dexycb_cli_refuses_to_overwrite_the_default_subject_demos(
    monkeypatch, tmp_path, capsys
):
    """A non-default subject must not silently replace subject-01's demos.

    `--subject` selects the input but the default `--output-dir` holds
    subject-01's demos, so honouring the default would destroy them.
    """
    subject = tmp_path / "s2"
    subject.mkdir()
    called = []
    monkeypatch.setattr(dexycb_mod, "discover_sequences", lambda d: [Path("x")])
    monkeypatch.setattr(
        dexycb_mod, "build_subject_demos", lambda *a, **k: called.append(a) or []
    )
    rc = dexycb_mod.main(
        ["--subject-dir", str(subject), "--subject", "20200813-subject-02"]
    )
    assert rc == 1
    assert called == []
    assert "--output-dir is required" in capsys.readouterr().err


def test_dexycb_cli_allows_a_second_subject_with_explicit_output(monkeypatch, tmp_path):
    captured = {}
    subject = tmp_path / "sub"
    subject.mkdir()
    monkeypatch.setattr(dexycb_mod, "discover_sequences", lambda d: [Path("x")])
    monkeypatch.setattr(
        dexycb_mod,
        "build_subject_demos",
        lambda s, o, seed=0, max_count=None: (
            captured.update(subject=Path(s), out=Path(o)) or []
        ),
    )
    out = tmp_path / "s2"
    rc = dexycb_mod.main(
        [
            "--subject-dir",
            str(subject),
            "--subject",
            "20200813-subject-02",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    assert captured["out"] == out


def test_dexycb_cli_honours_explicit_paths(monkeypatch, tmp_path):
    captured = {}
    subject = tmp_path / "sub"
    subject.mkdir()
    monkeypatch.setattr(dexycb_mod, "discover_sequences", lambda d: [Path("x")])
    monkeypatch.setattr(
        dexycb_mod,
        "build_subject_demos",
        lambda s, o, seed=0, max_count=None: (
            captured.update(subject=Path(s), out=Path(o)) or []
        ),
    )
    out = tmp_path / "custom_out"
    assert (
        dexycb_mod.main(["--subject-dir", str(subject), "--output-dir", str(out)]) == 0
    )
    assert captured["subject"] == subject
    assert captured["out"] == out


def test_build_subject_demos_writes_manifest(tmp_path, monkeypatch):
    traj = dexycb_mod.DemoTrajectory(
        q=np.zeros((4, 22), dtype=np.float32),
        qdot=np.zeros((4, 22), dtype=np.float32),
        a_demo=np.zeros((4, 22), dtype=np.float32),
        object_pose=np.zeros((4, 7), dtype=np.float32),
        object_vel=np.zeros((4, 6), dtype=np.float32),
        contact=np.zeros((4, 5), dtype=np.float32),
        trajectory_id="t0",
        task_id="allegro_pickup",
        source="unit-test",
    )
    monkeypatch.setattr(
        dexycb_mod, "discover_sequences", lambda d: [Path("a"), Path("b")]
    )
    monkeypatch.setattr(dexycb_mod, "sequence_to_demo", lambda p: traj)
    out = tmp_path / "demos"
    written = dexycb_mod.build_subject_demos(tmp_path, out, seed=3)
    assert len(written) == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["count"] == 2
    assert manifest["rejected"] == 0
    assert manifest["rng_seed"] == 3
    assert (out / ".rng_check").exists()


def test_build_subject_demos_counts_rejections_and_honours_max_count(
    tmp_path, monkeypatch
):
    def boom(p):
        if str(p) == "b":
            raise ValueError("bad sequence")
        return dexycb_mod.DemoTrajectory(
            q=np.zeros((3, 22), dtype=np.float32),
            qdot=np.zeros((3, 22), dtype=np.float32),
            a_demo=np.zeros((3, 22), dtype=np.float32),
            object_pose=np.zeros((3, 7), dtype=np.float32),
            object_vel=np.zeros((3, 6), dtype=np.float32),
            contact=np.zeros((3, 5), dtype=np.float32),
            trajectory_id="t",
            task_id="allegro_pickup",
            source="unit-test",
        )

    monkeypatch.setattr(
        dexycb_mod, "discover_sequences", lambda d: [Path("a"), Path("b")]
    )
    monkeypatch.setattr(dexycb_mod, "sequence_to_demo", boom)
    out = tmp_path / "demos"
    written = dexycb_mod.build_subject_demos(tmp_path, out, max_count=1)
    assert len(written) == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed_used"] is False
