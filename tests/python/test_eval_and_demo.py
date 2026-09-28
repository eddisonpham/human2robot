"""Evaluation env construction, demo loading helpers, and the DexYCB CLI."""

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


# --- dexycb ------------------------------------------------------------------
#
# The basis retargeter and its demo-building CLI were removed; see
# docs/FINDINGS_retargeting.md. Retargeting is scripts/retarget_dexycb_ik.py.
# These cover the sequence discovery and MANO loading that remain, and pin the
# removal so the degenerate path cannot be reintroduced quietly.


def test_discover_sequences_finds_nested_pose_files(tmp_path):
    for name in ("seq_b", "seq_a"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pose.npz").write_bytes(b"")
    (tmp_path / "seq_a" / "other.npz").write_bytes(b"")
    found = dexycb_mod.discover_sequences(tmp_path)
    assert [p.parent.name for p in found] == ["seq_a", "seq_b"]


def test_discover_sequences_on_empty_dir_returns_empty(tmp_path):
    assert dexycb_mod.discover_sequences(tmp_path) == []


def test_load_sequence_reads_mano_and_meta(tmp_path):
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    pose = np.zeros((5, 1, 51), dtype=np.float32)
    pose[:, 0, 0] = 0.1
    np.savez(seq_dir / "pose.npz", pose_m=pose, pose_y=np.zeros((5, 1, 8)))
    (seq_dir / "meta.yml").write_text("num_frames: 3\nother: 1\n")
    loaded = dexycb_mod.load_sequence(seq_dir / "pose.npz")
    assert loaded["pose_m"].shape == (3, 51)
    assert loaded["sequence_id"] == "seq"


def test_load_sequence_without_meta_keeps_every_frame(tmp_path):
    np.savez(
        tmp_path / "pose.npz",
        pose_m=np.zeros((4, 1, 51), dtype=np.float32),
        pose_y=np.zeros((4, 1, 8)),
    )
    assert dexycb_mod.load_sequence(tmp_path / "pose.npz")["pose_m"].shape == (4, 51)


def test_the_removed_retargeter_raises_rather_than_regenerating_demos():
    with pytest.raises(NotImplementedError, match="not a retargeting"):
        dexycb_mod._mano_to_joint_targets(np.zeros((4, 51)))


def test_the_module_no_longer_exposes_a_demo_building_cli():
    """The CLI could only produce degenerate demos, so it is gone."""
    for gone in ("main", "build_subject_demos", "sequence_to_demo"):
        assert not hasattr(dexycb_mod, gone), gone
