"""Tests for the SB3 cross-check wrapper (SB3 itself only used via monkeypatch)."""

from pathlib import Path

from dynhand.evaluation import sb3_check


class FakeModel:
    saved_to = None

    def learn(self, total_timesteps: int) -> None:
        assert total_timesteps > 0

    def save(self, path: str) -> None:
        FakeModel.saved_to = path


def test_run_sb3_check_writes_manifest_and_metrics(tmp_path: Path, monkeypatch) -> None:
    import gymnasium

    fake_sac_module = type("SAC_mod", (), {})()
    fake_sac_module.SAC = lambda *a, **k: FakeModel()

    class FakeVecEnv(list):
        def close(self):
            pass

    fake_sb3_common = type("sb3_common", (), {})()
    fake_sb3_common.make_vec_env = lambda *a, **k: FakeVecEnv()
    fake_sb3_common.SubprocVecEnv = FakeVecEnv
    fake_sb3_common.Monitor = lambda env: env
    fake_sb3_common.evaluate_policy = lambda model, env, n_eval_episodes: (3.5, 0.5)

    fake_sb3 = type("sb3", (), {})()
    fake_sb3.SAC = fake_sac_module.SAC
    fake_sac_module.SAC.__module__ = "stable_baselines3"

    class FakeGymEnv:
        def close(self) -> None:
            pass

    import sys

    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", type("m", (), {})())
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.evaluation", fake_sb3_common
    )
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.monitor", fake_sb3_common
    )
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.vec_env", fake_sb3_common
    )
    monkeypatch.setattr(gymnasium, "make", lambda env_id: FakeGymEnv())

    metrics = sb3_check.run_sb3_check(
        env_id="Pendulum-v1",
        total_timesteps=1000,
        seed=0,
        num_envs=2,
        output_dir=str(tmp_path),
    )
    assert metrics["eval_return_mean"] == 3.5
    assert metrics["eval_return_std"] == 0.5
    run_dir = tmp_path / "sb3_check"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert FakeModel.saved_to.startswith(str(run_dir))
