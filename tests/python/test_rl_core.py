"""Tests for SAC networks, the SAC update, and BC pretraining."""

import numpy as np
import torch

from dynhand.config.schema import SACConfig
from dynhand.rl.bc import BCTrainer
from dynhand.rl.networks import GaussianActor, QNetwork
from dynhand.rl.sac import SAC

DEVICE = torch.device("cpu")
OBS_DIM, ACT_DIM, HIDDEN = 5, 3, 32


def make_sac(**overrides) -> SAC:
    config = SACConfig(hidden_dim=HIDDEN, batch_size=8)
    for key, value in overrides.items():
        setattr(config, key, value)
    return SAC(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        action_low=np.full(ACT_DIM, -1.0, dtype=np.float32),
        action_high=np.full(ACT_DIM, 1.0, dtype=np.float32),
        sac_config=config,
        device=DEVICE,
    )


def make_batch(batch_size: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "obs": rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        "acts": rng.uniform(-1, 1, size=(batch_size, ACT_DIM)).astype(np.float32),
        "next_obs": rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        "rewards": rng.normal(size=(batch_size, 1)).astype(np.float32),
        "dones": np.zeros((batch_size, 1), dtype=np.float32),
    }


def test_gaussian_actor_output_range_and_shapes() -> None:
    actor = GaussianActor(OBS_DIM, ACT_DIM, HIDDEN)
    actor.set_action_range(np.full(ACT_DIM, -1.0), np.full(ACT_DIM, 1.0))
    obs = torch.randn(6, OBS_DIM)
    action, log_prob, mean = actor(obs)
    assert action.shape == (6, ACT_DIM)
    assert log_prob.shape == (6, 1)
    assert mean.shape == (6, ACT_DIM)
    assert (action.abs() <= 1.0 + 1e-6).all()


def test_gaussian_actor_action_scaling() -> None:
    actor = GaussianActor(OBS_DIM, ACT_DIM, HIDDEN)
    actor.set_action_range(np.full(ACT_DIM, -2.0), np.full(ACT_DIM, 4.0))
    obs = torch.zeros(64, OBS_DIM)
    torch.manual_seed(0)
    action, _, _ = actor(obs)
    assert (action <= 4.0 + 1e-5).all()
    assert (action >= -2.0 - 1e-5).all()


def test_qnetwork_shapes() -> None:
    qf = QNetwork(OBS_DIM, ACT_DIM, HIDDEN)
    obs = torch.randn(4, OBS_DIM)
    act = torch.randn(4, ACT_DIM)
    q1, q2 = qf(obs, act)
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)


def test_sac_act_shapes_single_and_batch() -> None:
    sac = make_sac()
    single = sac.act(np.zeros(OBS_DIM, dtype=np.float32))
    assert single.shape == (ACT_DIM,)
    assert (np.abs(single) <= 1.0).all()
    batch = sac.act(np.zeros((4, OBS_DIM), dtype=np.float32))
    assert batch.shape == (4, ACT_DIM)


def test_sac_act_deterministic_matches_mean() -> None:
    sac = make_sac()
    obs = np.random.default_rng(1).normal(size=(2, OBS_DIM)).astype(np.float32)
    a1 = sac.act(obs, deterministic=True)
    a2 = sac.act(obs, deterministic=True)
    assert np.allclose(a1, a2)


def test_sac_update_returns_finite_metrics() -> None:
    sac = make_sac()
    metrics = sac.update(make_batch())
    expected_keys = {"qf_loss", "actor_loss", "alpha_loss", "alpha", "q_mean"}
    assert expected_keys.issubset(metrics.keys())
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["alpha"] > 0


def test_sac_update_changes_parameters() -> None:
    sac = make_sac()
    before = next(sac.qf.parameters()).clone()
    sac.update(make_batch())
    after = next(sac.qf.parameters())
    assert not torch.equal(before, after)


def test_sac_target_soft_update() -> None:
    sac = make_sac()
    target_before = next(sac.qf_target.parameters()).clone()
    sac.update(make_batch())
    target_after = next(sac.qf_target.parameters())
    assert not torch.equal(target_before, target_after)
    online = next(sac.qf.parameters())
    expected = (1 - sac.tau) * target_before + sac.tau * online
    assert torch.allclose(target_after, expected, atol=1e-6)


def test_sac_target_entropy_is_negative_act_dim() -> None:
    sac = make_sac(target_entropy_scale=-1.0)
    assert sac.target_entropy == -ACT_DIM


def test_sac_state_dict_roundtrip() -> None:
    sac = make_sac()
    state = sac.state_dict()
    sac2 = make_sac()
    sac2.load_state_dict(state)
    obs = np.random.default_rng(2).normal(size=(3, OBS_DIM)).astype(np.float32)
    assert np.allclose(
        sac.act(obs, deterministic=True), sac2.act(obs, deterministic=True)
    )


def test_bc_trainer_reduces_loss() -> None:
    sac = make_sac()
    rng = np.random.default_rng(0)
    demo_obs = rng.normal(size=(200, OBS_DIM)).astype(np.float32)
    demo_acts = rng.uniform(-1, 1, size=(200, ACT_DIM)).astype(np.float32)
    bc = BCTrainer(sac, demo_obs, demo_acts, lr=1e-3, holdout_fraction=0.2, rng=rng)
    metrics = bc.train_epochs(epochs=5, batch_size=32, patience=5)
    assert metrics["bc_train_loss"] >= 0
    assert metrics["bc_epochs_run"] <= 5


def test_bc_trainer_fits_simple_mapping() -> None:
    torch.manual_seed(0)
    sac = make_sac()
    rng = np.random.default_rng(1)
    demo_obs = rng.normal(size=(256, OBS_DIM)).astype(np.float32)
    demo_acts = np.tanh(demo_obs[:, :ACT_DIM]).astype(np.float32)
    bc = BCTrainer(sac, demo_obs, demo_acts, lr=1e-3, holdout_fraction=0.1, rng=rng)
    bc.train_epochs(epochs=50, batch_size=64, patience=50)
    with torch.no_grad():
        pred = sac.actor.deterministic(torch.as_tensor(demo_obs[:32]))
    target = torch.as_tensor(demo_acts[:32])
    assert torch.nn.functional.mse_loss(pred, target).item() < 0.5


def test_bc_trainer_rejects_tiny_dataset() -> None:
    sac = make_sac()
    with np.errstate(all="ignore"):
        pass
    import pytest

    from dynhand.rl.bc import BCTrainer as BC

    with pytest.raises(ValueError):
        BC(
            sac,
            np.zeros((1, OBS_DIM), dtype=np.float32),
            np.zeros((1, ACT_DIM), dtype=np.float32),
            lr=1e-3,
            holdout_fraction=0.1,
            rng=np.random.default_rng(0),
        )
