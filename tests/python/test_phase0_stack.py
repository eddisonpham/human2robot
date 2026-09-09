"""Phase 0 smoke tests: torch/CUDA Blackwell support, mujoco, gymnasium-robotics."""

import mujoco
import numpy as np
import pytest
import torch

SPHERE_XML = """
<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="g" type="sphere" size="0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_torch_imports_and_versions() -> None:
    assert torch.__version__ >= "2.7.0"


def test_mlp_forward_backward_cpu() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    )
    out = model(torch.randn(4, 8))
    out.sum().backward()
    assert out.shape == (4, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_capability() -> None:
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) >= (12, 0), "Blackwell sm_120 requires torch >= 2.7 cu128"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mlp_forward_backward_cuda() -> None:
    device = torch.device("cuda")
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(device)
    out = model(torch.randn(4, 8, device=device))
    out.sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()


def test_mujoco_step_is_stable() -> None:
    model = mujoco.MjModel.from_xml_string(SPHERE_XML)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert data.qpos[2] < 1.0, "ball should fall under gravity"


def test_adroit_relocate_env_loads_and_steps() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    pytest.importorskip("gymnasium_robotics")
    env = gymnasium.make("AdroitHandRelocate-v1")
    obs, _ = env.reset(seed=0)
    obs2, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.action_space.shape == (30,)
    assert np.isfinite(obs2).all()
    env.close()


def test_minari_package_importable() -> None:
    minari = pytest.importorskip("minari")
    assert hasattr(minari, "load_dataset")


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="dex-retargeting requires Pinocchio, Windows wheels are not published",
)
def test_dex_retargeting_importable() -> None:
    pytest.importorskip("dex_retargeting")
    from dex_retargeting.retargeting_config import RetargetingConfig

    assert hasattr(RetargetingConfig, "load_from_file")
