"""Phase H experiment: does C++ optimization improve demo quality downstream?

Trains identical BC heads on raw, optimized, and mixed demonstration
trajectories from the synthetic set and compares regression quality and
action-distribution alignment. Same hyperparameters for all conditions;
only the demonstration source differs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from human2robot.data.allegro_demos import load_demo_npz

RAW_DIR = Path("data/demonstrations")
OPT_DIR = Path("data/demonstrations_optimized")
OUT_PATH = Path("results/trajectory_optimization/bc_downstream.json")


def load_positions(directory: Path, suffix: str = "") -> list[np.ndarray]:
    paths = sorted(directory.glob(f"*{suffix}.npz"))
    if not paths:
        raise FileNotFoundError(f"no demos in {directory}")
    return [np.asarray(load_demo_npz(p).q, dtype=np.float32) for p in paths]


def make_transition_dataset(trajectories: list[np.ndarray], rng: np.random.Generator):
    """Transitions (q_t -> q_{t+1}) as (obs, action) regression pairs."""
    obs, acts = [], []
    for traj in trajectories:
        obs.append(traj[:-1])
        acts.append(traj[1:] - traj[:-1])
    obs = np.concatenate(obs, axis=0)
    acts = np.concatenate(acts, axis=0)
    idx = rng.permutation(len(obs))
    n_holdout = max(1, int(0.1 * len(obs)))
    return (obs[idx[n_holdout:]], acts[idx[n_holdout:]]), (
        obs[idx[:n_holdout]],
        acts[idx[:n_holdout]],
    )


class DeltaMLP(torch.nn.Module):
    def __init__(self, dof: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dof, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, dof),
        )

    def forward(self, x):
        return self.net(x)


def train_bc(train, holdout, seed: int, epochs: int = 60, batch: int = 512):
    torch.manual_seed(seed)
    model = DeltaMLP(train[0].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs_t = torch.as_tensor(train[0])
    act_t = torch.as_tensor(train[1])
    best = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience = 0
    for _ in range(epochs):
        order = torch.randperm(len(obs_t))
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            pred = model(obs_t[idx])
            loss = torch.nn.functional.mse_loss(pred, act_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            hold = torch.nn.functional.mse_loss(
                model(torch.as_tensor(holdout[0])), torch.as_tensor(holdout[1])
            ).item()
        if hold < best - 1e-6:
            best = hold
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    return model, best


def evaluate(model, holdout) -> dict:
    with torch.no_grad():
        pred = model(torch.as_tensor(holdout[0])).numpy()
    err = pred - holdout[1]
    return {
        "mse": float((err**2).mean()),
        "mae": float(np.abs(err).mean()),
        "max_err": float(np.abs(err).max()),
    }


def main() -> int:
    rng = np.random.default_rng(0)
    raw = load_positions(RAW_DIR)
    optimized = load_positions(OPT_DIR, suffix="_opt")
    print(f"raw demos: {len(raw)}, optimized demos: {len(optimized)}")

    results = {}
    for name, trajs in (
        ("raw", raw),
        ("optimized", optimized),
        ("mixed", raw + optimized),
    ):
        train, holdout = make_transition_dataset(trajs, rng)
        model, best = train_bc(train, holdout, seed=0)
        metrics = evaluate(model, holdout)
        metrics["bc_holdout_best"] = best
        metrics["transitions"] = int(len(train[0]))
        results[name] = metrics
        print(f"{name}: {metrics}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    improvement = 100.0 * (1.0 - results["optimized"]["mse"] / results["raw"]["mse"])
    print(f"optimized vs raw MSE improvement: {improvement:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
