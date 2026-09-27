"""Diagnostic: how much signal is left for the residual dynamics head?

The residual ensemble is trained to predict

    residual = true_delta - physics_delta

so if the nominal physics delta is correct, the residual it must learn should be
near zero.  A large residual means the physics term is wrong and the ensemble is
being asked to compensate for a broken baseline, which is what happened before
the ctrl/quaternion/base-velocity fix.

Run directly:

    uv run python scripts/check_residual_target.py
"""

import numpy as np

from human2robot.dynamics.nominal_physics import compute_obs_delta
from human2robot.envs.allegro import AllegroPickupEnv


def measure(steps: int = 200, seed: int = 0) -> dict[str, float]:
    """Roll out the real env and compare true deltas to nominal ones."""
    env = AllegroPickupEnv()
    try:
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        physics_errors = []
        residual_norms = []
        true_norms = []
        for _ in range(steps):
            action = rng.uniform(-1.0, 1.0, 22).astype(np.float32)
            nominal = compute_obs_delta(env.model, obs, action)
            next_obs, _, _, _, _ = env.step(action)
            true_delta = next_obs - obs
            physics_errors.append(np.abs(nominal - true_delta).max())
            residual_norms.append(float(np.linalg.norm(true_delta - nominal)))
            true_norms.append(float(np.linalg.norm(true_delta)))
            obs = next_obs
    finally:
        env.close()

    physics_errors = np.asarray(physics_errors)
    residual_norms = np.asarray(residual_norms)
    true_norms = np.asarray(true_norms)
    return {
        "steps": float(steps),
        "physics_max_abs_err": float(physics_errors.max()),
        "physics_mean_abs_err": float(physics_errors.mean()),
        "residual_norm_mean": float(residual_norms.mean()),
        "true_delta_norm_mean": float(true_norms.mean()),
        "residual_fraction": float(
            residual_norms.mean() / max(true_norms.mean(), 1e-12)
        ),
    }


def main() -> None:
    result = measure()
    print(f"steps                     {result['steps']:.0f}")
    print(f"physics max abs error     {result['physics_max_abs_err']:.3e}")
    print(f"physics mean abs error    {result['physics_mean_abs_err']:.3e}")
    print(f"true delta norm (mean)    {result['true_delta_norm_mean']:.4f}")
    print(f"residual norm (mean)      {result['residual_norm_mean']:.4f}")
    print(f"residual / true           {result['residual_fraction']:.4%}")
    if result["residual_fraction"] < 0.01:
        print(
            "VERDICT: physics delta is correct; the residual head has little to learn"
        )
    else:
        print("VERDICT: residual is still large; the physics term is not trustworthy")


if __name__ == "__main__":
    main()
