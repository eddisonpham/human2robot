"""Stable-Baselines3 SAC cross-check.

Independent known-good SAC implementation used to validate our environment
and trainer: if SB3 and our SAC diverge wildly on the same task and seed
budget, the bug is in our code. This is a debugging tool per
agents/09_PHASE_PLAN.md Phase 1, not the primary training path.

Run with:
  uv sync --group sb3
  uv run dynhand-sb3-check --config configs/tier_a_relocate.yaml
"""

import argparse


def run_sb3_check(
    env_id: str,
    total_timesteps: int,
    seed: int,
    num_envs: int,
    output_dir: str,
) -> dict[str, float]:
    """Train SB3's SAC and return final evaluation metrics."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, make_vec_env

    from dynhand.config.schema import ExperimentConfig
    from dynhand.envs.record import RunRecorder

    config = ExperimentConfig(experiment_id="sb3_check", env_id=env_id)
    recorder = RunRecorder(config, output_dir)

    env = make_vec_env(
        env_id,
        n_envs=num_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=Monitor,
    )
    model = SAC(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        buffer_size=1_000_000,
        batch_size=256,
        learning_rate=3e-4,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(str(recorder.run_dir / "sb3_sac"))

    eval_env = Monitor(__import__("gymnasium").make(env_id))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    eval_env.close()
    env.close()
    metrics = {
        "eval_return_mean": float(mean_reward),
        "eval_return_std": float(std_reward),
    }
    recorder.log_metrics(total_timesteps, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="SB3 SAC cross-check")
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()

    from dynhand.config.loader import load_config

    config = load_config(args.config)
    timesteps = args.total_timesteps or config.total_env_steps
    metrics = run_sb3_check(
        env_id=config.env_id,
        total_timesteps=timesteps,
        seed=config.seed,
        num_envs=config.num_envs,
        output_dir=config.results_dir,
    )
    print(metrics)


if __name__ == "__main__":
    main()
