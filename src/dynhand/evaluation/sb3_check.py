"""Stable-Baselines3 SAC cross-check.

This independent implementation validates the environment and matched SAC
hyperparameters. It is a debugging tool, not the primary training path.
"""

import argparse


def run_sb3_check(
    env_id: str,
    total_timesteps: int,
    seed: int,
    num_envs: int,
    output_dir: str,
    eval_interval_steps: int = 10_000,
    learning_starts: int = 5_000,
    batch_size: int = 256,
    buffer_size: int = 1_000_000,
    gradient_steps: int = 1,
    run_name: str | None = None,
) -> dict[str, float]:
    """Train SB3 SAC in one process and record periodic evaluations."""
    del num_envs
    from stable_baselines3 import SAC
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import make_vec_env

    from dynhand.config.schema import ExperimentConfig
    from dynhand.envs.record import RunRecorder

    config = ExperimentConfig(
        experiment_id=run_name or "sb3_check",
        env_id=env_id,
        seed=seed,
        total_env_steps=total_timesteps,
        start_steps=learning_starts,
    )
    recorder = RunRecorder(config, output_dir)
    env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        wrapper_class=Monitor,
    )
    eval_env = None
    try:
        model = SAC(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts,
        )
        eval_env = Monitor(__import__("gymnasium").make(env_id))
        trained = 0
        last_metrics: dict[str, float] = {}
        while trained < total_timesteps:
            chunk = min(eval_interval_steps, total_timesteps - trained)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            trained += chunk
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=config.eval.episodes,
                deterministic=True,
            )
            last_metrics = {
                "eval_return_mean": float(mean_reward),
                "eval_return_std": float(std_reward),
            }
            recorder.log_metrics(trained, last_metrics)
        model.save(str(recorder.run_dir / "sb3_sac"))
        return last_metrics
    finally:
        if eval_env is not None:
            eval_env.close()
        env.close()
        recorder.close()


def main() -> None:
    """Run the matched SB3 cross-check from a DynHand config."""
    parser = argparse.ArgumentParser(description="SB3 SAC cross-check")
    parser.add_argument("--config", required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
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
        eval_interval_steps=config.eval.interval_steps,
        learning_starts=config.start_steps,
        batch_size=config.sac.batch_size,
        buffer_size=config.sac.buffer_size,
        gradient_steps=config.sac.utd_ratio,
        run_name=args.run_name,
    )
    print(metrics)


if __name__ == "__main__":
    main()
