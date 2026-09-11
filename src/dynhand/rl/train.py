"""Training entrypoint for all ablation conditions.

One trainer serves conditions A through E; the active condition is set
entirely by the config file. Condition C/D/E model-based rollout
augmentation is not implemented yet and raises NotImplementedError.
"""

import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dynhand.config.loader import load_config
from dynhand.config.schema import ExperimentConfig
from dynhand.envs.record import RunRecorder
from dynhand.envs.vec import make_vec_env
from dynhand.evaluation.evaluate import evaluate
from dynhand.rl.bc import BCTrainer
from dynhand.rl.demo import load_minari_transitions
from dynhand.rl.replay import ReplayBuffer
from dynhand.rl.sac import SAC
from dynhand.rl.schedules import demo_ratio, sample_mixed
from dynhand.utils.seed import seed_everything, set_torch_threads, worker_seed


def _demo_transitions(config: ExperimentConfig, seed: int) -> dict[str, np.ndarray]:
    if config.demo.minari_dataset is None:
        raise ValueError("demo.enabled requires demo.minari_dataset to be set")
    rng = np.random.default_rng(seed)
    demos = load_minari_transitions(config.demo.minari_dataset)
    return _subsample(demos, config.sac.buffer_size // 2, rng)


def _subsample(
    demos: dict[str, np.ndarray], max_transitions: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Cap demo transitions at half the replay capacity, keeping the first episodes."""
    n = len(demos["obs"])
    if n <= max_transitions:
        return demos
    idx = np.sort(rng.choice(n, size=max_transitions, replace=False))
    return {key: value[idx] for key, value in demos.items()}


def _load_resume_checkpoint(path, sac, buffer, rng) -> int:
    """Restore trainer state from a checkpoint, return its global step."""
    checkpoint = torch.load(path, map_location=sac.device, weights_only=False)
    sac.load_state_dict(checkpoint["sac"])
    if checkpoint.get("buffer") is not None:
        buffer.load_state(checkpoint["buffer"])
    return int(checkpoint["global_step"])


def train(
    config: ExperimentConfig,
    resume: bool = False,
    run_dir_override: str | None = None,
    run_name: str | None = None,
) -> dict[str, float]:
    """Run one training job from a validated config, return final metrics."""
    if config.dynamics_aug.enabled:
        raise NotImplementedError(
            "Dynamics-model augmentation (conditions C, D, E) is a later phase"
        )
    if run_name is not None:
        config.experiment_id = run_name
    seed_everything(config.seed)
    set_torch_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if run_dir_override is not None:
        config.results_dir = run_dir_override
    recorder = RunRecorder(config, config.results_dir)
    writer = SummaryWriter(log_dir=str(recorder.run_dir / "tb"))

    envs = make_vec_env(config.env_id, config.num_envs, seed=config.seed)
    single_obs = envs.single_observation_space
    single_act = envs.single_action_space
    obs_dim = int(np.prod(single_obs.shape))
    act_dim = int(np.prod(single_act.shape))

    rng = np.random.default_rng(config.seed)
    buffer = ReplayBuffer(obs_dim, act_dim, config.sac.buffer_size, rng)
    sac = SAC(
        obs_dim,
        act_dim,
        single_act.low.astype(np.float32),
        single_act.high.astype(np.float32),
        config.sac,
        device,
    )

    start_step = 0
    if resume:
        latest = recorder.latest_checkpoint()
        if latest is not None:
            start_step = _load_resume_checkpoint(latest, sac, buffer, rng)
            print(f"Resumed from {latest} at step {start_step}")

    demo_buffer = None
    if config.demo.enabled:
        demos = _demo_transitions(config, config.seed)
        demo_buffer = ReplayBuffer(obs_dim, act_dim, len(demos["obs"]), rng)
        demo_buffer.add_batch(
            demos["obs"],
            demos["acts"],
            demos["next_obs"],
            demos["rewards"],
            demos["dones"],
        )
        if start_step == 0:
            bc = BCTrainer(
                sac,
                demos["obs"],
                demos["acts"],
                lr=config.demo.bc_lr,
                holdout_fraction=config.demo.bc_holdout_fraction,
                rng=rng,
            )
            bc_metrics = bc.train_epochs(
                epochs=config.demo.bc_epochs,
                batch_size=config.demo.bc_batch_size,
                patience=config.demo.bc_patience,
            )
            recorder.log_metrics(0, bc_metrics)
            for key, value in bc_metrics.items():
                writer.add_scalar(f"train/{key}", value, 0)

    obs, _ = envs.reset(seed=worker_seed(config.seed, 0))
    global_step = start_step
    next_eval = (
        (start_step // config.eval.interval_steps) + 1
    ) * config.eval.interval_steps
    metrics: dict[str, float] = {}
    last_eval_step: int | None = None

    while global_step < config.total_env_steps:
        if global_step < config.start_steps:
            actions = np.stack(
                [envs.single_action_space.sample() for _ in range(config.num_envs)]
            )
        else:
            actions = sac.act(obs)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations
        real_next_obs = next_obs.copy()
        if "final_obs" in infos:
            for i, final in enumerate(infos["final_obs"]):
                if dones[i]:
                    real_next_obs[i] = final
        for i in range(config.num_envs):
            buffer.add(
                obs[i], actions[i], real_next_obs[i], float(rewards[i]), float(dones[i])
            )
        obs = next_obs
        global_step += config.num_envs

        if len(buffer) >= config.sac.batch_size:
            for _ in range(config.sac.utd_ratio):
                if demo_buffer is not None:
                    ratio = demo_ratio(
                        global_step,
                        config.demo.demo_ratio_start,
                        config.demo.demo_ratio_anneal_steps,
                    )
                    batch = sample_mixed(
                        buffer, demo_buffer, config.sac.batch_size, ratio, rng
                    )
                else:
                    batch = buffer.sample(config.sac.batch_size)
                metrics = sac.update(batch)
            if global_step % 1000 < config.num_envs:
                recorder.log_metrics(global_step, metrics)
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, global_step)

        if global_step >= next_eval:
            eval_metrics = evaluate(
                sac, config.env_id, config.eval.episodes, config.seed
            )
            recorder.log_metrics(global_step, eval_metrics)
            writer.add_scalar(
                "eval/return_mean", eval_metrics["eval_return_mean"], global_step
            )
            last_eval_step = global_step
            next_eval += config.eval.interval_steps

        if global_step % config.checkpoint_interval < config.num_envs:
            recorder.save_checkpoint(
                global_step,
                {
                    "sac": sac.state_dict(),
                    "buffer": buffer.state_dict(),
                    "global_step": global_step,
                },
            )

    envs.close()
    final = recorder.save_checkpoint(
        config.total_env_steps,
        {
            "sac": sac.state_dict(),
            "buffer": buffer.state_dict(),
            "global_step": global_step,
        },
    )
    final_metrics = evaluate(sac, config.env_id, config.eval.episodes, config.seed)
    if last_eval_step != global_step:
        recorder.log_metrics(global_step, final_metrics)
    writer.close()
    recorder.close()
    print(f"Training complete. Final checkpoint: {final}")
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DynHand SAC")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the latest checkpoint"
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override the results directory",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Unique output directory name for this seed or replicate",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    train(
        config,
        resume=args.resume,
        run_dir_override=args.results_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
