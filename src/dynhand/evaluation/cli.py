"""Evaluate a trained checkpoint and print metrics as JSON."""

import argparse
import json

import numpy as np
import torch

from dynhand.config.loader import load_config
from dynhand.evaluation.evaluate import build_single_env, evaluate
from dynhand.rl.sac import SAC


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DynHand checkpoint")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    episodes = args.episodes if args.episodes is not None else config.eval.episodes

    env = build_single_env(config.env_id)
    single_obs = env.observation_space
    single_act = env.action_space
    obs_dim = int(np.prod(single_obs.shape))
    act_dim = int(np.prod(single_act.shape))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sac = SAC(
        obs_dim,
        act_dim,
        single_act.low.astype(np.float32),
        single_act.high.astype(np.float32),
        config.sac,
        device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sac.load_state_dict(checkpoint["sac"])
    env.close()

    metrics = evaluate(sac, config.env_id, episodes, config.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
