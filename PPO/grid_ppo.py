"""Train PPO with a grid-aware observation and CNN.

Quick test:
    python grid_ppo.py --check-only --device cuda
    python grid_ppo.py --device cuda --timesteps 10000 --seed 0

A longer run:
    python grid_ppo.py --device cuda --timesteps 2000000 --seed 0 --output-dir outputs/grid_ppo_s0

The policy receives a separate channel for each visible ARC colour. It uses
only the rendered board, not hidden game state or a hand-written solution.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from arc_gym import ArcEnv
from policy_networks import GridCNNFeaturesExtractor
from grid_observation import OneHotGridObservation
from training_callbacks import ExperimentLoggingCallback



def choose_device(requested_device):
    if requested_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a CUDA GPU. "
            "Try --device cpu or check your PyTorch installation."
        )

    return requested_device


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train PPO using one-hot ARC colours and a grid CNN."
    )
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    # Keep these values the same as the PPO baseline.
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)

    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--progress-freq", type=int, default=10_000)
    parser.add_argument("--output-dir", default="outputs/grid_ppo_seed_0")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build the environment and model, but do not train.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    if args.n_steps < 1 or args.batch_size < 1 or args.n_epochs < 1:
        parser.error("n-steps, batch-size, and n-epochs must be positive")

    device = choose_device(args.device)
    set_seeds(args.seed)
    run_dir = Path(args.output_dir).resolve()

    if not args.check_only and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {run_dir}\n"
            "Choose another --output-dir to keep runs separate."
        )

    print("PPO experiment: GRID-AWARE CNN")
    print("Game:", args.game_id)
    print("Seed:", args.seed)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Output folder:", run_dir)

    env = None
    model = None
    start_time = time.perf_counter()

    try:
        # Create the normal ARC wrapper, then encode colours as separate planes.
        env = ArcEnv(
            game_id=args.game_id,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        env = OneHotGridObservation(env)

        if args.check_only:
            env = Monitor(env)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=str(run_dir / "monitor.csv"))

        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)

        # This CNN uses small filters and does not downsample the grid too early.
        policy_settings = {
            "features_extractor_class": GridCNNFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        }

        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.entropy_coefficient,
            policy_kwargs=policy_settings,
            seed=args.seed,
            device=device,
            verbose=0,
        )

        if args.check_only:
            # Also check that the real game produces the expected encoded shape.
            observation, _ = env.reset()
            print("Sample observation shape:", observation.shape)
            if observation.shape != (16, 64, 64):
                raise ValueError("Grid observation does not have shape (16, 64, 64).")
            if not env.observation_space.contains(observation):
                raise ValueError("The observation does not match the grid observation space.")
            print("Observation type:", observation.dtype)
            print("\nCHECK PASSED. The model was created; no training was done.")
            print("Model device:", model.device)
            return

        logger = configure(str(run_dir / "sb3_logs"), ["csv"])
        model.set_logger(logger)

        config = {
            "experiment": "grid_ppo",
            "game_id": args.game_id,
            "seed": args.seed,
            "requested_timesteps": args.timesteps,
            "max_steps": args.max_steps,
            "device": device,
            "observation_encoding": "16-channel one-hot colour grid, values 0 or 1",
            "feature_extractor": "GridCNNFeaturesExtractor",
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "entropy_coefficient": args.entropy_coefficient,
            "training_reward_is_official_arc_score": False,
        }
        with (run_dir / "run_config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        callback = ExperimentLoggingCallback(
            run_dir=run_dir,
            checkpoint_freq=args.checkpoint_freq,
            progress_freq=args.progress_freq,
            verbose=1,
        )

        print(f"\nTraining for {args.timesteps:,} interactions...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=False,
        )

        model.save(run_dir / "ppo_final.zip")
        elapsed_seconds = time.perf_counter() - start_time
        summary = {
            "status": "completed",
            "experiment": "grid_ppo",
            "requested_timesteps": args.timesteps,
            "actual_timesteps": int(model.num_timesteps),
            "elapsed_seconds": elapsed_seconds,
            "steps_per_second": model.num_timesteps / max(elapsed_seconds, 1e-9),
            "model_path": str(run_dir / "ppo_final.zip"),
            "official_arc_score": "Use the separate evaluation script.",
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

        print("\nTraining finished.")
        print("Model saved to:", run_dir / "ppo_final.zip")
        print("Logs saved in:", run_dir)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving the current model if possible.")
        if model is not None and not args.check_only:
            run_dir.mkdir(parents=True, exist_ok=True)
            model.save(run_dir / "ppo_interrupted.zip")
            print("Recoverable model saved to:", run_dir / "ppo_interrupted.zip")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
