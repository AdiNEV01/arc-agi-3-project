"""Train PPO with a small reward bonus for exploring new observations.

Quick test:
    python novelty_ppo.py --check-only --device cuda
    python novelty_ppo.py --device cuda --timesteps 10000 --seed 0

A longer run (use a new output directory for each experiment):
    python novelty_ppo.py --device cuda --timesteps 2000000 --seed 0 --output-dir outputs/novelty_s0

The novelty reward is only a training aid. It is not the official ARC score.
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
from reward_wrappers import CountNoveltyReward
from training_callbacks import ExperimentLoggingCallback


def choose_device(requested_device):
    """Use the requested device, or choose CUDA when it is available."""
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
    """Use the same seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train PPO with observation-count exploration reward."
    )
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    # These PPO settings match the baseline so the comparison is fair.
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)

    # The extra reward is deliberately small. Compare other values later.
    parser.add_argument("--novelty-scale", type=float, default=0.001)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--progress-freq", type=int, default=10_000)
    parser.add_argument("--output-dir", default="outputs/novelty_ppo_seed_0")
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
    if args.novelty_scale < 0:
        parser.error("--novelty-scale cannot be negative")

    device = choose_device(args.device)
    set_seeds(args.seed)
    run_dir = Path(args.output_dir).resolve()

    # Do not accidentally overwrite another experiment's files.
    if not args.check_only and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {run_dir}\n"
            "Choose another --output-dir to keep runs separate."
        )

    print("PPO experiment: NOVELTY REWARD")
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
        # First create the normal ARC wrapper.
        env = ArcEnv(
            game_id=args.game_id,
            max_steps=args.max_steps,
            seed=args.seed,
        )

        # Then add the count-based exploration reward.
        env = CountNoveltyReward(env, bonus_scale=args.novelty_scale)

        # Monitor records the reward actually used for PPO training.
        if args.check_only:
            env = Monitor(env)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=str(run_dir / "monitor.csv"))

        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)

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
            seed=args.seed,
            device=device,
            verbose=0,
        )

        if args.check_only:
            observation, _ = env.reset()
            print("Sample observation shape:", observation.shape)
            print("Sample observation type:", observation.dtype)
            if not env.observation_space.contains(observation):
                raise ValueError("The observation does not match the observation space.")
            print("\nCHECK PASSED. The PPO model was created; no training was done.")
            print("Model device:", model.device)
            return

        # Save PPO's training statistics in a CSV file.
        logger = configure(str(run_dir / "sb3_logs"), ["csv"])
        model.set_logger(logger)

        config = {
            "experiment": "novelty_ppo",
            "game_id": args.game_id,
            "seed": args.seed,
            "requested_timesteps": args.timesteps,
            "max_steps": args.max_steps,
            "device": device,
            "novelty_scale": args.novelty_scale,
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
            "experiment": "novelty_ppo",
            "requested_timesteps": args.timesteps,
            "actual_timesteps": int(model.num_timesteps),
            "elapsed_seconds": elapsed_seconds,
            "steps_per_second": model.num_timesteps / max(elapsed_seconds, 1e-9),
            "model_path": str(run_dir / "ppo_final.zip"),
            "official_arc_score": "Not measured by training reward; use evaluation script.",
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
