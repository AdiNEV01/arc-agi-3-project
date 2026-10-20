"""Train a readable PPO baseline or one of the controlled PPO experiments.

Examples (run these from the PPO project folder):
    python ppo_baseline.py --check-only --device cuda
    python ppo_baseline.py --version baseline --device cuda --timesteps 2000000 --seed 0 --output-dir outputs/ppo_2m_baseline_s0
    python ppo_baseline.py --version grid_cnn --device cuda --timesteps 2000000 --seed 0 --output-dir outputs/ppo_2m_gridcnn_s0
    python ppo_baseline.py --version novelty --device cuda --timesteps 2000000 --seed 0 --output-dir outputs/ppo_2m_novelty_s0

Versions:
  baseline          Stable-Baselines3 CNN + sparse level-completion reward.
  grid_cnn          Same reward, a CNN designed to preserve small grid details.
  novelty           Default CNN + observation-count exploration bonus.
  grid_cnn_novelty  Combines both improvements for an ablation.

Training reward is not the official ARC-AGI-3 score. Evaluation always reports
level completion separately from any intrinsic training reward.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from arc_gym import ArcEnv
from policy_networks import GridCNNFeaturesExtractor
from reward_wrappers import CountNoveltyReward
from training_callbacks import ExperimentLoggingCallback


VERSIONS = ("baseline", "grid_cnn", "novelty", "grid_cnn_novelty")


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def system_information(device: str) -> dict:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "device_requested": device,
        "packages": {
            "stable-baselines3": package_version("stable-baselines3"),
            "gymnasium": package_version("gymnasium"),
            "numpy": package_version("numpy"),
            "arc-agi": package_version("arc-agi"),
            "arcengine": package_version("arcengine"),
        },
    }


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see a CUDA GPU. "
            "Run the diagnostic command or use --device cpu."
        )
    return requested


def make_environment(args: argparse.Namespace, run_dir: Path | None = None):
    """Construct the ARC environment and selected reward version."""
    env = ArcEnv(
        game_id=args.game_id,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    if args.version in ("novelty", "grid_cnn_novelty"):
        env = CountNoveltyReward(env, bonus_scale=args.novelty_scale)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(run_dir / "monitor.csv"))
    else:
        env = Monitor(env)
    return env


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PPO on ARC-AGI-3 ls20 with reproducible logs."
    )
    parser.add_argument(
        "--version", choices=VERSIONS, default="baseline",
        help="Which PPO experiment to train.",
    )
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--novelty-scale", type=float, default=0.005)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--progress-freq", type=int, default=10_000)
    parser.add_argument("--output-dir", default="outputs/ppo_2m_baseline_s0")
    parser.add_argument("--resume", default=None,
                        help="Optional PPO .zip checkpoint to continue from."
                        " --timesteps then means additional interactions.")
    parser.add_argument("--check-only", action="store_true",
                        help="Create the environment and model, but do not train.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    if args.n_steps < 1 or args.batch_size < 1 or args.n_epochs < 1:
        parser.error("n_steps, batch_size, and n_epochs must be positive")
    if args.checkpoint_freq < 0 or args.progress_freq < 0:
        parser.error("checkpoint/progress frequencies cannot be negative")

    device = choose_device(args.device)
    set_random_seeds(args.seed)
    run_dir = Path(args.output_dir).resolve()

    if not args.check_only:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already contains files: {run_dir}\n"
                "Choose a new --output-dir so earlier runs are not overwritten."
            )
        run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("ARC-AGI-3 PPO EXPERIMENT")
    print("=" * 64)
    print(f"Experiment version: {args.version}")
    print(f"Game: {args.game_id}; seed: {args.seed}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Selected device: {device}")
    print(f"Output directory: {run_dir}")

    env = None
    model = None
    status = "not_started"
    start_time = time.perf_counter()

    try:
        print("\nCreating ARC environment...")
        env = make_environment(args, None if args.check_only else run_dir)
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        policy_kwargs = {}
        if args.version in ("grid_cnn", "grid_cnn_novelty"):
            policy_kwargs = {
                "features_extractor_class": GridCNNFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": 256},
            }

        if args.resume:
            print(f"Loading checkpoint: {args.resume}")
            model = PPO.load(args.resume, env=env, device=device)
        else:
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
                policy_kwargs=policy_kwargs,
                seed=args.seed,
                device=device,
                verbose=0,
            )

        if args.check_only:
            print("\nPPO model constructed successfully.")
            print(f"Policy: {model.policy.__class__.__name__}")
            print(f"Device selected: {model.device}")
            print("CHECK PASSED. No training was performed.")
            return

        # CSV output includes rollout losses and speed, without filling the
        # terminal with thousands of training lines during a 2M-step run.
        sb3_logger = configure(str(run_dir / "sb3_logs"), ["csv"])
        model.set_logger(sb3_logger)

        config = {
            "version": args.version,
            "game_id": args.game_id,
            "seed": args.seed,
            "requested_timesteps": args.timesteps,
            "max_steps_per_episode": args.max_steps,
            "device": device,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "entropy_coefficient": args.entropy_coefficient,
            "novelty_scale": args.novelty_scale,
            "resume_checkpoint": args.resume,
            "policy": model.policy.__class__.__name__,
            "reward_definition": (
                "level-completion reward only"
                if args.version in ("baseline", "grid_cnn")
                else "level-completion reward plus count-based observation novelty"
            ),
            "system": system_information(device),
        }
        save_json(run_dir / "run_config.json", config)

        callback = ExperimentLoggingCallback(
            run_dir=run_dir,
            checkpoint_freq=args.checkpoint_freq,
            progress_freq=args.progress_freq,
            verbose=1,
        )

        print(f"\nStarting training for {args.timesteps:,} interactions.")
        print("A checkpoint will be saved every "
              f"{args.checkpoint_freq:,} interactions (0 disables checkpoints).")
        status = "running"
        starting_timesteps = int(model.num_timesteps)
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback,
                reset_num_timesteps=(args.resume is None),
                progress_bar=False,
            )
            status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
            print("\nTraining interrupted by the user; saving the current model.")

        elapsed_seconds = time.perf_counter() - start_time
        interactions_this_run = int(model.num_timesteps) - starting_timesteps
        final_path = run_dir / "ppo_final.zip"
        last_path = run_dir / "ppo_last.zip"
        model.save(last_path)
        if status == "completed":
            model.save(final_path)
            print(f"\nFinal model saved: {final_path}")
        else:
            print(f"\nRecoverable model saved: {last_path}")

        summary = {
            "status": status,
            "version": args.version,
            "game_id": args.game_id,
            "seed": args.seed,
            "requested_timesteps": args.timesteps,
            "actual_interactions_this_run": interactions_this_run,
            "actual_total_timesteps_in_model": int(model.num_timesteps),
            "elapsed_seconds": elapsed_seconds,
            "average_steps_per_second": interactions_this_run / max(elapsed_seconds, 1e-9),
            "model_path": str(final_path if status == "completed" else last_path),
            "run_config_path": str(run_dir / "run_config.json"),
            "monitor_path": str(run_dir / "monitor.csv"),
            "sb3_progress_path": str(run_dir / "sb3_logs" / "progress.csv"),
            "training_reward_is_official_arc_score": False,
        }
        save_json(run_dir / "summary.json", summary)
        print(f"Summary saved: {run_dir / 'summary.json'}")
        print(f"Training logs saved under: {run_dir}")

    except Exception:
        status = "failed"
        if run_dir.exists():
            if model is not None:
                try:
                    model.save(run_dir / "ppo_recoverable_after_error.zip")
                except Exception:
                    pass
            save_json(
                run_dir / "failure_summary.json",
                {
                    "status": status,
                    "game_id": args.game_id,
                    "seed": args.seed,
                    "version": args.version,
                    "error": "See the terminal traceback for the full exception.",
                },
            )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
