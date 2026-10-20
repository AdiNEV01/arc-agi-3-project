"""Create training-learning and optimisation plots from one training run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def numeric_column(frame: pd.DataFrame, name: str):
    if name not in frame.columns:
        return None
    values = pd.to_numeric(frame[name], errors="coerce")
    if values.notna().sum() == 0:
        return None
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True,
                        help="Training folder containing sb3_logs/progress.csv.")
    parser.add_argument("--output", default=None,
                        help="Optional PNG path. Defaults to <run-dir>/plots/training_dashboard.png")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    progress_path = run_dir / "sb3_logs" / "progress.csv"
    episodes_path = run_dir / "episodes.csv"

    if not progress_path.exists() and not episodes_path.exists():
        raise FileNotFoundError(
            f"No training CSV found under {run_dir.resolve()}. "
            "Check the run folder or run the training script first."
        )

    progress = pd.read_csv(progress_path) if progress_path.exists() else pd.DataFrame()
    episodes = pd.read_csv(episodes_path) if episodes_path.exists() else pd.DataFrame()

    output_path = Path(args.output) if args.output else run_dir / "plots" / "training_dashboard.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    axes = axes.ravel()

    steps = numeric_column(progress, "time/total_timesteps")
    mean_reward = numeric_column(progress, "rollout/ep_rew_mean")
    if steps is not None and mean_reward is not None:
        axes[0].plot(steps, mean_reward, label="SB3 rollout mean reward")
        axes[0].set_xlabel("Environment interactions")
        axes[0].set_ylabel("Training episode return")
        axes[0].set_title("Training reward")
    elif not episodes.empty and "episode_reward_training" in episodes:
        episode_reward = pd.to_numeric(episodes["episode_reward_training"], errors="coerce")
        axes[0].plot(range(1, len(episode_reward) + 1), episode_reward)
        axes[0].set_xlabel("Completed episode")
        axes[0].set_ylabel("Training episode return")
        axes[0].set_title("Episode rewards")
    else:
        axes[0].text(0.5, 0.5, "No reward data", ha="center", va="center")
        axes[0].set_title("Training reward")

    mean_length = numeric_column(progress, "rollout/ep_len_mean")
    if steps is not None and mean_length is not None:
        axes[1].plot(steps, mean_length)
        axes[1].set_xlabel("Environment interactions")
        axes[1].set_ylabel("Actions per episode")
        axes[1].set_title("Mean episode length")
    elif not episodes.empty and "episode_length" in episodes:
        lengths = pd.to_numeric(episodes["episode_length"], errors="coerce")
        axes[1].plot(range(1, len(lengths) + 1), lengths)
        axes[1].set_xlabel("Completed episode")
        axes[1].set_ylabel("Actions per episode")
        axes[1].set_title("Episode length")
    else:
        axes[1].text(0.5, 0.5, "No episode length data", ha="center", va="center")
        axes[1].set_title("Mean episode length")

    if not episodes.empty and "levels_completed" in episodes.columns:
        levels = pd.to_numeric(episodes["levels_completed"], errors="coerce").fillna(0)
        episode_numbers = range(1, len(levels) + 1)
        axes[2].plot(episode_numbers, levels, marker=".", linestyle="none", label="Levels per episode")
        window = min(25, max(1, len(levels)))
        axes[2].plot(
            episode_numbers,
            levels.rolling(window=window, min_periods=1).mean(),
            label=f"Rolling mean ({window} episodes)",
        )
        axes[2].set_xlabel("Completed episode")
        axes[2].set_ylabel("Levels completed")
        axes[2].set_title("Observed task progress")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No episode level data", ha="center", va="center")
        axes[2].set_title("Observed task progress")

    loss_columns = [
        ("train/policy_gradient_loss", "Policy loss"),
        ("train/value_loss", "Value loss"),
    ]
    loss_found = False
    if steps is not None:
        for name, label in loss_columns:
            values = numeric_column(progress, name)
            if values is not None:
                axes[3].plot(steps, values, label=label)
                loss_found = True
    if loss_found:
        axes[3].set_xlabel("Environment interactions")
        axes[3].set_ylabel("Loss")
        axes[3].set_title("PPO optimisation losses")
        axes[3].legend()
    else:
        axes[3].text(0.5, 0.5, "Loss metrics not found in progress.csv", ha="center", va="center")
        axes[3].set_title("PPO optimisation losses")

    fps = numeric_column(progress, "time/fps")
    if steps is not None and fps is not None:
        axes[4].plot(steps, fps)
        axes[4].set_xlabel("Environment interactions")
        axes[4].set_ylabel("Steps per second")
        axes[4].set_title("Training throughput")
    else:
        axes[4].text(0.5, 0.5, "No throughput data", ha="center", va="center")
        axes[4].set_title("Training throughput")

    approx_kl = numeric_column(progress, "train/approx_kl")
    explained_variance = numeric_column(progress, "train/explained_variance")
    found = False
    if steps is not None and approx_kl is not None:
        axes[5].plot(steps, approx_kl, label="Approximate KL")
        found = True
    if steps is not None and explained_variance is not None:
        axes[5].plot(steps, explained_variance, label="Explained variance")
        found = True
    if found:
        axes[5].set_xlabel("Environment interactions")
        axes[5].set_title("Policy/value diagnostics")
        axes[5].legend()
    else:
        axes[5].text(0.5, 0.5, "No KL/value diagnostic data", ha="center", va="center")
        axes[5].set_title("Policy/value diagnostics")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"PPO training dashboard: {run_dir.name}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Training dashboard saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
