"""Compare training curves across baseline, novelty, and grid PPO seeds.

This script reads the CSV files created by the training scripts. It does not
run new training. Training returns are shown separately by version because the
novelty experiment includes an intrinsic reward bonus.

Example:
    python plot_training_comparison.py --input-root outputs --output outputs/analysis_final/training_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


VARIANTS = ("baseline", "novelty", "grid")
SEEDS = (0, 1, 2)


def read_run_files(input_root: Path):
    """Read each run's rollout and episode CSV files when available."""
    rollout_frames = {variant: [] for variant in VARIANTS}
    episode_frames = {variant: [] for variant in VARIANTS}

    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = input_root / f"{variant}_s{seed}"

            rollout_path = run_dir / "rollout_metrics.csv"
            if rollout_path.exists():
                try:
                    frame = pd.read_csv(rollout_path)
                    frame["training_seed"] = seed
                    rollout_frames[variant].append(frame)
                except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
                    print(f"Skipping {rollout_path}: {error}")

            episode_path = run_dir / "episodes.csv"
            if episode_path.exists():
                try:
                    frame = pd.read_csv(episode_path)
                    frame["training_seed"] = seed
                    episode_frames[variant].append(frame)
                except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
                    print(f"Skipping {episode_path}: {error}")

    return rollout_frames, episode_frames


def draw_mean_and_spread(ax, frame, x_column, y_column, x_label, y_label, title):
    """Draw the mean curve and a shaded one-standard-deviation band."""
    if frame.empty or x_column not in frame.columns or y_column not in frame.columns:
        ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    frame = frame.copy()
    frame[x_column] = pd.to_numeric(frame[x_column], errors="coerce")
    frame[y_column] = pd.to_numeric(frame[y_column], errors="coerce")
    frame = frame.dropna(subset=[x_column, y_column])

    if frame.empty:
        ax.text(0.5, 0.5, "No numeric data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    grouped = frame.groupby(x_column)[y_column].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    x_values = grouped[x_column].to_numpy(dtype=float)
    means = grouped["mean"].to_numpy(dtype=float)
    spread = grouped["std"].to_numpy(dtype=float)

    ax.plot(x_values, means, label="Mean across available seeds")
    ax.fill_between(x_values, means - spread, means + spread, alpha=0.2,
                    label="±1 standard deviation")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves across PPO experiment seeds.")
    parser.add_argument("--input-root", default="outputs",
                        help="Folder containing baseline_s0, novelty_s0, grid_s0, etc.")
    parser.add_argument("--output", default="outputs/analysis_final/training_comparison.png",
                        help="Where to save the combined PNG figure.")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    rollout_frames, episode_frames = read_run_files(input_root)

    has_any_data = any(rollout_frames[name] or episode_frames[name] for name in VARIANTS)
    if not has_any_data:
        raise FileNotFoundError(
            f"No training CSV files found under {input_root.resolve()}. "
            "Run the training experiments first."
        )

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))

    for column, variant in enumerate(VARIANTS):
        rollouts = (
            pd.concat(rollout_frames[variant], ignore_index=True)
            if rollout_frames[variant] else pd.DataFrame()
        )
        episodes = (
            pd.concat(episode_frames[variant], ignore_index=True)
            if episode_frames[variant] else pd.DataFrame()
        )

        # Top row: the reward PPO actually optimises during training.
        # For novelty, this contains extrinsic plus intrinsic reward.
        draw_mean_and_spread(
            axes[0, column], rollouts,
            "global_timestep", "rollout_mean_reward",
            "Environment interactions", "Mean training episode return",
            f"{variant}: PPO training reward",
        )

        # Bottom row: extrinsic reward only, so this remains comparable across
        # baseline, novelty, and grid versions.
        draw_mean_and_spread(
            axes[1, column], episodes,
            "episode_number", "episode_reward_extrinsic",
            "Completed training episode", "Extrinsic episode return",
            f"{variant}: level-completion reward only",
        )

    figure.suptitle(
        "PPO training comparison across seeds\n"
        "Top: optimisation reward (novelty includes its exploration bonus). "
        "Bottom: extrinsic reward only.",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Training comparison plot saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
