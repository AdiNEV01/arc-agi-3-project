"""Small Stable-Baselines3 callbacks for checkpoints and readable CSV logs."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class ExperimentLoggingCallback(BaseCallback):
    """Save checkpoints and one row per episode/rollout."""

    def __init__(
        self,
        run_dir: str | Path,
        checkpoint_freq: int = 50_000,
        progress_freq: int = 10_000,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_freq = int(checkpoint_freq)
        self.progress_freq = int(progress_freq)
        self.next_checkpoint = self.checkpoint_freq
        self.next_progress = self.progress_freq
        self.start_time = time.perf_counter()
        self.episode_number = 0

        self.episodes_path = self.run_dir / "episodes.csv"
        self.rollouts_path = self.run_dir / "rollout_metrics.csv"

        self.episode_fields = [
            "global_timestep",
            "episode_number",
            "episode_reward_training",
            "episode_reward_extrinsic",
            "episode_reward_intrinsic",
            "episode_length",
            "levels_completed",
            "game_state",
            "game_id",
            "game_seed",
        ]
        self.rollout_fields = [
            "global_timestep",
            "elapsed_seconds",
            "approx_fps",
            "rollout_mean_reward",
            "rollout_mean_episode_length",
        ]

    @staticmethod
    def _append_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # When resuming, schedule the next future threshold instead of saving
        # several duplicate checkpoints for timesteps already completed.
        if self.checkpoint_freq > 0:
            self.next_checkpoint = (
                self.num_timesteps // self.checkpoint_freq + 1
            ) * self.checkpoint_freq
        if self.progress_freq > 0:
            self.next_progress = (
                self.num_timesteps // self.progress_freq + 1
            ) * self.progress_freq

    def _on_step(self) -> bool:
        # SB3 includes an info dict for each environment in this list.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if not done:
                continue

            episode_info = info.get("episode", {})
            self.episode_number += 1
            self._append_row(
                self.episodes_path,
                self.episode_fields,
                {
                    "global_timestep": self.num_timesteps,
                    "episode_number": self.episode_number,
                    "episode_reward_training": episode_info.get("r", ""),
                    "episode_reward_extrinsic": info.get(
                        "episode_extrinsic_return", info.get("extrinsic_reward", "")
                    ),
                    "episode_reward_intrinsic": info.get(
                        "episode_intrinsic_return", 0.0
                    ),
                    "episode_length": episode_info.get(
                        "l", info.get("episode_steps", "")
                    ),
                    "levels_completed": info.get("levels_completed", 0),
                    "game_state": info.get("game_state", "UNKNOWN"),
                    "game_id": info.get("game_id", ""),
                    "game_seed": info.get("game_seed", ""),
                },
            )

        while self.checkpoint_freq > 0 and self.num_timesteps >= self.next_checkpoint:
            checkpoint_path = (
                self.checkpoint_dir
                / f"ppo_{self.next_checkpoint:09d}_steps.zip"
            )
            self.model.save(checkpoint_path)
            if self.verbose:
                print(f"Checkpoint saved: {checkpoint_path}")
            self.next_checkpoint += self.checkpoint_freq

        while self.progress_freq > 0 and self.num_timesteps >= self.next_progress:
            elapsed = max(time.perf_counter() - self.start_time, 1e-9)
            fps = self.num_timesteps / elapsed
            if self.verbose:
                print(
                    f"Progress: {self.num_timesteps:,} steps | "
                    f"{fps:.1f} steps/s | {elapsed / 60:.1f} min elapsed"
                )
            self.next_progress += self.progress_freq

        return True

    def _on_rollout_end(self) -> None:
        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        logger_values = self.model.logger.name_to_value
        self._append_row(
            self.rollouts_path,
            self.rollout_fields,
            {
                "global_timestep": self.num_timesteps,
                "elapsed_seconds": elapsed,
                "approx_fps": self.num_timesteps / elapsed,
                "rollout_mean_reward": logger_values.get("rollout/ep_rew_mean", ""),
                "rollout_mean_episode_length": logger_values.get("rollout/ep_len_mean", ""),
            },
        )
