"""Run a short random-action diagnostic and save the observed frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from arc_gym import ACTION_NAMES, ArcEnv


def save_frame(observation: np.ndarray, path: Path) -> None:
    image = Image.fromarray(observation[0], mode="L")
    image = image.resize((512, 512), resample=Image.Resampling.NEAREST)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="outputs/diagnostics/seed_0")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ArcEnv(args.game_id, max_steps=args.max_steps, seed=args.seed)
    records = []

    try:
        observation, info = env.reset(seed=args.seed)
        save_frame(observation, output_dir / "step_000.png")
        print("Initial observation:", observation.shape, observation.dtype)
        print("Initial levels completed:", info["levels_completed"])

        for step_number in range(1, args.steps + 1):
            action = int(env.action_space.sample())
            previous_observation = observation.copy()
            observation, reward, terminated, truncated, info = env.step(action)
            changed_pixels = int(np.count_nonzero(previous_observation != observation))
            save_frame(observation, output_dir / f"step_{step_number:03d}.png")

            row = {
                "step": step_number,
                "action_index": action,
                "action_name": ACTION_NAMES[action],
                "changed_pixels": changed_pixels,
                "reward": reward,
                "levels_completed": info["levels_completed"],
                "game_state": info["game_state"],
                "terminated": terminated,
                "truncated": truncated,
            }
            records.append(row)
            print(
                f"step={step_number:3d} action={ACTION_NAMES[action]:4s} "
                f"changed_pixels={changed_pixels:4d} reward={reward:.1f} "
                f"levels={info['levels_completed']} state={info['game_state']}"
            )
            if terminated or truncated:
                print("Episode ended during diagnostic.")
                break
    finally:
        env.close()

    if records:
        with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    print("Saved images and trajectory to:", output_dir.resolve())


if __name__ == "__main__":
    main()
