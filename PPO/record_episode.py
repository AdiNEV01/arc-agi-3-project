"""Record a policy episode as frames, an action trajectory CSV, and a GIF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from arc_gym import ACTION_NAMES, ArcEnv
from grid_observation import OneHotGridObservation, grid_observation_to_board
from policy_networks import GridCNNFeaturesExtractor  # noqa: F401


def board_for_display(observation: np.ndarray, observation_mode: str) -> np.ndarray:
    """Get one 64x64 board from either observation format for display."""
    if observation_mode == "grid":
        return grid_observation_to_board(observation)
    return observation[0]


def save_annotated_frame(observation: np.ndarray, path: Path, step: int,
                         total_reward: float, state: str,
                         observation_mode: str) -> None:
    """Save the board enlarged with nearest-neighbour pixels."""
    board = board_for_display(observation, observation_mode)
    image = Image.fromarray(board, mode="L").convert("RGB")
    image = image.resize((512, 512), resample=Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    label = f"Step {step} | return {total_reward:.2f} | {state}"
    draw.rectangle((0, 0, 512, 28), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "ppo"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--observation-mode", choices=["standard", "grid"], default="standard",
        help="Use standard observations, or the one-hot grid encoding required by grid_ppo.py.",
    )
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample PPO actions; default is deterministic.")
    parser.add_argument("--output-dir", default="outputs/replays/episode_0")
    args = parser.parse_args()

    if args.policy == "ppo" and not args.model:
        parser.error("--model is required when --policy ppo")
    if args.policy == "ppo" and args.stochastic:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model = PPO.load(args.model, device="cpu") if args.policy == "ppo" else None
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Replay output directory is not empty: {output_dir.resolve()}. "
            "Choose a new --output-dir to avoid mixing frames from different episodes."
        )
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = ArcEnv(args.game_id, max_steps=args.max_steps, seed=args.seed)
    if args.observation_mode == "grid":
        env = OneHotGridObservation(env)
    trajectory = []
    total_reward = 0.0

    try:
        observation, info = env.reset(seed=args.seed)
        save_annotated_frame(
            observation, frames_dir / "step_000.png", 0, 0.0,
            info.get("game_state", "RESET"), args.observation_mode,
        )

        terminated = False
        truncated = False
        step_number = 0
        while not (terminated or truncated):
            if model is None:
                action = int(env.action_space.sample())
            else:
                predicted, _ = model.predict(
                    observation,
                    deterministic=not args.stochastic,
                )
                action = int(np.asarray(predicted).item())

            previous_board = board_for_display(observation, args.observation_mode)
            observation, reward, terminated, truncated, info = env.step(action)
            step_number += 1
            total_reward += float(reward)
            current_board = board_for_display(observation, args.observation_mode)
            changed_pixels = int(np.count_nonzero(previous_board != current_board))

            save_annotated_frame(
                observation,
                frames_dir / f"step_{step_number:03d}.png",
                step_number,
                total_reward,
                info.get("game_state", "UNKNOWN"), args.observation_mode,
            )
            trajectory.append({
                "step": step_number,
                "action_index": action,
                "action_name": ACTION_NAMES[action],
                "reward": float(reward),
                "cumulative_extrinsic_return": total_reward,
                "changed_pixels": changed_pixels,
                "levels_completed": info.get("levels_completed", 0),
                "game_state": info.get("game_state", "UNKNOWN"),
                "terminated": int(bool(terminated)),
                "truncated": int(bool(truncated)),
            })

            if step_number % 25 == 0 or terminated or truncated:
                print(
                    f"step={step_number:3d} action={ACTION_NAMES[action]:4s} "
                    f"reward={reward:.1f} levels={info.get('levels_completed', 0)} "
                    f"state={info.get('game_state', 'UNKNOWN')}"
                )

    finally:
        env.close()

    trajectory_path = output_dir / "trajectory.csv"
    if trajectory:
        with trajectory_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=trajectory[0].keys())
            writer.writeheader()
            writer.writerows(trajectory)

    # Use the shared GIF builder after saving PNGs.
    gif_path = output_dir / "episode.gif"
    frame_paths = sorted(frames_dir.glob("step_*.png"))
    if len(frame_paths) >= 2:
        gif_frames = []
        for path in frame_paths:
            with Image.open(path) as frame:
                gif_frames.append(frame.convert("RGB").quantize(colors=256))
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=150,
            loop=0,
            optimize=False,
            disposal=2,
        )

    print(f"Episode steps: {step_number}")
    print(f"Extrinsic return: {total_reward:.3f}")
    print(f"Levels completed: {info.get('levels_completed', 0)}")
    print(f"Ending state: {info.get('game_state', 'UNKNOWN')}")
    print(f"Trajectory CSV: {trajectory_path.resolve()}")
    if gif_path.exists():
        print(f"Replay GIF: {gif_path.resolve()}")
    print(f"Frame directory: {frames_dir.resolve()}")


if __name__ == "__main__":
    main()
