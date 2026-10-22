"""Evaluate a random policy or saved PPO checkpoint under matched conditions.

One ArcEnv is created for each value in --seeds. The ARC game's seed is set
when that environment is created. Repeated resets within that environment do
not reseed the ARC game; they do reseed the Gymnasium action space for random
policy reproducibility. Use the same --seeds and --episodes-per-seed for each
policy being compared.

Results are placed in outputs/evaluations/<tag>_<timestamp>/episodes.csv and
summary.json. The official ARC competition score is not calculated here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

# Import the custom extractor module so its class is available when custom
# policy checkpoints are deserialized.
from policy_networks import GridCNNFeaturesExtractor  # noqa: F401
from arc_gym import ACTION_NAMES, ArcEnv
from grid_observation import OneHotGridObservation


def safe_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return cleaned.strip("._-") or "evaluation"


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, default=str)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a random or PPO policy.")
    parser.add_argument("--policy", choices=["random", "ppo"], required=True)
    parser.add_argument("--model", default=None,
                        help="Path to a saved PPO .zip file (required for PPO).")
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--observation-mode", choices=["standard", "grid"], default="standard",
        help="Use standard observations, or the one-hot grid encoding required by grid_ppo.py.",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample PPO actions. Default: deterministic actions.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-root", default="outputs/evaluations")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.policy == "ppo" and not args.model:
        parser.error("--model is required when --policy ppo")
    if args.episodes_per_seed < 1:
        parser.error("--episodes-per-seed must be at least 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but torch.cuda.is_available() is False")

    model = None
    saved_model_timesteps = None
    if args.policy == "ppo":
        model = PPO.load(args.model, device=args.device)
        saved_model_timesteps = int(getattr(model, "num_timesteps", 0))
        print(f"Loaded PPO model: {args.model}")
        print(f"Model training timesteps recorded in checkpoint: {saved_model_timesteps:,}")
        print(f"PPO action mode: {'stochastic' if args.stochastic else 'deterministic'}")

    tag = args.tag or args.policy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{safe_tag(tag)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    episodes_path = run_dir / "episodes.csv"

    fields = [
        "tag", "policy", "game_id", "game_seed", "episode_index",
        "observation_mode", "action_seed", "levels_completed", "success", "steps",
        "end_state", "terminated", "truncated", "extrinsic_return",
        "model_path", "ppo_stochastic", "elapsed_seconds",
    ]

    rows = []
    started_at = time.perf_counter()

    with episodes_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()

        for game_seed in args.seeds:
            env = ArcEnv(
                game_id=args.game_id,
                max_steps=args.max_steps,
                seed=game_seed,
            )
            if args.observation_mode == "grid":
                env = OneHotGridObservation(env)
            try:
                for episode_index in range(args.episodes_per_seed):
                    # Resetting the action space allows random action sequences
                    # to be reproduced without pretending to reseed the ARC game.
                    action_seed = int(game_seed * 100_000 + episode_index + 1)
                    observation, info = env.reset(seed=action_seed)
                    env.action_space.seed(action_seed)
                    if args.policy == "ppo" and args.stochastic:
                        torch.manual_seed(action_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(action_seed)

                    episode_started = time.perf_counter()
                    terminated = False
                    truncated = False
                    steps = 0
                    extrinsic_return = 0.0
                    levels_completed = int(info.get("levels_completed", 0))
                    end_state = "UNKNOWN"

                    while not (terminated or truncated):
                        if model is None:
                            action = int(env.action_space.sample())
                        else:
                            predicted_action, _ = model.predict(
                                observation,
                                deterministic=not args.stochastic,
                            )
                            action = int(np.asarray(predicted_action).item())

                        observation, reward, terminated, truncated, info = env.step(action)
                        steps += 1
                        extrinsic_return += float(reward)
                        levels_completed = max(
                            levels_completed,
                            int(info.get("levels_completed", 0)),
                        )

                    if terminated:
                        end_state = info.get("game_state", "UNKNOWN")
                    elif truncated:
                        end_state = "STEP_LIMIT"

                    row = {
                        "tag": tag,
                        "policy": args.policy,
                        "game_id": args.game_id,
                        "game_seed": game_seed,
                        "episode_index": episode_index,
                        "observation_mode": args.observation_mode,
                        "action_seed": action_seed,
                        "levels_completed": levels_completed,
                        "success": int(levels_completed > 0),
                        "steps": steps,
                        "end_state": end_state,
                        "terminated": int(bool(terminated)),
                        "truncated": int(bool(truncated)),
                        "extrinsic_return": extrinsic_return,
                        "model_path": args.model or "",
                        "ppo_stochastic": int(bool(args.stochastic)) if model is not None else "",
                        "elapsed_seconds": time.perf_counter() - episode_started,
                    }
                    writer.writerow(row)
                    output_file.flush()
                    rows.append(row)
                    print(
                        f"{tag} | game_seed={game_seed} | episode={episode_index} | "
                        f"levels={levels_completed} | steps={steps} | end={end_state}"
                    )
            finally:
                env.close()

    count = len(rows)
    successes = sum(int(row["success"]) for row in rows)
    summary = {
        "tag": tag,
        "policy": args.policy,
        "game_id": args.game_id,
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "observation_mode": args.observation_mode,
        "episodes": count,
        "successful_episodes": successes,
        "success_rate": successes / count if count else 0.0,
        "mean_levels_completed": (
            sum(int(row["levels_completed"]) for row in rows) / count if count else 0.0
        ),
        "mean_steps": sum(int(row["steps"]) for row in rows) / count if count else 0.0,
        "mean_extrinsic_return": (
            sum(float(row["extrinsic_return"]) for row in rows) / count if count else 0.0
        ),
        "ending_states": {
            state: sum(row["end_state"] == state for row in rows)
            for state in sorted({row["end_state"] for row in rows})
        },
        "stochastic_ppo": bool(args.stochastic) if model is not None else None,
        "model_path": args.model,
        "model_training_timesteps": saved_model_timesteps,
        "elapsed_seconds": time.perf_counter() - started_at,
        "local_levels_completed_are_official_arc_score": False,
        "episode_results_csv": str(episodes_path.resolve()),
    }
    write_json(run_dir / "summary.json", summary)

    print(f"\n{tag}: {count} episodes")
    print(f"Successful episodes: {successes}/{count} ({100 * summary['success_rate']:.1f}%)")
    print(f"Mean levels completed: {summary['mean_levels_completed']:.3f}")
    print(f"Mean steps per episode: {summary['mean_steps']:.2f}")
    print(f"Ending states: {summary['ending_states']}")
    print(f"Results saved under: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
