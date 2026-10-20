
"""Evaluate a frozen DQN checkpoint on ARC-AGI-3."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import arc_agi
import numpy as np
import torch
from arcengine import GameState

from DQN.action_codec import decode_action
from DQN.dqn_agent import DQNAgent
from DQN.dqn_improvement_2 import FrameHistory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="Evaluations/DQN")
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("max-actions must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    run_name = f"{checkpoint_path.stem}_eval_{args.game}_seed{args.seed}"
    for suffix in (".csv", ".json"):
        path = output_dir / (run_name + suffix)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; use --output-dir")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    num_frames = checkpoint.get("config", {}).get("num_frames", 1)
    agent = DQNAgent(seed=args.seed, device="cpu", num_frames=num_frames)
    agent.online.load_state_dict(
        checkpoint["online_state_dict"]
    )
    agent.online.eval()
    gradients_before = agent.gradient_steps
    weights_before = {key: value.clone() for key, value in agent.online.state_dict().items()}

    arc = arc_agi.Arcade()
    env = arc.make(args.game, seed=args.seed)

    if env is None:
        raise RuntimeError("Could not create environment")

    obs = env.reset()
    if obs is None:
        raise RuntimeError("Environment reset failed")

    history = FrameHistory(num_frames)
    state = history.reset(obs)
    initial_levels = obs.levels_completed
    best_levels = initial_levels
    trajectory = []
    started = time.perf_counter()

    print("\n=== FROZEN DQN EVALUATION ===")
    print("Checkpoint:", checkpoint_path)
    print("Game:", args.game)
    print("Evaluation seed:", args.seed)

    actions_used = 0

    for step in range(args.max_actions):
        # epsilon=0: select the highest-valued valid action.
        with torch.no_grad():
            action_index = agent.select_action(
                state,
                env.action_space,
                epsilon=0.0,
            )

        action, data = decode_action(action_index)

        next_obs = env.step(action, data=data)
        if next_obs is None:
            raise RuntimeError("Evaluation step failed")

        actions_used += 1
        best_levels = max(
            best_levels,
            next_obs.levels_completed,
        )

        trajectory.append({
            "step": actions_used,
            "action": action.name,
            "x": data["x"] if data else "",
            "y": data["y"] if data else "",
            "levels_completed": next_obs.levels_completed,
            "game_state": next_obs.state.name,
        })

        state = history.advance(next_obs)
        obs = next_obs

        if obs.state in (
            GameState.WIN,
            GameState.GAME_OVER,
        ):
            break

    runtime = time.perf_counter() - started

    scorecard_score = None
    try:
        scorecard = arc.get_scorecard()
        if scorecard is not None:
            raw_score = getattr(scorecard, "score", None)
            if raw_score is not None:
                scorecard_score = float(raw_score)
    except Exception as error:
        print("Scorecard retrieval failed:", error)

    run_name = (
        f"{checkpoint_path.stem}"
        f"_eval_{args.game}_seed{args.seed}"
    )

    csv_path = output_dir / f"{run_name}.csv"
    json_path = output_dir / f"{run_name}.json"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=[
                "step", "action", "x", "y",
                "levels_completed", "game_state",
            ]
        )
        writer.writeheader()
        writer.writerows(trajectory)

    unchanged = all(torch.equal(value, weights_before[key]) for key, value in agent.online.state_dict().items())
    assert unchanged and agent.gradient_steps == gradients_before, "Frozen policy changed during evaluation"

    summary = {
        "checkpoint": str(checkpoint_path),
        "game": obs.game_id,
        "evaluation_seed": args.seed,
        "actions_used": actions_used,
        "max_actions": args.max_actions,
        "levels_completed": best_levels,
        "total_levels": obs.win_levels,
        "success": obs.state == GameState.WIN,
        "final_state": obs.state.name,
        "runtime_seconds": round(runtime, 3),
        "local_scorecard_score": scorecard_score,
        "gradient_updates_during_eval": agent.gradient_steps - gradients_before,
        "num_frames": num_frames,
        "policy_parameters_unchanged": unchanged,
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== EVALUATION SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nTrajectory:", csv_path)
    print("Summary:", json_path)


if __name__ == "__main__":
    main()
