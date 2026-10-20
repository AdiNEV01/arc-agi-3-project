"""Check the Gymnasium contract for the ARC adapter (no PPO training)."""

import argparse

from gymnasium.utils.env_checker import check_env

from arc_gym import ArcEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", default="ls20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    env = ArcEnv(game_id=args.game_id, max_steps=args.max_steps, seed=args.seed)
    try:
        observation, info = env.reset(seed=args.seed)
        print("Observation shape:", observation.shape)
        print("Observation dtype:", observation.dtype)
        print("Observation minimum/maximum:", int(observation.min()), int(observation.max()))
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        print("Initial info:", info)
        check_env(env, warn=True, skip_render_check=True)
        print("\nGYMNASIUM CHECK PASSED.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
