
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import arc_agi
from arcengine import GameAction, GameState


# PPO chooses a number from 0 to 4.
# Each number represents one ARC action.
ACTIONS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    # GameAction.ACTION5,
]


class ArcEnv(gym.Env):
    """Convert an ARC game into a Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(self, game_id="ls20", max_steps=500, seed=0):
        super().__init__()

        self.game_id = game_id
        self.max_steps = max_steps
        self.seed_value = seed

        # Connect to ARC and create the selected game.
        self.arc = arc_agi.Arcade()
        self.env = self.arc.make(game_id, seed=seed)

        if self.env is None:
            raise RuntimeError(
                f"Could not create ARC game: {game_id}"
            )

        # PPO chooses 0, 1, 2, 3, or 4.
        self.action_space = spaces.Discrete(len(ACTIONS))

        # Observation: one 64 x 64 grid, stored as image values.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, 64, 64),
            dtype=np.uint8,
        )

        self.steps = 0
        self.levels = 0
        self.current_obs = None

        # Check that our five actions exist in this game's action list.
        available_actions = list(self.env.action_space)

        missing_actions = [
            action.name
            for action in ACTIONS
            if action not in available_actions
        ]

        if missing_actions:
            raise ValueError(
                f"{game_id} does not list these actions as available: "
                f"{missing_actions}. Check the game's action space "
                "before training PPO."
            )

    def _picture(self, obs):
        """Convert the ARC frame into a (1, 64, 64) image."""

        if obs is None:
            raise RuntimeError("ARC returned no observation.")

        frame = np.asarray(obs.frame)

        # Some games return multiple grids/frames.
        # Keep the most recent one.
        if frame.ndim == 3:
            frame = frame[-1]

        if frame.shape != (64, 64):
            raise ValueError(
                f"Expected a 64 x 64 grid, got {frame.shape}"
            )

        # ARC colours are 0-15; scale them to 0-240.
        frame = np.clip(frame * 16, 0, 255).astype(np.uint8)

        # Add the channel dimension: (64, 64) -> (1, 64, 64).
        return frame[np.newaxis, :, :]

    def reset(self, seed=None, options=None):
        """Start or reset an episode."""

        super().reset(seed=seed)

        # Seed Gymnasium's action space for reproducible random actions.
        if seed is not None:
            self.action_space.seed(seed)

        # Reset the ARC game itself.
        obs = self.env.reset()

        if obs is None:
            raise RuntimeError("ARC reset failed; no initial observation.")

        self.steps = 0
        self.levels = obs.levels_completed
        self.current_obs = obs

        observation = self._picture(obs)

        info = {
            "game_id": self.game_id,
            "levels_completed": self.levels,
            "episode_steps": self.steps,
        }

        return observation, info

    def step(self, action):
        """Apply one action and return the new observation and reward."""

        if self.current_obs is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        # Convert PPO's number into an ARC action.
        action_number = int(action)

        if not self.action_space.contains(action_number):
            raise ValueError(f"Invalid PPO action: {action_number}")

        arc_action = ACTIONS[action_number]

        # Do not execute an action that is unavailable in the current state.
        if arc_action not in self.env.action_space:
            raise ValueError(
                f"{arc_action.name} is not currently available. "
                "This wrapper needs action masking or a different "
                "action mapping for this game."
            )

        # Perform the ARC action.
        obs = self.env.step(arc_action)

        if obs is None:
            raise RuntimeError("ARC step failed; no observation returned.")

        self.steps += 1

        # Reward PPO when the number of completed levels increases.
        new_levels = obs.levels_completed
        reward = float(new_levels - self.levels)
        self.levels = new_levels

        # Compare actual game states, not strings.
        terminated = obs.state in (
            GameState.WIN,
            GameState.GAME_OVER,
        )

        # Time limit: this is a truncation, not a game ending.
        truncated = self.steps >= self.max_steps and not terminated

        self.current_obs = obs

        observation = self._picture(obs)

        info = {
            "game_id": self.game_id,
            "levels_completed": self.levels,
            "episode_steps": self.steps,
            "game_state": obs.state.name,
        }

        return observation, reward, terminated, truncated, info

    def close(self):
        """Release resources if the ARC environment supports it."""

        if hasattr(self.env, "close"):
            self.env.close()


if __name__ == "__main__":
    # This tests the wrapper. It does not train PPO.
    env = ArcEnv(game_id="ls20", max_steps=20, seed=0)

    observation, info = env.reset()

    print("Reset successful")
    print("Observation shape:", observation.shape)
    print("Observation type:", observation.dtype)
    print("Initial info:", info)
    print("PPO action space:", env.action_space)
    print("Observation space:", env.observation_space)

    for step in range(10):
        # Choose a random action to test the environment.
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        print(
            f"Step {step + 1}: "
            f"action={action}, "
            f"reward={reward}, "
            f"state={info['game_state']}, "
            f"levels={info['levels_completed']}"
        )

        if terminated or truncated:
            print("Episode ended.")
            break

    env.close()
    print("Environment smoke test finished.")
