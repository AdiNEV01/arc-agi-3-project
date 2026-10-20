"""Gymnasium adapter for the ARC-AGI-3 ls20 game.

The current action abstraction intentionally exposes four movement actions.
This is appropriate for the current ls20 experiment, not a universal wrapper
for every ARC-AGI-3 game (some games require clicks or coordinate actions).

The agent receives only the rendered observation and the legal action mapping.
No hidden game variables or solution scripts are used by the policy.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import arc_agi
from arcengine import GameAction, GameState


# Stable policy action indices. Keep this table unchanged for a trained model.
ACTIONS = [
    GameAction.ACTION1,  # up
    GameAction.ACTION2,  # down
    GameAction.ACTION3,  # left
    GameAction.ACTION4,  # right
]
ACTION_NAMES = ["up", "down", "left", "right"]
OBSERVATION_SHAPE = (1, 64, 64)


class ArcEnv(gym.Env):
    """A small Gymnasium wrapper around one ARC-AGI-3 game instance."""

    metadata = {"render_modes": []}

    def __init__(self, game_id: str = "ls20", max_steps: int = 500, seed: int = 0):
        super().__init__()
        self.render_mode = None

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.game_id = game_id
        self.max_steps = int(max_steps)
        self.game_seed = int(seed)

        # The ARC Toolkit environment is seeded when it is created.
        self.arcade = arc_agi.Arcade()
        self.env = self.arcade.make(self.game_id, seed=self.game_seed)
        if self.env is None:
            raise RuntimeError(f"ARC could not create game '{self.game_id}'.")

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=OBSERVATION_SHAPE,
            dtype=np.uint8,
        )

        available_actions = list(self.env.action_space)
        missing_actions = [
            action.name for action in ACTIONS if action not in available_actions
        ]
        if missing_actions:
            self.close()
            raise ValueError(
                f"Game '{self.game_id}' does not expose the expected four "
                f"movement actions. Missing: {missing_actions}. This adapter "
                "is intended for the current ls20 experiment."
            )

        self.steps = 0
        self.levels_completed = 0
        self.current_observation_data = None
        self.episode_extrinsic_return = 0.0

    def _picture(self, observation_data: Any) -> np.ndarray:
        """Convert the toolkit frame to a (1, 64, 64) uint8 observation."""
        if observation_data is None:
            raise RuntimeError("ARC returned an empty observation.")

        frame = np.asarray(observation_data.frame)

        # The current toolkit frame can include a leading frame/layer dimension.
        # Keep the same last-frame convention used by the working project code.
        if frame.ndim == 3:
            frame = frame[-1]

        if frame.shape != OBSERVATION_SHAPE[1:]:
            raise ValueError(
                "Unexpected ARC frame shape: "
                f"{frame.shape}; expected {OBSERVATION_SHAPE[1:]}. "
                "Inspect the toolkit frame before changing this conversion."
            )

        frame = frame.astype(np.float32, copy=False)

        # ARC puzzle frames often encode palette colours as integers 0..15.
        # Spread those values across the uint8 intensity range for the CNN.
        # Already-rendered 0..255 images are left at their existing scale.
        if frame.size and float(np.nanmax(frame)) <= 15:
            frame = frame * 16.0

        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        result = frame[np.newaxis, :, :]

        if result.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape mismatch: {result.shape}")
        return result

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the current game and return observation plus diagnostic info.

        Important: `seed` here seeds Gymnasium's RNG and action space. The ARC
        game's own seed is set in `__init__`; reset(seed=...) does not recreate
        the ARC game with a new layout. Create a new ArcEnv(seed=...) for that.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)

        observation_data = self.env.reset()
        if observation_data is None:
            raise RuntimeError("ARC reset returned no observation.")

        self.steps = 0
        self.levels_completed = int(
            getattr(observation_data, "levels_completed", 0)
        )
        self.current_observation_data = observation_data
        self.episode_extrinsic_return = 0.0

        observation = self._picture(observation_data)
        info = {
            "game_id": self.game_id,
            "game_seed": self.game_seed,
            "levels_completed": self.levels_completed,
            "episode_steps": self.steps,
            "game_state": getattr(
                getattr(observation_data, "state", None), "name", "UNKNOWN"
            ),
            "extrinsic_reward": 0.0,
            "intrinsic_reward": 0.0,
        }
        return observation, info

    def step(self, action: int):
        """Apply one of the four mapped movement actions."""
        if self.current_observation_data is None:
            raise RuntimeError("Call reset() before step().")

        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError(f"Action index {action_index} is outside Discrete(4).")

        arc_action = ACTIONS[action_index]
        available_actions = list(self.env.action_space)
        if arc_action not in available_actions:
            raise ValueError(
                f"{arc_action.name} is not available in the current game state. "
                "This four-action adapter does not implement dynamic action masking."
            )

        observation_data = self.env.step(arc_action)
        if observation_data is None:
            raise RuntimeError("ARC step returned no observation.")

        self.steps += 1
        previous_levels_completed = self.levels_completed
        new_levels_completed = int(
            getattr(observation_data, "levels_completed", self.levels_completed)
        )
        reward = float(new_levels_completed - previous_levels_completed)
        self.levels_completed = new_levels_completed
        self.episode_extrinsic_return += reward

        state = getattr(observation_data, "state", None)
        state_name = getattr(state, "name", "UNKNOWN")
        terminated = state in (GameState.WIN, GameState.GAME_OVER)
        truncated = self.steps >= self.max_steps and not terminated

        self.current_observation_data = observation_data
        observation = self._picture(observation_data)

        info = {
            "game_id": self.game_id,
            "game_seed": self.game_seed,
            "levels_completed": self.levels_completed,
            "levels_delta": new_levels_completed - previous_levels_completed,
            "episode_steps": self.steps,
            "game_state": state_name,
            "extrinsic_reward": reward,
            "intrinsic_reward": 0.0,
        }
        if terminated or truncated:
            info["episode_extrinsic_return"] = self.episode_extrinsic_return
            info["episode_intrinsic_return"] = 0.0
            info["episode_total_return"] = self.episode_extrinsic_return

        return observation, reward, terminated, truncated, info

    def close(self):
        """Close the toolkit environment when it exposes a close method."""
        if getattr(self, "env", None) is not None:
            close_method = getattr(self.env, "close", None)
            if callable(close_method):
                close_method()
