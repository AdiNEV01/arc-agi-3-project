"""Common policy interface used by evaluate.py.

DQN and PPO agents (Tasks 1 and 2) should implement this, so all agents are
evaluated by the same script under the same protocol.

What persists (report this, assignment section 4.4):
* parameters loaded from a checkpoint persist across games and evaluation runs;
* ``reset_game`` is called before each game, so any per-game memory must be
  cleared there; ``observe`` is where online adaptation during evaluation
  happens (leave it as a no-op if the agent does not adapt online).
"""
from __future__ import annotations

from typing import Any

import numpy as np


class Policy:
    name: str = "policy"
    version: str = "0"

    def reset_game(self, game_id: str, seed: int) -> None:
        """Called once before each evaluation game. Clear per-game memory here."""

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        """Return a Discrete action index. info['action_mask'] lists legal indices."""
        raise NotImplementedError

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info) -> None:
        """Optional online update. Default: no learning during evaluation."""
