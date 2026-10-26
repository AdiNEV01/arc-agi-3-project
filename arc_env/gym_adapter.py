"""Gymnasium adapter for ARC-AGI-3 (the native interface is NOT Gymnasium).

Semantics (also documented in docs/interface.md):

* One adapter instance == one ``Arcade.make`` call == one game session.
  This mirrors competition mode, where each environment may be made only once.
* observation : uint8 Box(64, 64), last frame of the response (see encoding.py).
* action      : Discrete(len(ActionMap)); RESET is never an agent action.
* action_masks(): legal actions from the latest ``available_actions``
  (compatible with sb3-contrib MaskablePPO).
* reward      : TRAINING signal only, never the ARC-AGI-3 score.
                "levels" mode: +1 for each newly completed level, 0 otherwise.
                Teams may add shaping through ``reward_fn`` (Task 1/2).
* terminated  : state is WIN or GAME_OVER.
* truncated   : ``max_episode_actions`` reached in this episode.
* reset()     : first call uses the frame from Arcade.make(), which has already
                started the game (no extra RESET). Later calls send RESET:
                reset_mode="level" -> level reset (progress kept; the only kind
                allowed in competition mode). reset_mode="full" -> restart from
                level 1 (training only; sends a second RESET if needed).
* Illegal action (not in available_actions): NOT sent to the game, so it does
  not consume an official action. Observation is unchanged, reward is
  ``illegal_action_penalty`` and info["illegal_action"] is True. It still counts
  towards the episode step limit, so a policy cannot loop forever.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from arcengine import GameAction, GameState

from .encoding import GRID, PAD_VALUE, ActionMap, frame_to_grid

RewardFn = Callable[[dict[str, Any], dict[str, Any]], float]


class ArcGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        game_id: str,
        arcade: Any = None,
        game_seed: int = 0,
        scorecard_id: Optional[str] = None,
        click_grid: int = 8,
        reset_mode: str = "level",
        max_episode_actions: Optional[int] = None,
        illegal_action_penalty: float = 0.0,
        reward_fn: Optional[RewardFn] = None,
        save_recording: bool = False,
    ):
        super().__init__()
        if reset_mode not in ("level", "full"):
            raise ValueError("reset_mode must be 'level' or 'full'")
        self.game_id = game_id
        self.arcade = arcade
        self.game_seed = game_seed
        self.scorecard_id = scorecard_id
        self.reset_mode = reset_mode
        self.max_episode_actions = max_episode_actions
        self.illegal_action_penalty = illegal_action_penalty
        self.reward_fn = reward_fn
        self.save_recording = save_recording

        self.amap = ActionMap(click_grid)
        self.action_space = gym.spaces.Discrete(len(self.amap))
        self.observation_space = gym.spaces.Box(0, PAD_VALUE, shape=(GRID, GRID), dtype=np.uint8)

        self._env = None
        self._last = None
        self._obs = np.full((GRID, GRID), PAD_VALUE, dtype=np.uint8)
        # counters
        self.official_actions = 0    # actions actually sent to the game (excl. RESET)
        self.resets_sent = 0         # RESETs sent after the initial start
        self.episode_steps = 0
        self.full_game_info = None

    # ------------------------------------------------------------------ helpers
    def _ensure_env(self):
        if self._env is not None:
            return
        if self.arcade is None:
            import arc_agi
            self.arcade = arc_agi.Arcade()
        self._env = self.arcade.make(
            self.game_id, seed=self.game_seed, scorecard_id=self.scorecard_id,
            save_recording=self.save_recording,
        )
        if self._env is None:
            raise RuntimeError(f"Could not create environment '{self.game_id}'")
        self.full_game_info = getattr(self._env, "info", None)

    def _snapshot(self, frame) -> dict[str, Any]:
        return {
            "state": frame.state.value if hasattr(frame.state, "value") else str(frame.state),
            "levels_completed": int(frame.levels_completed),
            "win_levels": int(frame.win_levels),
            "available_actions": [int(a) for a in frame.available_actions],
            "full_reset": bool(getattr(frame, "full_reset", False)),
            "n_frames": len(frame.frame),
        }

    def _info(self, extra: Optional[dict] = None) -> dict[str, Any]:
        snap = self._snapshot(self._last)
        info = {
            **snap,
            "game_id": getattr(self._last, "game_id", self.game_id) or self.game_id,
            "official_actions": self.official_actions,
            "resets_sent": self.resets_sent,
            "episode_steps": self.episode_steps,
            "action_mask": self.action_masks(),
        }
        if extra:
            info.update(extra)
        return info

    def _accept(self, frame):
        if frame is None:
            raise RuntimeError("ARC environment returned None (step/reset failed)")
        self._last = frame
        if len(frame.frame) > 0:          # WIN/GAME_OVER may return no frame
            self._obs = frame_to_grid(frame.frame)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        first = self._env is None
        self._ensure_env()
        started = getattr(self._env, "observation_space", None) if first else None
        if started is not None and len(started.frame) > 0:
            # Arcade.make() already sent the starting RESET (one scorecard play).
            # Sending another would open a second, empty play on the scorecard.
            frame = started
        else:
            frame = self._env.reset()
            if not first:
                self.resets_sent += 1
        if self.reset_mode == "full" and not first and not getattr(frame, "full_reset", False):
            # A RESET after actions is a level reset; a second consecutive RESET is a full reset.
            frame = self._env.reset()
            self.resets_sent += 1
        self._accept(frame)
        self.episode_steps = 0
        return self._obs.copy(), self._info()

    def step(self, action: int):
        if self._last is None:
            raise RuntimeError("Call reset() before step()")
        spec = self.amap.decode(action)
        before = self._snapshot(self._last)
        self.episode_steps += 1
        t0 = time.perf_counter()

        if spec.action_id not in before["available_actions"]:
            reward = self.illegal_action_penalty
            extra = {"illegal_action": True, "action_label": spec.label(), "step_time_s": 0.0}
        else:
            frame = self._env.step(spec.game_action, data=spec.data or None)
            self.official_actions += 1
            self._accept(frame)
            after = self._snapshot(self._last)
            if self.reward_fn is not None:
                reward = float(self.reward_fn(before, after))
            else:
                reward = float(max(0, after["levels_completed"] - before["levels_completed"]))
            extra = {"illegal_action": False, "action_label": spec.label(),
                     "step_time_s": time.perf_counter() - t0}

        state = self._last.state
        terminated = state in (GameState.WIN, GameState.GAME_OVER)
        truncated = (not terminated and self.max_episode_actions is not None
                     and self.episode_steps >= self.max_episode_actions)
        return self._obs.copy(), reward, terminated, truncated, self._info(extra)

    def action_masks(self) -> np.ndarray:
        if self._last is None:
            return np.zeros(len(self.amap), dtype=bool)
        return self.amap.mask(self._last.available_actions)

    @property
    def raw_frame(self):
        """Latest native FrameDataRaw (for analysis only, not for the policy)."""
        return self._last
