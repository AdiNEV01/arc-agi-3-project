"""Tiny fake ARC game for offline tests (no game download needed).

Rules: 2 levels. Three ACTION1 presses complete a level. ACTION2 -> GAME_OVER.
ACTION6 paints the clicked cell. Reset semantics copy arcengine.handle_reset.
"""
import os
from types import SimpleNamespace

import numpy as np
from arcengine import GameAction, GameState
from arcengine.enums import FrameDataRaw


class FakeGameEnv:
    def __init__(self, game_id="fake-0001"):
        self.info = SimpleNamespace(game_id=game_id, title="Fake", baseline_actions=[3, 3], tags=[])
        self.level, self.presses, self.levels_completed = 0, 0, 0
        self.action_count, self.state = 0, GameState.NOT_PLAYED
        self.grid = np.zeros((64, 64), dtype=np.int8)

    def _frame(self, full_reset=False, frames=True):
        f = FrameDataRaw()
        f.game_id, f.state = self.info.game_id, self.state
        f.levels_completed, f.win_levels = self.levels_completed, 2
        f.available_actions = [] if self.state in (GameState.GAME_OVER, GameState.WIN) else [1, 2, 6]
        f.full_reset = full_reset
        g = self.grid.copy(); g[0, :self.presses] = 3; g[1, :self.level] = 5
        f.frame = [g, g] if frames else []          # 2 animation frames per action
        return f

    def reset(self):
        only_levels = os.getenv("ONLY_RESET_LEVELS") == "true" and self.state != GameState.WIN
        full = (not only_levels) and (self.action_count == 0 or self.state == GameState.WIN)
        if full:
            self.level, self.levels_completed, self.action_count = 0, 0, 0
            self.grid[:] = 0
        self.presses, self.state = 0, GameState.NOT_FINISHED
        self.action_count = 0          # arcengine set_level() zeroes the counter on any reset
        return self._frame(full_reset=full)

    @property
    def action_space(self):
        return [GameAction.from_id(a) for a in self._frame().available_actions]

    def step(self, action, data=None):
        if self.state in (GameState.GAME_OVER, GameState.WIN):
            return self._frame(frames=False)
        self.action_count += 1
        if action.value == 1:
            self.presses += 1
            if self.presses == 3:
                self.levels_completed += 1; self.level += 1; self.presses = 0
                self.action_count = 0      # set_level() on level change
                if self.levels_completed == 2:
                    self.state = GameState.WIN
        elif action.value == 2:
            self.state = GameState.GAME_OVER
        elif action.value == 6:
            self.grid[data["y"], data["x"]] = 9
        return self._frame()


class FakeArcade:
    def __init__(self):
        self.made = []

    def make(self, game_id, seed=0, scorecard_id=None, save_recording=False, **kw):
        self.made.append(game_id)
        return FakeGameEnv(game_id)

    def create_scorecard(self, tags=None):
        return "card"

    def get_scorecard(self, card_id=None):
        return None

    def close_scorecard(self, card_id=None):
        return None
