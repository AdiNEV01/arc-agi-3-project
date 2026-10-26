"""Expose the shared Gymnasium contract to the existing native DQN loop.

Only permitted grid and info fields are used. Full-pixel actions preserve the
original Q head exactly. RESET remains controller-owned, outside selected actions.
"""
from types import SimpleNamespace

from arcengine import GameAction, GameState

from arc_env.gym_adapter import ArcGymEnv
from DQN.action_codec import N_ACTIONS, encode_action


class SharedDQNEnvironment:
    def __init__(self, game, arcade, seed, episode_limit=0, reset_mode="level"):
        self.shared = ArcGymEnv(game, arcade=arcade, game_seed=seed, click_grid=64,
                                reset_mode=reset_mode,
                                max_episode_actions=episode_limit or None)
        if self.shared.action_space.n != N_ACTIONS:
            raise ValueError("DQN training requires the full-pixel shared action map")
        for index, spec in enumerate(self.shared.amap.specs):
            if encode_action(spec.game_action, spec.data or None) != index:
                raise ValueError("Native and shared action maps differ")
        self.info = None

    def _view(self, grid, info, reward=0.0, terminated=False, truncated=False):
        self.info = info
        return SimpleNamespace(frame=[grid.copy()], levels_completed=info["levels_completed"],
                               state=GameState(info["state"]), win_levels=info["win_levels"],
                               task_reward=reward, terminated=terminated, truncated=truncated,
                               info=info.copy())

    @property
    def action_space(self):
        return [GameAction.from_id(a) for a in self.info["available_actions"]]

    def reset(self):
        grid, info = self.shared.reset()
        return self._view(grid, info)

    def step(self, action, data=None):
        grid, reward, term, trunc, info = self.shared.step(encode_action(action, data))
        if info["illegal_action"]:
            raise RuntimeError("DQN selected an illegal action")
        return self._view(grid, info, reward, term, trunc)
