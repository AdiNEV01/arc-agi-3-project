"""Bridge a Policy into the Kaggle starter's agent/my_agent.py.

The Kaggle starter calls choose_action(frames, latest_frame) with FrameData
objects. This helper applies the SAME encoding as the Gymnasium adapter, so a
policy trained through ArcGymEnv sees identical inputs on Kaggle.

Usage inside agent/my_agent.py (starter kit):

    from agents.kaggle_bridge import PolicyBridge
    BRIDGE = PolicyBridge(load_my_policy())          # load checkpoint once

    class MyAgent(Agent):
        def is_done(self, frames, latest_frame):
            return latest_frame.state is GameState.WIN
        def choose_action(self, frames, latest_frame):
            return BRIDGE.choose_action(latest_frame)
"""
from __future__ import annotations

from arcengine import GameAction, GameState

from arc_env.encoding import ActionMap, frame_to_grid


class PolicyBridge:
    def __init__(self, policy, click_grid: int = 8):
        self.policy = policy
        self.amap = ActionMap(click_grid)
        self._game = None

    def choose_action(self, latest_frame) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET          # becomes a level reset in competition mode
        if latest_frame.game_id != self._game:
            self._game = latest_frame.game_id
            self.policy.reset_game(self._game, seed=0)
        obs = frame_to_grid(latest_frame.frame)
        info = {"action_mask": self.amap.mask(latest_frame.available_actions),
                "levels_completed": latest_frame.levels_completed,
                "win_levels": latest_frame.win_levels}
        spec = self.amap.decode(self.policy.act(obs, info))
        action = spec.game_action
        if spec.data:
            action.set_data(spec.data)
        return action
