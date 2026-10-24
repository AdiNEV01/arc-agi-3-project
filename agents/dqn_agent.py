"""Frozen original DQN through Task 3. click_grid=64 retains all original actions.

Smaller grids restrict the click head to the shared lattice. No optimiser,
replay or training reward is used by this compatibility bridge.
"""
import numpy as np
import torch
from arcengine import GameAction

from agents.base import Policy
from agents.checkpoint_encoding import GridHistory
from arc_env.encoding import ActionMap
from DQN.action_codec import encode_action
from DQN.q_network import QNetwork


class DQNPolicy(Policy):
    name = "dqn-frozen-legacy"
    version = "1"

    def __init__(self, checkpoint, click_grid=8, seed=0, device="cpu"):
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.config = saved.get("config", {})
        self.num_frames = int(self.config.get("num_frames", 1))
        self.model = QNetwork(num_frames=self.num_frames).to(self.device)
        self.model.load_state_dict(saved["online_state_dict"])
        self.model.eval()
        self.history = GridHistory(self.num_frames)
        self.amap = ActionMap(click_grid)
        self.indices = torch.tensor([
            encode_action(GameAction.from_id(spec.action_id), spec.data or None)
            for spec in self.amap.specs
        ], device=self.device)

    def reset_game(self, game_id, seed):
        self.reset_episode()

    def reset_episode(self):
        self.history.clear()

    def act(self, obs, info):
        mask = np.asarray(info["action_mask"], dtype=bool)
        if mask.shape != (len(self.amap),) or not mask.any():
            raise ValueError("DQN needs a nonempty shared legal-action mask")
        state = torch.as_tensor(self.history.push(obs)[None], device=self.device)
        with torch.inference_mode():
            q = self.model(state)[0][self.indices]
            q = q.masked_fill(~torch.as_tensor(mask, device=self.device), -torch.inf)
            return int(q.argmax().item())

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info):
        if terminated or truncated:
            self.reset_episode()
