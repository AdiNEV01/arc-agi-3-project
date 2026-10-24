"""Frozen seven-action MaskablePPO checkpoints through the shared evaluator.

Masks are passed explicitly. ACTION6 samples a legal shared click uniformly
using a per-game seeded RNG. click_grid=64 preserves full-pixel random clicks.
"""
import numpy as np
from sb3_contrib import MaskablePPO

from agents.base import Policy
from agents.checkpoint_encoding import GridHistory
from arc_env.encoding import ActionMap


class PPOPolicy(Policy):
    name = "ppo-frozen-legacy"
    version = "1"

    def __init__(self, checkpoint, click_grid=8, seed=0, device="cpu"):
        self.model = MaskablePPO.load(checkpoint, device=device)
        if self.model.action_space.n != 7:
            raise ValueError("This bridge accepts legacy seven-action PPO checkpoints only")
        channels, h, w = self.model.observation_space.shape
        if h != 64 or w != 64 or channels not in (16, 64):
            raise ValueError("Expected legacy one- or four-frame PPO checkpoint")
        self.history = GridHistory(channels // 16)
        self.amap = ActionMap(click_grid)
        self.rng = np.random.default_rng(seed)

    def reset_game(self, game_id, seed):
        self.reset_episode()
        self.rng = np.random.default_rng(seed)

    def reset_episode(self):
        self.history.clear()

    def act(self, obs, info):
        mask = np.asarray(info["action_mask"], dtype=bool)
        if mask.shape != (len(self.amap),) or not mask.any():
            raise ValueError("PPO needs a nonempty shared legal-action mask")
        candidates = [np.array([
            i for i, spec in enumerate(self.amap.specs)
            if spec.action_id == action_id and mask[i]
        ], dtype=int) for action_id in range(1, 8)]
        native_mask = np.array([len(xs) > 0 for xs in candidates], dtype=bool)
        action, _ = self.model.predict(self.history.push(obs), deterministic=True,
                                       action_masks=native_mask)
        choices = candidates[int(np.asarray(action).item())]
        if not len(choices):
            raise RuntimeError("PPO returned a masked action")
        return int(choices[0] if len(choices) == 1 else self.rng.choice(choices))

    def observe(self, obs, action, reward, next_obs, terminated, truncated, info):
        if terminated or truncated:
            self.reset_episode()
