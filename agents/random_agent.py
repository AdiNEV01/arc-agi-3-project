"""Random reference policy (sanity check, assignment requirement 2).

Sampling is two-stage so ACTION6 (64 click targets by default) does not drown
out the simple actions: first a legal action TYPE uniformly, then, for ACTION6,
a click cell uniformly. Set mode="flat" to sample uniformly over legal indices.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_env.encoding import ActionMap
from .base import Policy


class RandomPolicy(Policy):
    name = "random"
    version = "1.0"

    def __init__(self, click_grid: int = 8, seed: int = 0, mode: str = "type_uniform"):
        self.amap = ActionMap(click_grid)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self._by_type: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.amap)):
            self._by_type[self.amap.type_of(i)].append(i)

    def reset_game(self, game_id: str, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info) -> int:
        mask = np.asarray(info["action_mask"], dtype=bool)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError("No legal actions (game over without reset?)")
        if self.mode == "flat":
            return int(self.rng.choice(legal))
        types = sorted({self.amap.type_of(i) for i in legal})
        t = types[self.rng.integers(len(types))]
        options = [i for i in self._by_type[t] if mask[i]]
        return int(options[self.rng.integers(len(options))])
