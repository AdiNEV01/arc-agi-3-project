
"""DQN V1: Count-based observation novelty reward."""

from collections import Counter
from hashlib import blake2b
import math

import numpy as np


class StateNoveltyReward:
    def __init__(self, beta=0.01):
        if beta < 0:
            raise ValueError("beta must be non-negative")

        self.beta = beta
        self.visits = Counter()

    @staticmethod
    def state_key(obs):
        # Use only observations available to the agent.
        frame = np.asarray(
            obs.frame[-1], dtype=np.uint8
        )

        return blake2b(
            frame.tobytes(),
            digest_size=16
        ).digest()

    def register_reset(self, obs):
        """Count the initial observation after a reset."""
        self.visits[self.state_key(obs)] += 1

    def bonus(self, obs, next_obs):
        """Calculate novelty bonus after an action."""
        before = self.state_key(obs)
        after = self.state_key(next_obs)

        self.visits[after] += 1

        # No bonus when the observation is unchanged.
        if before == after:
            return 0.0

        count = self.visits[after]

        # Less frequently visited states get larger bonuses.
        return self.beta / math.sqrt(count)

    @property
    def unique_states(self):
        return len(self.visits)
