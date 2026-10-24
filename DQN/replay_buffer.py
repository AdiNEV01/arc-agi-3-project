
"""Experience replay for ARC-AGI-3 DQN."""

from dataclasses import dataclass
from random import Random

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    next_valid_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity=3000):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.storage = []
        self.position = 0

    def __len__(self):
        return len(self.storage)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        next_valid_mask,
    ):
        # Store raw uint8 grids instead of large float tensors.
        transition = Transition(
            state=np.asarray(
                state, dtype=np.uint8
            ).copy(),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(
                next_state, dtype=np.uint8
            ).copy(),
            terminated=bool(terminated),
            next_valid_mask=np.asarray(
                next_valid_mask, dtype=np.bool_
            ).copy(),
        )

        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition

        self.position = (
            self.position + 1
        ) % self.capacity

    def sample(self, batch_size, rng: Random):
        if len(self.storage) < batch_size:
            raise ValueError(
                "Not enough transitions to sample"
            )

        return rng.sample(self.storage, batch_size)
