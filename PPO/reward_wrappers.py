"""Observation-only reward shaping used by one PPO exploration experiment."""

from __future__ import annotations

import hashlib
import math

import gymnasium as gym
import numpy as np


class CountNoveltyReward(gym.Wrapper):
    """Add a small visit-count bonus based only on the visible observation.

    This is an experimental exploration strategy, not the official ARC score.
    The extrinsic reward and intrinsic bonus are logged separately. The policy
    cannot access hidden game variables through this wrapper.
    """

    def __init__(self, env: gym.Env, bonus_scale: float = 0.005):
        super().__init__(env)
        if bonus_scale < 0:
            raise ValueError("bonus_scale must be non-negative")
        self.bonus_scale = float(bonus_scale)
        self.visit_counts: dict[bytes, int] = {}
        self.episode_extrinsic_return = 0.0
        self.episode_intrinsic_return = 0.0

    @staticmethod
    def _observation_key(observation: np.ndarray) -> bytes:
        array = np.ascontiguousarray(observation)
        return hashlib.blake2b(array.tobytes(), digest_size=8).digest()

    def _count_observation(self, observation: np.ndarray) -> int:
        key = self._observation_key(observation)
        count = self.visit_counts.get(key, 0) + 1
        self.visit_counts[key] = count
        return count

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.episode_extrinsic_return = 0.0
        self.episode_intrinsic_return = 0.0
        self._count_observation(observation)
        info = dict(info)
        info["intrinsic_reward"] = 0.0
        return observation, info

    def step(self, action):
        observation, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        visit_count = self._count_observation(observation)
        intrinsic_reward = self.bonus_scale / math.sqrt(visit_count)
        total_reward = float(extrinsic_reward) + intrinsic_reward

        self.episode_extrinsic_return += float(extrinsic_reward)
        self.episode_intrinsic_return += intrinsic_reward

        info = dict(info)
        info["extrinsic_reward"] = float(extrinsic_reward)
        info["intrinsic_reward"] = intrinsic_reward
        info["episode_extrinsic_return"] = self.episode_extrinsic_return
        info["episode_intrinsic_return"] = self.episode_intrinsic_return
        info["episode_total_return"] = (
            self.episode_extrinsic_return + self.episode_intrinsic_return
        )
        info["unique_observations_seen"] = len(self.visit_counts)

        return observation, total_reward, terminated, truncated, info
