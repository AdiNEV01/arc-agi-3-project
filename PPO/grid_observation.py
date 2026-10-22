"""Convert an ARC board into separate channels for each colour.

The normal ArcEnv observation has shape (1, 64, 64). In that observation,
colour IDs are scaled by 16 (for example, colour 3 is stored as 48).

This wrapper converts the values back to colour IDs, then creates 16 channels:
channel 0 marks colour 0, channel 1 marks colour 1, and so on.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class OneHotGridObservation(gym.ObservationWrapper):
    """Represent a board using 16 colour channels with values 0.0 or 1.0."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(16, 64, 64),
            dtype=np.float32,
        )

    def observation(self, observation):
        # The normal observation has one channel, so remove that first channel.
        scaled_grid = observation[0].astype(np.int16)

        # ArcEnv.py scales colour IDs by 16. Check that this is still true.
        if np.any(scaled_grid % 16 != 0):
            raise ValueError(
                "The grid contains values that are not multiples of 16. "
                "Check the colour conversion in arc_gym.py."
            )

        # Convert values such as 48 back to colour ID 3.
        colours = scaled_grid // 16

        if np.any(colours < 0) or np.any(colours > 15):
            raise ValueError("Expected ARC colour IDs between 0 and 15.")

        # Create 16 empty channels, then mark cells of each colour.
        encoded_grid = np.zeros((16, 64, 64), dtype=np.float32)
        for colour_id in range(16):
            encoded_grid[colour_id] = (colours == colour_id).astype(np.float32)

        return encoded_grid


def grid_observation_to_board(observation):
    """Convert a 16-channel grid back to one board for drawing a replay."""
    colour_ids = np.argmax(observation, axis=0)
    return (colour_ids * 16).astype(np.uint8)
