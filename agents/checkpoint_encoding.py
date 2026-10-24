"""Grid history for frozen legacy checkpoints; padding has no colour channel."""
from collections import deque

import numpy as np


class GridHistory:
    def __init__(self, num_frames):
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def clear(self):
        self.frames.clear()

    def push(self, obs):
        grid = np.asarray(obs)
        if grid.shape != (64, 64) or not np.issubdtype(grid.dtype, np.integer):
            raise ValueError("Expected integer 64x64 shared grid")
        if grid.min() < 0 or grid.max() > 16:
            raise ValueError("Expected colours 0-15 or padding 16")
        if not self.frames:
            self.frames.extend(grid.copy() for _ in range(self.num_frames))
        else:
            self.frames.append(grid.copy())
        encoded = np.eye(17, dtype=np.float32)[np.stack(self.frames)][..., :16]
        return encoded.transpose(0, 3, 1, 2).reshape(self.num_frames * 16, 64, 64)
