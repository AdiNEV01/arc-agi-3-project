"""V2: decision-time frame history; novelty still uses the latest grid."""
from collections import deque
import numpy as np

class FrameHistory:
    def __init__(self, num_frames=4):
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, obs):
        self.frames.clear()
        frame = np.asarray(obs.frame[-1], dtype=np.uint8)
        self.frames.extend(frame.copy() for _ in range(self.num_frames))
        return self.state()

    def advance(self, obs):
        if not self.frames:
            raise RuntimeError("Reset history before advancing")
        self.frames.append(np.asarray(obs.frame[-1], dtype=np.uint8).copy())
        return self.state()

    def state(self):
        return np.stack(self.frames).copy()
