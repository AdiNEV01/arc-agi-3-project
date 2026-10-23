import numpy as np
from collections import deque
from gymnasium import spaces
from arc_env import ARCEnv

class ARCEnvMemory(ARCEnv):
    def __init__(self, game_id: str = "ls20", stack_size: int = 4):
        super().__init__(game_id)
        self.stack_size = stack_size
        # 4 frames * 16 channels = 64 channels
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(16 * stack_size, 64, 64), dtype=np.float32
        )
        self.frames = deque(maxlen=stack_size)

    def _get_stacked_obs(self, obs):
        """Stacks the frames along the channel dimension."""
        if len(self.frames) == 0:
            # Fill deque with the first frame
            for _ in range(self.stack_size):
                self.frames.append(obs)
        else:
            self.frames.append(obs)
            
        # Concatenate along the channel axis (axis 0)
        return np.concatenate(self.frames, axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs) if hasattr(self, 'env') else super().reset(**kwargs)
        single_frame = self._encode_obs(self.current_obs)
        self.frames.clear()
        stacked_obs = self._get_stacked_obs(single_frame)
        return stacked_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # super().step() returns the single frame one-hot encoding, 
        # but we need to extract it from the raw frame data again to stack it.
        single_frame = self._encode_obs(self.current_obs)
        stacked_obs = self._get_stacked_obs(single_frame)
        return stacked_obs, reward, terminated, truncated, info