import gymnasium as gym
import numpy as np

class CuriosityWrapper(gym.Wrapper):
    """
    Adds a count-based intrinsic curiosity reward.
    reward = extrinsic_reward + (beta * (1 / sqrt(visit_count)))
    """
    def __init__(self, env, beta=0.1):
        super().__init__(env)
        self.beta = beta
        self.visits = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_visits(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        intrinsic_reward = self._update_visits(obs)
        
        # Total reward is only modified if beta > 0 (i.e., during training)
        total_reward = reward + (self.beta * intrinsic_reward)
        
        # Store the original reward in info for logging
        info["extrinsic_reward"] = reward
        return obs, total_reward, terminated, truncated, info

    def action_masks(self):
        """Pass-through to the underlying environment."""
        return self.env.action_masks()

    def _update_visits(self, obs):
        """Hash the 64x64 grid to count visits."""
        # obs is (16, 64, 64) one-hot. Convert back to (64,64) integer grid
        grid = np.argmax(obs, axis=0) 
        
        # Create a hashable key from the grid bytes
        state_hash = hash(grid.tobytes())
        
        # Increment visit count
        self.visits[state_hash] = self.visits.get(state_hash, 0) + 1
        
        # Novelty bonus: 1 / sqrt(N)
        return 1.0 / np.sqrt(self.visits[state_hash])