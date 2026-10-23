import gymnasium as gym
import numpy as np
from gymnasium import spaces
import arc_agi
from arcengine import GameAction, GameState

class ARCEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, game_id: str = "ls20"):
        super().__init__()
        self.game_id = game_id
        self.arcade = arc_agi.Arcade()
        self.raw_env = self.arcade.make(game_id)
        
        # 16 colors, 64x64 grid -> one-hot encoded image (C, H, W)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(16, 64, 64), dtype=np.float32
        )
        
        # Maximum 7 actions (ACTION1-ACTION7). RESET is handled internally.
        # For the baseline, ACTION6 coordinates are sampled randomly.
        self.action_space = spaces.Discrete(7)
        
        self.current_obs = None
        self._action_map = {
            0: GameAction.ACTION1,
            1: GameAction.ACTION2,
            2: GameAction.ACTION3,
            3: GameAction.ACTION4,
            4: GameAction.ACTION5,
            5: GameAction.ACTION6,  # needs coordinates
            6: GameAction.ACTION7,
        }

    def _encode_obs(self, frame_data) -> np.ndarray:
        """Convert FrameDataRaw grid to one-hot (16,64,64) float32."""
        grid = np.array(frame_data.frame, dtype=np.int32)  # Could be (64,64) or (6,64,64)
        
        # If it's a sequence of frames, grab the most recent one (the last frame)
        if grid.ndim == 3:
            grid = grid[-1]
            
        one_hot = np.zeros((16, 64, 64), dtype=np.float32)
        for c in range(16):
            one_hot[c] = (grid == c).astype(np.float32)
        return one_hot

    def _get_action_mask(self) -> np.ndarray:
        """Boolean mask of available actions from the last response."""
        mask = np.zeros(7, dtype=bool)
        
        # Safely extract integer values from available_actions
        available_ints = set()
        for a in self.current_obs.available_actions:
            if hasattr(a, 'value'):  # If it's an Enum
                available_ints.add(a.value)
            else:                    # If it's already an int
                available_ints.add(int(a))
                
        for i, act in self._action_map.items():
            act_val = act.value if hasattr(act, 'value') else int(act)
            if act_val in available_ints:
                mask[i] = True
                
        return mask

    def action_masks(self) -> np.ndarray:
        """Required by sb3-contrib MaskablePPO."""
        return self._get_action_mask()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_obs = self.raw_env.reset()
        obs = self._encode_obs(self.current_obs)
        
        # Updated info dict to match step()
        info = {
            "available_actions": self.current_obs.available_actions,
            "state": self.current_obs.state,
            "levels_completed": getattr(self.current_obs, "levels_completed", 0)
        }
        return obs, info

    def step(self, action: int):
        game_action = self._action_map[action]
        
        # Handle ACTION6 coordinates (random baseline)
        data = None
        if game_action == GameAction.ACTION6:
            data = {"x": np.random.randint(0, 64), "y": np.random.randint(0, 64)}
        
        self.current_obs = self.raw_env.step(game_action, data=data)
        
        obs = self._encode_obs(self.current_obs)
        
        # Sparse reward: only on WIN (or you can use levels_completed delta)
        reward = 1.0 if self.current_obs.state == GameState.WIN else 0.0
        
        terminated = self.current_obs.state in (GameState.WIN, GameState.GAME_OVER)
        truncated = False  # no time limit in baseline; add if needed
        
        info = {
            "available_actions": self.current_obs.available_actions,
            "state": self.current_obs.state,
            "levels_completed": self.current_obs.levels_completed,
        }
        return obs, reward, terminated, truncated, info