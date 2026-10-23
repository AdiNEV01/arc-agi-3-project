import numpy as np
from arc_env import ARCEnv
from sb3_contrib.common.maskable.utils import get_action_masks

env = ARCEnv(game_id="ls20")  # validation game
obs, info = env.reset()

print("Raw available_actions from env:", info.get("available_actions"))
print("Action mask:", env.action_masks())
print("--------------------")

done = False
total_reward = 0
steps = 0

while not done:
    masks = get_action_masks(env)
    valid_actions = np.flatnonzero(masks)
    if len(valid_actions) == 0:
        break
    action = np.random.choice(valid_actions)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    done = terminated or truncated

print(f"Random policy: reward={total_reward}, steps={steps}, levels={info.get('levels_completed', 0)}")