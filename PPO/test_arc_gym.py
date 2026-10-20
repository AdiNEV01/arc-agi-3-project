
from arc_gym import ArcEnv

env = ArcEnv()

obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Observation type:", obs.dtype)
print("Initial info:", info)
print("Action space:", env.action_space)

action = 0

obs, reward, terminated, truncated, info = env.step(action)

print("\nAFTER ONE ACTION")
print("Observation shape:", obs.shape)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)

env.close()
