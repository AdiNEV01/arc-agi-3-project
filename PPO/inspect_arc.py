
import arc_agi
import numpy as np
from arcengine import GameState

# 1. Create the game
arcade = arc_agi.Arcade()
env = arcade.make("ls20", seed=0)

# 2. Reset the game
obs = env.reset()

print("GAME STARTED")
print("State:", obs.state)
print("Levels completed:", obs.levels_completed)

# 3. Inspect the observation
print("\nOBSERVATION")
print("Observation type:", type(obs))
print("Frame type:", type(obs.frame))
print("Number of frame elements:", len(obs.frame))

# Convert the frame into a NumPy array
frame = np.array(obs.frame)

print("Frame array shape:", frame.shape)
print("Frame array data type:", frame.dtype)
print("Minimum frame value:", frame.min())
print("Maximum frame value:", frame.max())

# Print the frame itself
print("\nFRAME CONTENTS")
print(frame)

# 4. List the available actions
print("\nAVAILABLE ACTIONS")
actions = list(env.action_space)

for action in actions:
    print(action.name)

# 5. Execute one valid action
if actions:
    action = actions[0]

    print("\nExecuting action:", action.name)
    obs = env.step(action)

    # 6. Inspect the result
    print("\nAFTER THE ACTION")
    print("State:", obs.state)
    print("Levels completed:", obs.levels_completed)
    print("Frame type:", type(obs.frame))
    print("Available actions:", [a.name for a in env.action_space])

    if obs.state == GameState.WIN:
        print("Game won")
    elif obs.state == GameState.GAME_OVER:
        print("Game over")
    else:
        print("Game continues")


print("\nTESTING RESET")

obs = env.reset()

print("State after reset:", obs.state)
print("Levels after reset:", obs.levels_completed)
print("Available actions after reset:",
      [action.name for action in env.action_space])
print("Frame shape after reset:", np.array(obs.frame).shape)


print("\nTESTING PROGRESSION")

for step in range(10):
    actions = list(env.action_space)

    if not actions:
        print("No actions available.")
        break

    action = actions[0]
    obs = env.step(action)

    print(
        "Step:", step + 1,
        "| Action:", action.name,
        "| State:", obs.state.name,
        "| Levels:", obs.levels_completed
    )

    if obs.state in (GameState.WIN, GameState.GAME_OVER):
        print("Episode ended.")
        break


