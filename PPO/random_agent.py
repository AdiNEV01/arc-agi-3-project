"""Step 1: play a game with random moves. No PPO, no Gym - just the ARC toolkit."""
import random

import arc_agi
from arcengine import GameAction

GAME = "ls20"                      # the game we are playing
N_STEPS = 200                      # how many actions to take

arc = arc_agi.Arcade()             # connects to the ARC toolkit
env = arc.make(GAME)               # creates the game

# The simple "keyboard" actions the agent may choose from
actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
           GameAction.ACTION4, GameAction.ACTION5]

obs = env.step(GameAction.RESET)   # start the game
print("start:", obs.state, "levels completed:", obs.levels_completed)

for step in range(N_STEPS):
    action = random.choice(actions)    # pick a random action
    obs = env.step(action)             # do it, get the new game state back
    print(step, action, obs.state, obs.levels_completed)

print(arc.get_scorecard())             # the (local) score summary