import os, glob
import numpy as np
import imageio.v2 as imageio
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from arc_env import ARCEnv
from arc_env_memory import ARCEnvMemory

TEST_GAMES = ["dc22", "ft09", "g50t", "ka59"]
os.makedirs("trajectories", exist_ok=True)

def make_env(game, mem):
    def _init():
        e = ARCEnvMemory(game) if mem else ARCEnv(game)
        return ActionMasker(e, lambda e: e.action_masks())
    return _init

def record(model_path, game, mem, tag):
    env = DummyVecEnv([make_env(game, mem)])
    model = MaskablePPO.load(model_path, env=env)
    obs = env.reset()
    frames, done, steps = [], False, 0
    while not done and steps < 1000:
        # grab the most recent 16-channel frame for display
        frame = obs[0][-16:].argmax(axis=0).astype(np.uint8) * 16
        frames.append(np.kron(frame, np.ones((4, 4))))  # upscale 4x
        a, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(a)
        done = dones[0]; steps += 1
    path = f"trajectories/{tag}_{game}.gif"
    imageio.mimsave(path, frames, fps=8)
    print(f"Wrote {path} ({steps} steps)")

# Success: memory_curiosity seed 0
record("./logs/memory_curiosity_seed0/final_model.zip", "dc22", True, "success")
# Failure: baseline seed 0
record("./logs/baseline_seed0/final_model.zip", "dc22", False, "failure")