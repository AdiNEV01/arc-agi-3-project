import os, csv, glob
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from arc_env import ARCEnv
from arc_env_memory import ARCEnvMemory

TEST_GAMES = ["dc22", "ft09", "g50t", "ka59"] 
EXPERIMENTS = ["baseline", "curiosity", "memory_only", "memory_curiosity"]
SEEDS = [0, 1, 2]
MAX_STEPS = 1000

def make_eval_env(game_id, use_memory):
    def _init():
        env = ARCEnvMemory(game_id) if use_memory else ARCEnv(game_id)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init

def run_episode(model, env, max_steps=MAX_STEPS):
    obs = env.reset()
    done = False
    steps, levels = 0, 0
    failure_mode = "timeout"
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        steps += 1
        done = dones[0]
        if infos[0].get("levels_completed", 0) > levels:
            levels = infos[0]["levels_completed"]
        if done:
            state = str(infos[0].get("state", ""))
            failure_mode = "win" if "WIN" in state else "game_over"
    return levels, steps, failure_mode

rows = []
for exp in EXPERIMENTS:
    use_memory = exp in ["memory_only", "memory_curiosity"]
    for seed in SEEDS:
        ckpt_dir = f"./logs/{exp}_seed{seed}/checkpoints"
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.zip")))
        if not ckpts:
            continue
        for ckpt in ckpts:
            # parse timestep from filename like ppo_25000_steps.zip
            ts = int(os.path.basename(ckpt).split("_")[1])
            for game in TEST_GAMES:
                env = DummyVecEnv([make_eval_env(game, use_memory)])
                model = MaskablePPO.load(ckpt, env=env)
                levels, steps, fail = run_episode(model, env)
                rhae = min(1.0, (100 / steps) ** 2) if levels > 0 else 0.0
                rows.append({
                    "experiment": exp, "seed": seed, "timestep": ts,
                    "game": game, "levels": levels, "steps": steps,
                    "rhae": rhae, "failure_mode": fail,
                })

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("evaluation_results.csv", index=False)
print(f"Wrote {len(df)} rows to evaluation_results.csv")