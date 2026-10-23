import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from arc_env import ARCEnv

# Config

# Pick a few valid games from the list for your initial test
TRAIN_GAMES = ["ls20", "ar25", "bp35"] 
VAL_GAMES = ["cd82"]                   
SEED = 1
TOTAL_STEPS = 10_000  # Keep it small for the very first test
LOG_DIR = "./logs/baseline"

os.makedirs(LOG_DIR, exist_ok=True)

# Vectorized environment
def make_env(game_id):
    def _init():
        env = ARCEnv(game_id=game_id)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init

train_env = DummyVecEnv([make_env(g) for g in TRAIN_GAMES])
train_env = VecMonitor(train_env)

val_env = DummyVecEnv([make_env(g) for g in VAL_GAMES])
val_env = VecMonitor(val_env)

# MaskableEvalCallback (required for masking)
eval_callback = MaskableEvalCallback(
    val_env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

# Baseline MaskablePPO
model = MaskablePPO(
    "CnnPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,      # standard exploration entropy
    seed=SEED,
    verbose=1,
    tensorboard_log=LOG_DIR,
    policy_kwargs=dict(normalize_images=False)
)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=eval_callback,
    log_interval=1,
)

model.save(os.path.join(LOG_DIR, f"ppo_baseline_seed{SEED}"))
print("Baseline training complete.")

# Final evaluation on validation set
mean_reward, std_reward = evaluate_policy(
    model, val_env, n_eval_episodes=10, deterministic=True
)

print(f"Validation: mean reward = {mean_reward:.3f} ± {std_reward:.3f}")