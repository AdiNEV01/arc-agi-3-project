import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from arc_env_memory import ARCEnvMemory  # <--- IMPORT NEW MEMORY ENV
from curiosity_wrapper import CuriosityWrapper

# --- Config ---
TRAIN_GAMES = ["ls20", "ar25", "bp35"] 
VAL_GAMES = ["cd82"]                   
SEED = 0
TOTAL_STEPS = 10_000  
LOG_DIR = "./logs/improvement2_memory_curiosity"
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(game_id, is_training=True):
    def _init():
        env = ARCEnvMemory(game_id=game_id) # <--- USE MEMORY ENV
        beta = 0.1 if is_training else 0.0
        env = CuriosityWrapper(env, beta=beta)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init

train_env = DummyVecEnv([make_env(g, is_training=True) for g in TRAIN_GAMES])
train_env = VecMonitor(train_env)

val_env = DummyVecEnv([make_env(g, is_training=False) for g in VAL_GAMES])
val_env = VecMonitor(val_env)

eval_callback = MaskableEvalCallback(
    val_env,
    best_model_save_path=os.path.join(LOG_DIR, "best_model"),
    log_path=LOG_DIR,
    eval_freq=10_000,
    deterministic=True,
)

model = MaskablePPO(
    "CnnPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
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

model.save(os.path.join(LOG_DIR, f"ppo_memory_seed{SEED}"))
print("Improvement 2 training complete.")

mean_reward, std_reward = evaluate_policy(model, val_env, n_eval_episodes=10, deterministic=True)
print(f"Validation (Extrinsic Reward Only): mean = {mean_reward:.3f} ± {std_reward:.3f}")