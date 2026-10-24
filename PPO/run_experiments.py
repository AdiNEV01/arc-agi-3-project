import argparse
import os
import numpy as np
import time, json, platform, torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from arc_env import ARCEnv
from arc_env_memory import ARCEnvMemory
from curiosity_wrapper import CuriosityWrapper

def make_env(game_id, use_curiosity, use_memory, is_training=True):
    def _init():
        if use_memory:
            env = ARCEnvMemory(game_id=game_id)
        else:
            env = ARCEnv(game_id=game_id)
        if use_curiosity:
            beta = 0.1 if is_training else 0.0
            env = CuriosityWrapper(env, beta=beta)
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["baseline", "curiosity", "memory_only", "memory_curiosity"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500_000)
    args = parser.parse_args()

    TRAIN_GAMES = ["ls20", "ar25", "bp35"]
    VAL_GAMES = ["cd82"]
    
    use_curiosity = args.exp in ["curiosity", "memory_curiosity"]
    use_memory = args.exp in ["memory_only", "memory_curiosity"]
    
    LOG_DIR = f"./logs/{args.exp}_seed{args.seed}"
    os.makedirs(LOG_DIR, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix="ppo"
    )

    train_env = DummyVecEnv([make_env(g, use_curiosity, use_memory, True) for g in TRAIN_GAMES])
    train_env = VecMonitor(train_env)

    val_env = DummyVecEnv([make_env(g, False, use_memory, False) for g in VAL_GAMES])
    val_env = VecMonitor(val_env)

    eval_cb = MaskableEvalCallback(
        val_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=LOG_DIR,
        eval_freq=25_000, 
        deterministic=True,
    )

    model = MaskablePPO(
        "CnnPolicy", train_env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        seed=args.seed, verbose=1, tensorboard_log=LOG_DIR,
        policy_kwargs=dict(normalize_images=False)
    )

    print(f"Starting {args.exp} with seed {args.seed} for {args.steps} steps...")
    
    t0 = time.time()
    # SINGLE LEARN CALL
    model.learn(total_timesteps=args.steps, callback=[eval_cb, checkpoint_cb])
    elapsed = time.time() - t0
    
    model.save(os.path.join(LOG_DIR, f"final_model"))

    resources = {
        "experiment": args.exp,
        "seed": args.seed,
        "steps": args.steps,
        "wall_clock_sec": elapsed,
        "fps": args.steps / elapsed,
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0),
    }
    with open(os.path.join(LOG_DIR, "resources.json"), "w") as f:
        json.dump(resources, f, indent=2)

    print(f"Finished {args.exp} seed {args.seed}.")