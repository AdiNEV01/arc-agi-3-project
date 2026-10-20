"""Step 1-3 of the PPO to-do list: record versions, inspect the raw ARC-AGI-3
interface, and run a minimal end-to-end agent.

Run:  python test_env.py [game_id]
Paste the printed output back to whoever is helping you -- the wrapper in
PPO/arc_gym.py makes a few assumptions about the observation object that this
script lets you verify.
"""
import json
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

GAME = sys.argv[1] if len(sys.argv) > 1 else "ls20"


def record_versions():
    import platform
    import torch

    pkgs = ["arc-agi", "arcengine", "gymnasium", "stable-baselines3",
            "sb3-contrib", "torch", "numpy"]
    info = {"python": platform.python_version(), "os": platform.platform()}
    for p in pkgs:
        try:
            info[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info[p] = "NOT INSTALLED"
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    Path("Configs").mkdir(exist_ok=True)
    Path("Configs/versions.json").write_text(json.dumps(info, indent=2))
    print("== versions (saved to Configs/versions.json) ==")
    print(json.dumps(info, indent=2))


def inspect_raw():
    import arc_agi
    from arcengine import GameAction

    arc = arc_agi.Arcade()
    try:
        envs = arc.get_environments()
        print("\n== available environments ==")
        for e in envs:
            print(" ", getattr(e, "game_id", e))
    except Exception as exc:  # listing is a nice-to-have
        print("could not list environments:", exc)

    env = arc.make(GAME)
    print("\n== action_space ==\n", env.action_space)

    obs = env.step(GameAction.RESET)
    print("\n== raw observation after RESET ==")
    print("type:", type(obs))
    print("attributes:", [a for a in dir(obs) if not a.startswith("_")])
    frame = np.asarray(obs.frame)
    print("frame shape:", frame.shape, "dtype:", frame.dtype,
          "min/max:", frame.min(), frame.max())
    for name in ("state", "levels_completed", "available_actions", "win_levels"):
        print(f"{name}:", getattr(obs, name, "<missing>"))

    print("\n== 20 random steps ==")
    rng = np.random.default_rng(0)
    simple = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
              GameAction.ACTION4, GameAction.ACTION5]
    for t in range(20):
        a = simple[rng.integers(len(simple))]
        obs = env.step(a)
        print(t, a, "state=", getattr(obs, "state", None),
              "levels=", getattr(obs, "levels_completed", None))
    print("\nscorecard:", arc.get_scorecard())


def test_wrapper():
    sys.path.insert(0, str(Path(__file__).parent / "PPO"))
    from PPO.arc_gym import ArcGymEnv
    from stable_baselines3.common.env_checker import check_env

    env = ArcGymEnv(GAME, max_steps=50)
    obs, info = env.reset(seed=0)
    print("\n== wrapper ==")
    print("obs shape:", obs.shape, obs.dtype, "action space:", env.action_space)
    ret, steps = 0.0, 0
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        ret += r
        steps += 1
        done = term or trunc
    print(f"episode finished: steps={steps} return={ret} info={info}")
    print("action mask sum:", env.action_masks().sum(), "/", env.action_space.n)
    check_env(env, warn=True)
    print("check_env passed")


if __name__ == "__main__":
    record_versions()
    inspect_raw()
    test_wrapper()