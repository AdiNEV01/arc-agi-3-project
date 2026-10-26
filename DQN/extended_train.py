"""Controlled multi-game DQN with complete training-state continuation.

The engine has no public snapshot API. Restore replays the exact interleaved
action/reset transcript, checking every observable response before restoring RNGs.
No hidden game object or game source is inspected/serialized.
"""
import argparse
from collections import Counter, deque
import csv
import gzip
import hashlib
import json
from importlib import metadata as package_metadata
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import arc_agi
import numpy as np
import torch
from arcengine import GameState

from DQN.action_codec import decode_action
from DQN.dqn_agent import DQNAgent
from DQN.dqn_improvement_1 import StateNoveltyReward
from DQN.dqn_improvement_2 import FrameHistory
from DQN.replay_buffer import Transition
from DQN.shared_environment import SharedDQNEnvironment
from DQN.timing import StageTimer


def digest(obs):
    h = hashlib.sha256(np.asarray(obs.frame[-1], dtype=np.uint8).tobytes())
    h.update(json.dumps([obs.state.value, obs.levels_completed, obs.win_levels,
                        obs.info["available_actions"], obs.info["full_reset"]]).encode())
    return h.hexdigest()


def source_hashes():
    return {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("DQN", "arc_env") for p in sorted(Path(directory).glob("*.py"))}


def runtime_hashes(games):
    result = {}
    for game in games:
        short, version = game.split("-", 1)
        for p in sorted((Path("environment_files") / short / version).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".json"):
                result[p.as_posix()] = hash_file(p)
    return result


def json_write(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def replay_save(replay, path):
    ts = replay.storage
    np.savez_compressed(path, states=np.stack([t.state for t in ts]),
        next_states=np.stack([t.next_state for t in ts]),
        actions=np.array([t.action for t in ts], dtype=np.int64),
        rewards=np.array([t.reward for t in ts], dtype=np.float64),
        terminals=np.array([t.terminated for t in ts], dtype=bool),
        masks=np.stack([t.next_valid_mask for t in ts]))


def replay_load(replay, path, position):
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    replay.storage = [Transition(data["states"][i].copy(), int(data["actions"][i]),
        float(data["rewards"][i]), data["next_states"][i].copy(), bool(data["terminals"][i]),
        data["masks"][i].copy()) for i in range(len(data["actions"]))]
    replay.position = position


def publish(run_dir, persistent):
    """Copy a completed generation last; remote COMPLETE is the commit marker."""
    if persistent is None:
        return
    dst = Path(persistent) / run_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(run_dir.rglob("*"), key=lambda p: p.name == "COMPLETE.json"):
        if not p.is_file() or p.name.endswith(".partial"):
            continue
        target = dst / p.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.name == "COMPLETE.json":
            record = json.loads(p.read_text())
            for name, expected in record["sha256"].items():
                if hash_file(target.parent / name) != expected:
                    raise IOError(f"Cannot commit an unverified Drive checkpoint: {target.parent / name}")
        if (target.exists() and target.stat().st_size == p.stat().st_size
                and hash_file(target) == hash_file(p)):
            continue
        tmp = target.with_suffix(target.suffix + ".partial")
        shutil.copy2(p, tmp)
        if hash_file(tmp) != hash_file(p):
            raise IOError(f"Persistent copy checksum mismatch: {target}")
        tmp.replace(target)


def prune_resume(root, keep=2):
    """Only our new run's completed resume generations, after persistence succeeds."""
    if not root.exists():
        return
    old = sorted(p.parent for p in root.glob("step*/COMPLETE.json"))[:-keep]
    for p in old:
        if p.resolve().parent != root.resolve():
            raise RuntimeError("Unsafe resume retention path")
        shutil.rmtree(p)


def build_contexts(config, seed, frames, beta, arcade):
    contexts = []
    for game in config["training_games"]:
        env = SharedDQNEnvironment(game, arcade, seed, 0, "level")
        obs = env.reset()
        history = FrameHistory(frames)
        history.reset(obs)
        novelty = StateNoveltyReward(beta) if beta else None
        if novelty:
            novelty.register_reset(obs)
        contexts.append(dict(env=env, obs=obs, history=history, novelty=novelty,
                             actions=0, task_reward=0.0, intrinsic_reward=0.0,
                             wins=0, gameovers=0, episodes=1, action_counts=Counter()))
    return contexts


def check_resume(path):
    record = json.loads((path / "COMPLETE.json").read_text())
    for name, expected in record["sha256"].items():
        if hash_file(path / name) != expected:
            raise ValueError(f"Incomplete or corrupt continuation checkpoint: {path / name}")
    return torch.load(path / "state.pt", map_location="cpu", weights_only=True)


def train(config, variant, seed, device, run_dir, stop_at=None, resume=None,
          persistent=None, arcade_factory=None, profile=False):
    cfg = json.loads(json.dumps(config))
    total = cfg["total_actions"]
    stop_at = total if stop_at is None else stop_at
    if variant not in cfg["variants"] or not 0 < stop_at <= total:
        raise ValueError("Invalid variant or stop budget")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if os.getenv("ONLY_RESET_LEVELS", "").lower() == "true":
        raise RuntimeError("ONLY_RESET_LEVELS=true violates the initial reset contract")
    torch.set_num_threads(cfg["torch_threads"])
    torch.use_deterministic_algorithms(cfg["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    frames = 4 if variant in ("v2", "v2_history_only") else 1
    beta = cfg["novelty_beta"] if variant in ("v1", "v2") else 0.0
    hashes = source_hashes()
    assets = runtime_hashes(cfg["training_games"])
    software = {"python":platform.python_version(), "numpy":np.__version__, "torch":str(torch.__version__),
        **{name:package_metadata.version(name) for name in ("arc-agi","arcengine","gymnasium")}}
    saved = check_resume(Path(resume)) if resume else None
    if saved and (saved["protocol"] != cfg or saved["variant"] != variant or saved["seed"] != seed
                  or saved["source_sha256"] != hashes):
        raise ValueError("Resume protocol/variant/seed/source differs; refusing mixed experiments")
    if saved and saved["runtime_assets"] != assets:
        raise ValueError("Game runtime assets differ; refusing continuation")
    if saved and (saved["torch_version"] != str(torch.__version__) or saved["device_type"] != device.type):
        raise ValueError("Exact continuation requires same torch version and device type")
    if saved and saved["software"] != software:
        raise ValueError("Exact continuation requires the same recorded Python and package versions")
    start_step = saved["global_step"] if saved else 0
    if stop_at <= start_step:
        raise ValueError("Stop budget must exceed saved step")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    for d in ("frozen", "resume"):
        (run_dir / d).mkdir()
    metadata = dict(protocol=cfg, variant=variant, seed=seed, num_frames=frames,
        novelty_beta=beta, device=str(device), torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, source_sha256=hashes, platform=platform.platform(),
        runtime_assets=assets,
        software=software,
        parent_resume=str(resume) if resume else None, initial_step=start_step,
        checkpoint_kind="complete learner state plus verified environment transcript reconstruction")
    json_write(run_dir / "metadata.json", metadata)
    arcade = arcade_factory() if arcade_factory else arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    contexts = build_contexts(cfg, seed, frames, beta, arcade)
    journal = []
    initial_digests = [digest(c["obs"]) for c in contexts]
    if saved:
        if initial_digests != saved["initial_digests"]:
            raise ValueError("Initial game observations differ on reconstruction")
        for gi, action_index, expected in saved["journal"]:
            c = contexts[gi]
            if action_index == -1:
                obs = c["env"].reset()
                c["history"].reset(obs)
            else:
                action, data = decode_action(action_index)
                obs = c["env"].step(action, data)
                c["history"].advance(obs)
            if digest(obs) != expected:
                raise ValueError("Environment reconstruction diverged; continuation rejected")
            c["obs"] = obs
        journal = saved["journal"]
        for c, state in zip(contexts, saved["contexts"]):
            if not np.array_equal(c["history"].state(), state["history"].numpy()):
                raise ValueError("Reconstructed frame history differs")
            for key in ("actions", "task_reward", "intrinsic_reward", "wins", "gameovers", "episodes"):
                c[key] = state[key]
            c["action_counts"] = Counter(state["action_counts"])
            if c["novelty"]:
                c["novelty"].visits = Counter({bytes.fromhex(k): v for k, v in state["visits"].items()})
    agent = DQNAgent(seed=seed, lr=cfg["lr"], gamma=cfg["gamma"], batch_size=cfg["batch_size"],
        buffer_capacity=cfg["buffer_capacity"], target_update_interval=cfg["target_update_interval"],
        device=device, num_frames=frames)
    timer = StageTimer(device, profile)
    agent.profiler = timer
    if saved:
        agent.online.load_state_dict(saved["online_state_dict"])
        agent.target.load_state_dict(saved["target_state_dict"])
        agent.optimizer.load_state_dict(saved["optimizer_state_dict"])
        agent.gradient_steps = saved["gradient_steps"]
        replay_load(agent.replay, Path(resume) / "replay.npz", saved["replay_position"])
        agent.rng.setstate(saved["agent_rng"])
        random.setstate(saved["python_rng"])
        nr = saved["numpy_rng"]
        np.random.set_state((nr[0], nr[1].numpy().astype(np.uint32), nr[2], nr[3], nr[4]))
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    setup_done = time.perf_counter()
    started = setup_done
    prior_seconds = saved["training_seconds"] if saved else 0.0
    checkpoint_seconds = 0.0
    losses = []

    def snapshot(step, frozen):
        nonlocal checkpoint_seconds
        timer.sync()
        t0 = time.perf_counter()
        model = dict(online_state_dict=agent.online.state_dict(), target_state_dict=agent.target.state_dict(),
            optimizer_state_dict=agent.optimizer.state_dict(), seed=seed, global_step=step,
            gradient_steps=agent.gradient_steps, config={**cfg, "num_frames": frames, "variant": variant,
            "novelty_beta": beta, "device": str(device)}, training_backend="shared",
            training_seconds=prior_seconds + t0 - started,
            checkpoint_kind="frozen evaluation snapshot; full continuation is in resume generation")
        if not all(torch.isfinite(p).all().item() for net in (agent.online, agent.target) for p in net.parameters()):
            raise FloatingPointError("Nonfinite model parameters")
        if frozen:
            tmp = run_dir / "frozen" / f"step{step:06d}.pt.partial"
            torch.save(model, tmp)
            tmp.replace(tmp.with_suffix(""))
        dest = run_dir / "resume" / f"step{step:06d}"
        dest.mkdir(exist_ok=False)
        nr = np.random.get_state()
        state = {**model, **metadata, "global_step": step, "protocol": cfg,
            "replay_position": agent.replay.position, "agent_rng": agent.rng.getstate(),
            "python_rng": random.getstate(), "numpy_rng": (nr[0], torch.from_numpy(nr[1].copy().astype(np.int64)), nr[2], nr[3], nr[4]),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "device_type": device.type, "journal": journal, "initial_digests": initial_digests,
            "contexts": [{**{k: c[k] for k in ("actions", "task_reward", "intrinsic_reward", "wins", "gameovers", "episodes")},
                "action_counts": dict(c["action_counts"]), "history": torch.from_numpy(c["history"].state()),
                "visits": {k.hex(): v for k, v in c["novelty"].visits.items()} if c["novelty"] else {}}
                for c in contexts]}
        torch.save(state, dest / "state.pt")
        replay_save(agent.replay, dest / "replay.npz")
        with (run_dir / "actions.csv").open("rb") as src, gzip.open(dest / "history.csv.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        json_write(dest / "COMPLETE.json", {"step": step, "sha256": {n: hash_file(dest / n) for n in ("state.pt", "replay.npz", "history.csv.gz")}})
        publish(run_dir, persistent)
        prune_resume(run_dir / "resume")
        if persistent:
            prune_resume(Path(persistent) / run_dir.name / "resume")
        checkpoint_seconds += time.perf_counter() - t0

    columns = ["global_step", "game", "game_actions", "action_index", "action", "x", "y", "epsilon",
               "task_reward", "intrinsic_reward", "loss", "levels_completed", "state", "terminated",
               "truncated", "controller_resets", "elapsed_seconds"]
    if saved:
        # Canonical committed prefix, not the possibly unfinished tail in an old log.
        with gzip.open(Path(resume) / "history.csv.gz", "rb") as src, (run_dir / "actions.csv").open("xb") as dst:
            shutil.copyfileobj(src, dst)
    with (run_dir / "actions.csv").open("a" if saved else "x", newline="") as f, (run_dir / "gpu_samples.csv").open("x") as gpu_log:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not saved:
            writer.writeheader()
        gpu_log.write("global_step,utilization_percent,memory_used_MiB,power_W\n")
        for step in range(start_step + 1, stop_at + 1):
            gi = (step - 1) % len(contexts)
            c = contexts[gi]
            obs = c["obs"]
            epsilon = cfg["epsilon_start"] + min(1.0, (step - 1) / cfg["epsilon_decay"]) * (cfg["epsilon_end"] - cfg["epsilon_start"])
            with timer.measure("action_selection"):
                index = agent.select_action(c["history"].state(), c["env"].action_space, epsilon)
            action, data = decode_action(index)
            with timer.measure("environment_step"):
                nxt = c["env"].step(action, data)
            if nxt.truncated:
                raise RuntimeError("Unexpected artificial truncation")
            journal.append((gi, index, digest(nxt)))
            intrinsic = c["novelty"].bonus(obs, nxt) if c["novelty"] else 0.0
            current = c["history"].state()
            following = c["history"].advance(nxt)
            agent.remember(current, index, nxt.task_reward + intrinsic, following,
                           nxt.terminated, c["env"].action_space)
            loss = None
            if step >= cfg["warmup"] and step % cfg["train_every"] == 0:
                with timer.measure("gradient_update"):
                    loss = agent.learn()
                if loss is None or not np.isfinite(loss):
                    raise FloatingPointError(f"Missing/nonfinite loss at {step}")
                losses.append(loss)
            c["actions"] += 1
            c["task_reward"] += nxt.task_reward
            c["intrinsic_reward"] += intrinsic
            c["action_counts"][index] += 1
            c["wins"] += int(nxt.state == GameState.WIN)
            c["gameovers"] += int(nxt.state == GameState.GAME_OVER)
            writer.writerow(dict(global_step=step, game=cfg["training_games"][gi], game_actions=c["actions"],
                action_index=index, action=action.name, x=data["x"] if data else "", y=data["y"] if data else "",
                epsilon=epsilon, task_reward=nxt.task_reward, intrinsic_reward=intrinsic, loss=loss if loss is not None else "",
                levels_completed=nxt.levels_completed, state=nxt.state.name, terminated=nxt.terminated, truncated=False,
                controller_resets=c["env"].shared.resets_sent, elapsed_seconds=prior_seconds+time.perf_counter()-started))
            c["obs"] = nxt
            if nxt.terminated:
                c["obs"] = c["env"].reset()
                if nxt.state == GameState.GAME_OVER and c["obs"].info["full_reset"]:
                    raise RuntimeError("Unexpected full reset after GAME_OVER")
                c["history"].reset(c["obs"])
                if c["novelty"]:
                    c["novelty"].register_reset(c["obs"])
                c["episodes"] += 1
                journal.append((gi, -1, digest(c["obs"])))
            if step % cfg["log_every"] == 0 or step == stop_at:
                timer.sync()
                print(f"{variant} seed{seed} {step}/{total} epsilon={epsilon:.4f} loss={loss} task={sum(x['task_reward'] for x in contexts)}", flush=True)
                f.flush()
                if device.type == "cuda":
                    sample = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                    gpu_log.write(f"{step},{sample.stdout.strip()}\n")
                    gpu_log.flush()
            if step in cfg["checkpoint_actions"] or step % cfg["resume_every"] == 0 or step == stop_at:
                f.flush()
                snapshot(step, step in cfg["checkpoint_actions"] or step == stop_at)
    timer.sync()
    seconds = time.perf_counter() - started
    report = {**metadata, "global_step": stop_at, "new_actions": stop_at-start_step,
        "gradient_steps": agent.gradient_steps, "loop_seconds_including_persistence": seconds,
        "checkpoint_and_persistence_seconds": checkpoint_seconds,
        "actions_per_second_including_persistence": (stop_at-start_step)/seconds,
        "compute_actions_per_second": (stop_at-start_step)/max(1e-9, seconds-checkpoint_seconds),
        "finite_losses": all(np.isfinite(losses)), "new_updates": len(losses),
        "per_game": [{"game": g, **{k: c[k] for k in ("actions", "task_reward", "intrinsic_reward", "wins", "gameovers", "episodes")},
            "resets": c["env"].shared.resets_sent, "final_levels": c["obs"].levels_completed,
            "action_counts": dict(c["action_counts"])} for g,c in zip(cfg["training_games"], contexts)],
        "stages_seconds": dict(timer.seconds), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
    json_write(run_dir / "summary.json", report)
    publish(run_dir, persistent)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Configs/dqn_extended_v1.json")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--stop-at", type=int)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--persistent-root", type=Path)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    train(json.loads(Path(args.config).read_text()), args.variant, args.seed, args.device,
          args.run_dir, args.stop_at, args.resume, args.persistent_root, profile=args.profile)


if __name__ == "__main__":
    main()
