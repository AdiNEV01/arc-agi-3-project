"""Shared evaluation script for ALL agents (random, DQN, PPO and their improvements).

Protocol (keep identical across agents; see docs/interface.md):
* Competition-style resets: the game is started once (a full reset, which opens
  the scorecard play), and afterwards RESET is sent only after GAME_OVER. Since
  GAME_OVER always follows at least one action, every later RESET is a level
  reset, as in Kaggle competition mode. This is VERIFIED per run: the record
  field full_resets_after_start must be 0 and scorecard_runs must be 1.
  (A full reset would open a new scorecard run and the game score would become
  the MAX over runs, overstating local scores relative to Kaggle.)
  Do NOT set ONLY_RESET_LEVELS=true: it turns the first RESET into a level reset,
  the scorecard then never opens a play, and the official score is empty.
* One scorecard and one make() per (agent, seed, game).
* Fixed budget of agent actions per game (ACTION1-7). The official scorecard also
  counts each level RESET as one action; level_actions and scored_actions include
  those resets, official_actions does not.
* Score = official RHAE from the toolkit scorecard (0-100). A recomputed RHAE
  is logged alongside as a cross-check.

Outputs in --out:
  episodes.jsonl   one record per (agent, seed, game)
  summary.csv      same records, flat
  aggregate.json   mean/std across seeds, per game and overall
  run_meta.json    versions, hardware, git commit, config
  traj/*.jsonl     per-step trajectories (with --save-traj), for failure analysis
  recordings/      official toolkit recordings with frames (with --save-recording), for replays

Examples:
  python evaluate.py --agent random --split dev
  python evaluate.py --agent agents.ppo_agent:PPOPolicy \
      --agent-kwargs '{"checkpoint": "results/checkpoints/ppo_base/step_200000.zip"}' \
      --agent-version ppo-base --checkpoint-step 200000 --split heldout --save-traj
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml  # noqa: E402
from arcengine import GameState  # noqa: E402

from arc_env.gym_adapter import ArcGymEnv  # noqa: E402

PACKAGES = ["arc-agi", "arcengine", "gymnasium", "numpy", "torch",
            "stable-baselines3", "sb3-contrib"]


# --------------------------------------------------------------------- metadata
def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def hardware() -> dict[str, Any]:
    hw = {"platform": platform.platform(), "cpu_count": os.cpu_count(),
          "processor": platform.processor()}
    try:
        import torch
        hw["cuda"] = torch.cuda.is_available()
        if hw["cuda"]:
            hw["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        hw["cuda"] = None
    return hw


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------- scoring
def rhae_game_score(level_actions: list[int], baselines: list[int] | None,
                    completed: int) -> float | None:
    """Recompute the documented RHAE game score (0-100) as a cross-check."""
    if not baselines:
        return None
    total_w, score = 0, 0.0
    for i, base in enumerate(baselines):
        w = i + 1
        total_w += w
        if i < completed and i < len(level_actions) and level_actions[i] > 0:
            score += w * min((base / level_actions[i]) ** 2 * 100, 115.0)
    max_score = 100.0 * sum(range(1, completed + 1)) / total_w
    return min(score / total_w, max_score)


def official_score(arcade, card_id) -> dict[str, Any] | None:
    """Pull this game's run from the toolkit scorecard. None if unavailable."""
    try:
        card = arcade.get_scorecard(card_id)
    except Exception:
        return None
    if card is None or not getattr(card, "environments", None):
        return None
    env = card.environments[0]
    runs = getattr(env, "runs", []) or []
    if not runs:
        return None
    best = max(runs, key=lambda r: r.score)
    return {"score": float(best.score), "n_runs": len(runs),
            "levels_completed": best.levels_completed, "actions": best.actions,
            "resets": best.resets, "level_scores": best.level_scores,
            "level_actions": best.level_actions,
            "level_baseline_actions": best.level_baseline_actions,
            "message": getattr(best, "message", None)}


# ------------------------------------------------------------------------ agent
def load_policy(spec: str, kwargs: dict, seed: int, click_grid: int):
    if spec == "random":
        from agents.random_agent import RandomPolicy
        return RandomPolicy(click_grid=click_grid, seed=seed, **kwargs)
    module, _, cls = spec.partition(":")
    klass = getattr(importlib.import_module(module), cls)
    return klass(click_grid=click_grid, seed=seed, **kwargs)


# ------------------------------------------------------------------- one game
def run_game(arcade, policy, game_id: str, seed: int, game_seed: int, budget: int,
             click_grid: int, max_seconds: float | None, traj_path: Path | None,
             tags: list[str], save_recording: bool = False) -> dict[str, Any]:
    card = arcade.create_scorecard(tags=tags)
    env = ArcGymEnv(game_id, arcade=arcade, game_seed=game_seed, scorecard_id=card,
                    click_grid=click_grid, reset_mode="level", save_recording=save_recording)
    policy.reset_game(game_id, seed)

    t0 = time.perf_counter()
    obs, info = env.reset(seed=seed)
    win_levels = info["win_levels"]
    level_end_actions: list[int] = []        # official action count when each level finished
    illegal, game_overs, stop_reason = 0, 0, "budget"
    full_resets_after_start = 0
    traj = [] if traj_path else None

    while env.official_actions < budget:
        if max_seconds and time.perf_counter() - t0 > max_seconds:
            stop_reason = "time_limit"
            break
        a = policy.act(obs, info)
        nobs, r, term, trunc, ninfo = env.step(a)
        if ninfo["illegal_action"]:
            raise RuntimeError(f"Policy returned illegal action {a} for {game_id}; evaluation aborted")
        policy.observe(obs, a, r, nobs, term, trunc, ninfo)
        illegal += int(ninfo["illegal_action"])
        while len(level_end_actions) < ninfo["levels_completed"]:
            # scored actions = agent actions + level RESETs (the scorecard counts each reset as an action)
            level_end_actions.append(ninfo["official_actions"] + ninfo["resets_sent"])
        if traj is not None:
            traj.append({"t": ninfo["episode_steps"], "official_actions": ninfo["official_actions"],
                         "action": ninfo["action_label"], "illegal": ninfo["illegal_action"],
                         "reward": r, "state": ninfo["state"],
                         "levels_completed": ninfo["levels_completed"]})
        obs, info = nobs, ninfo
        if info["state"] == GameState.WIN.value:
            stop_reason = "win"
            break
        if info["state"] == GameState.GAME_OVER.value:
            game_overs += 1
            obs, info = env.reset()                # level reset (competition semantics)
            full_resets_after_start += int(info["full_reset"])

    runtime = time.perf_counter() - t0
    per_level = [b - a for a, b in zip([0] + level_end_actions[:-1], level_end_actions)]
    off = official_score(arcade, card)
    try:
        arcade.close_scorecard(card)
    except Exception:
        pass

    gi = env.full_game_info
    baselines = list(getattr(gi, "baseline_actions", None) or []) or None
    if traj is not None:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traj_path, "w") as f:
            for row in traj:
                f.write(json.dumps(row) + "\n")

    completed = info["levels_completed"]
    return {
        "game_id": info["game_id"], "game_title": getattr(gi, "title", None),
        "seed": seed, "game_seed": game_seed,
        "success": info["state"] == GameState.WIN.value,
        "levels_completed": completed, "win_levels": win_levels,
        "score_official": None if off is None else off["score"],
        "score_recomputed": rhae_game_score(per_level, baselines, completed),
        "score_note": None if off is None else off.get("message"),
        "official_actions": env.official_actions, "action_budget": budget,
        "scored_actions": env.official_actions + env.resets_sent,
        "level_actions": per_level, "baseline_actions": baselines,
        "level_resets": env.resets_sent, "game_overs": game_overs,
        "full_resets_after_start": full_resets_after_start,
        "illegal_actions": illegal, "stop_reason": stop_reason,
        "runtime_s": round(runtime, 3),
        "actions_per_s": round(env.official_actions / runtime, 2) if runtime else None,
        "scorecard_runs": None if off is None else off["n_runs"],
        "trajectory_file": str(traj_path) if traj_path else None,
    }


# ------------------------------------------------------------------------ main
def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main(argv=None, arcade_factory=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="Configs/eval.yaml")
    ap.add_argument("--agent", default="random", help="'random' or 'module.path:ClassName'")
    ap.add_argument("--agent-kwargs", default="{}", help="JSON dict passed to the policy")
    ap.add_argument("--agent-version", default=None, help="e.g. ppo-base, ppo-imp1, dqn-imp2")
    ap.add_argument("--checkpoint-step", type=int, default=None, help="training interactions at checkpoint")
    ap.add_argument("--split", default="dev", help="key under games: in the config")
    ap.add_argument("--games", nargs="*", help="override the split with explicit game ids")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--budget", type=int, help="official actions per game")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-traj", action="store_true", help="per-step action log (light)")
    ap.add_argument("--save-recording", action="store_true",
                    help="official toolkit recordings with frames (replays for failure analysis; large)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    ev = cfg["eval"]
    games = args.games or cfg["games"].get(args.split) or []
    if not games:
        raise SystemExit(f"No games in split '{args.split}'. Fill Configs/eval.yaml (python test_env.py --list).")
    seeds = args.seeds or ev["seeds"]
    budget = args.budget or ev["max_actions_per_game"]
    click_grid = ev["click_grid"]
    kwargs = json.loads(args.agent_kwargs)
    version = args.agent_version or args.agent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"results/eval/{version}_{args.split}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    if os.environ.get("ONLY_RESET_LEVELS") == "true":
        raise SystemExit("Unset ONLY_RESET_LEVELS: it prevents the official scorecard from recording (see docstring).")
    if args.save_recording:
        os.environ["RECORDINGS_DIR"] = str(out / "recordings")   # read by Arcade() at construction
    if arcade_factory is None:
        import arc_agi
        mode = arc_agi.OperationMode(ev.get("operation_mode", "normal"))
        arcade_factory = lambda: arc_agi.Arcade(operation_mode=mode)  # noqa: E731

    meta = {"timestamp_utc": stamp, "agent": args.agent, "agent_version": version,
            "agent_kwargs": kwargs, "checkpoint_step": args.checkpoint_step,
            "split": args.split, "games": games, "seeds": seeds, "budget": budget,
            "click_grid": click_grid, "only_reset_levels_env": os.environ.get("ONLY_RESET_LEVELS"),
            "git_commit": git_commit(), "versions": versions(), "hardware": hardware(),
            "config": cfg}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    records = []
    for seed in seeds:
        arcade = arcade_factory()
        policy = load_policy(args.agent, kwargs, seed, click_grid)
        game_seed = seed if ev.get("game_seed", "same_as_run") == "same_as_run" else int(ev["game_seed"])
        for g in games:
            traj = out / "traj" / f"{version}_s{seed}_{g}.jsonl" if args.save_traj else None
            rec = run_game(arcade, policy, g, seed, game_seed, budget, click_grid,
                           ev.get("max_seconds_per_game"), traj,
                           tags=[version, f"seed{seed}", args.split],
                           save_recording=args.save_recording)
            rec.update({"agent": args.agent, "agent_version": version,
                        "checkpoint_step": args.checkpoint_step, "split": args.split,
                        "timestamp_utc": stamp})
            records.append(rec)
            if rec["full_resets_after_start"] or (rec["scorecard_runs"] or 1) > 1:
                print(f"WARNING: {rec['game_id']} seed {seed} had a full reset; score is not competition-equivalent")
            print(f"[{version} seed={seed}] {rec['game_id']}: levels {rec['levels_completed']}/"
                  f"{rec['win_levels']} score={rec['score_official']} actions={rec['official_actions']} "
                  f"stop={rec['stop_reason']} {rec['runtime_s']}s")

    with open(out / "episodes.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        for r in records:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})

    def score_of(r):
        return r["score_official"] if r["score_official"] is not None else r["score_recomputed"]

    agg = {"agent_version": version, "split": args.split, "budget": budget, "per_game": {}, "overall": {}}
    for g in sorted({r["game_id"] for r in records}):
        rs = [r for r in records if r["game_id"] == g]
        m, s = mean_std([score_of(r) for r in rs])
        lm, ls = mean_std([r["levels_completed"] for r in rs])
        agg["per_game"][g] = {"score_mean": m, "score_std": s, "levels_mean": lm, "levels_std": ls,
                              "win_rate": sum(r["success"] for r in rs) / len(rs), "n": len(rs)}
    per_seed = []
    for seed in seeds:
        rs = [score_of(r) for r in records if r["seed"] == seed]
        rs = [x for x in rs if x is not None]
        if rs:
            per_seed.append(statistics.fmean(rs))   # ARC total = mean of game scores
    m, s = mean_std(per_seed)
    agg["overall"] = {"score_mean_over_seeds": m, "score_std_over_seeds": s,
                      "per_seed_scores": per_seed,
                      "total_levels_completed": sum(r["levels_completed"] for r in records),
                      "total_official_actions": sum(r["official_actions"] for r in records),
                      "total_runtime_s": round(sum(r["runtime_s"] for r in records), 2)}
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(f"\nOverall score (mean over seeds): {m}  std: {s}\nResults in {out}")
    return out


if __name__ == "__main__":
    main()
