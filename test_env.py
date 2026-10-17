"""Interface probe and minimal end-to-end check (Task 3A).

Run this first; its output is the evidence for docs/interface.md.

  python test_env.py --list               # print every available game id
  python test_env.py --game ls20          # probe one game, write results/interface_probe_<game>.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from importlib import metadata
from pathlib import Path

import arc_agi
from arcengine import GameAction, GameState


def list_games(arc):
    for e in sorted(arc.get_environments(), key=lambda e: e.game_id):
        n = len(e.baseline_actions) if e.baseline_actions else "?"
        print(f"{e.game_id:20s} levels={n!s:3s} baseline={e.baseline_actions} title={e.title} tags={e.tags}")


def probe(arc, game: str, steps: int, seed: int):
    random.seed(seed)
    env = arc.make(game, seed=seed)
    if env is None:
        raise SystemExit(f"Could not make {game}")
    rep = {"game_id": env.info.game_id, "title": env.info.title,
           "baseline_actions": env.info.baseline_actions, "tags": env.info.tags,
           "versions": {p: metadata.version(p) for p in ("arc-agi", "arcengine")}}

    f0 = env.reset()
    rep["initial"] = {"state": f0.state.value, "n_frames": len(f0.frame),
                      "frame_shape": list(f0.frame[-1].shape) if f0.frame else None,
                      "dtype": str(f0.frame[-1].dtype) if f0.frame else None,
                      "colours_present": sorted(int(c) for c in set(f0.frame[-1].flatten())) if f0.frame else [],
                      "available_actions": list(f0.available_actions),
                      "levels_completed": f0.levels_completed, "win_levels": f0.win_levels,
                      "full_reset": f0.full_reset}

    frames_per_action, states, avail_sets, shapes = Counter(), Counter(), Counter(), Counter()
    events, level_ups, changed = [], 0, 0
    prev = f0
    for t in range(steps):
        legal = [a for a in env.action_space if a is not GameAction.RESET]
        a = random.choice(legal)
        data = {"x": random.randint(0, 63), "y": random.randint(0, 63)} if a.is_complex() else {}
        f = env.step(a, data=data)
        if f is None:
            events.append({"t": t, "event": "step returned None"})
            break
        frames_per_action[len(f.frame)] += 1
        states[f.state.value] += 1
        avail_sets[tuple(f.available_actions)] += 1
        if f.frame:
            shapes[tuple(f.frame[-1].shape)] += 1
            if prev.frame and (f.frame[-1] != prev.frame[-1]).any():
                changed += 1
        if f.levels_completed > prev.levels_completed:
            level_ups += 1
            events.append({"t": t, "event": "level completed", "levels": f.levels_completed})
        if f.state in (GameState.GAME_OVER, GameState.WIN):
            events.append({"t": t, "event": f.state.value})
            r1 = env.reset()
            events.append({"t": t, "event": "RESET after end", "full_reset": r1.full_reset,
                           "levels_completed": r1.levels_completed})
            f = r1
        prev = f

    # Reset semantics check: RESET after actions vs two RESETs in a row
    if env.observation_space is not None and env.observation_space.state != GameState.NOT_FINISHED:
        env.reset()
    legal = [a for a in env.action_space if a is not GameAction.RESET]
    if legal:                                  # take one real action (works for click-only games too)
        a = legal[0]
        env.step(a, data={"x": 32, "y": 32} if a.is_complex() else {})
    r1 = env.reset()
    r2 = env.reset()
    rep["reset_semantics"] = {"reset_after_actions_full_reset": r1.full_reset,
                              "second_consecutive_reset_full_reset": r2.full_reset}
    rep["random_run"] = {"steps": steps, "frames_per_action": dict(frames_per_action),
                         "states": dict(states), "frame_shapes": {str(k): v for k, v in shapes.items()},
                         "available_action_sets": {str(k): v for k, v in avail_sets.items()},
                         "frame_changed_fraction": changed / max(1, steps),
                         "levels_completed_events": level_ups, "events": events[:50]}
    rep["scorecard"] = json.loads(arc.get_scorecard().model_dump_json()) if arc.get_scorecard() else None
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    arc = arc_agi.Arcade()
    if args.list:
        list_games(arc)
    else:
        rep = probe(arc, args.game, args.steps, args.seed)
        Path("results").mkdir(exist_ok=True)
        out = Path(f"results/interface_probe_{args.game}.json")
        out.write_text(json.dumps(rep, indent=2, default=str))
        print(json.dumps({k: rep[k] for k in ("game_id", "initial", "reset_semantics")}, indent=2, default=str))
        print(f"Full report: {out}")
