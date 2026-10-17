"""Offline tests for the adapter, random agent and evaluator: python -m pytest tests -q"""
import json
import os

import numpy as np

from arc_env.encoding import ActionMap, frame_to_grid, PAD_VALUE
from arc_env.gym_adapter import ArcGymEnv
from agents.random_agent import RandomPolicy
from tests.fake_arc import FakeArcade

IDX = {spec.label(): i for i, spec in enumerate(ActionMap(8).specs)}


def make_env(**kw):
    return ArcGymEnv("fake-0001", arcade=FakeArcade(), **kw)


def test_encoding_pads_and_takes_last_frame():
    small = [np.zeros((10, 12)), np.full((10, 12), 7)]
    g = frame_to_grid(small)
    assert g.shape == (64, 64) and g.dtype == np.uint8
    assert (g[:10, :12] == 7).all() and (g[10:, :] == PAD_VALUE).all()


def test_action_map_and_mask():
    am = ActionMap(8)
    assert len(am) == 6 + 64
    assert am.decode(IDX["ACTION6(4,4)"]).data == {"x": 4, "y": 4}
    m = am.mask([1, 2, 6])
    assert m[IDX["ACTION1"]] and not m[IDX["ACTION3"]] and m[IDX["ACTION6(60,60)"]]


def test_reward_levels_and_win_termination():
    env = make_env()
    obs, info = env.reset()
    assert obs.shape == (64, 64) and info["win_levels"] == 2
    rewards = []
    for _ in range(6):
        obs, r, term, trunc, info = env.step(IDX["ACTION1"])
        rewards.append(r)
    assert rewards == [0, 0, 1, 0, 0, 1]
    assert term and info["state"] == "WIN" and env.official_actions == 6


def test_illegal_action_not_sent():
    env = make_env()
    env.reset()
    _, r, term, _, info = env.step(IDX["ACTION3"])
    assert info["illegal_action"] and env.official_actions == 0 and not term


def test_game_over_then_level_reset_keeps_progress():
    os.environ["ONLY_RESET_LEVELS"] = "false"
    env = make_env(reset_mode="level")
    env.reset()
    for _ in range(3):
        env.step(IDX["ACTION1"])          # level 1 done
    _, _, term, _, info = env.step(IDX["ACTION2"])
    assert term and info["state"] == "GAME_OVER"
    _, info = env.reset()
    assert info["levels_completed"] == 1 and not info["full_reset"]


def test_full_reset_mode_restarts_game():
    os.environ["ONLY_RESET_LEVELS"] = "false"
    env = make_env(reset_mode="full")
    env.reset()
    for _ in range(3):
        env.step(IDX["ACTION1"])
    env.step(IDX["ACTION2"])
    _, info = env.reset()
    assert info["levels_completed"] == 0 and info["full_reset"]


def test_truncation():
    env = make_env(max_episode_actions=2)
    env.reset()
    env.step(IDX["ACTION6(4,4)"])
    _, _, term, trunc, _ = env.step(IDX["ACTION6(4,4)"])
    assert trunc and not term


def test_random_policy_only_legal_and_seeded():
    env = make_env()
    _, info = env.reset()
    a = [RandomPolicy(seed=1).act(None, info) for _ in range(3)]
    b = [RandomPolicy(seed=1).act(None, info) for _ in range(3)]
    assert a == b
    p = RandomPolicy(seed=0)
    for _ in range(200):
        assert info["action_mask"][p.act(None, info)]


def test_evaluate_end_to_end(tmp_path):
    import evaluate
    cfg = tmp_path / "eval.yaml"
    cfg.write_text("games: {dev: [fake-0001]}\neval: {seeds: [0, 1, 2], max_actions_per_game: 200, click_grid: 8}\n")
    out = evaluate.main(["--config", str(cfg), "--split", "dev", "--out", str(tmp_path / "o"),
                         "--save-traj"], arcade_factory=FakeArcade)
    recs = [json.loads(l) for l in open(out / "episodes.jsonl")]
    assert len(recs) == 3 and {r["seed"] for r in recs} == {0, 1, 2}
    for r in recs:
        assert r["official_actions"] <= 200
        assert r["stop_reason"] in ("win", "budget")
        assert sum(r["level_actions"]) <= r["scored_actions"]
        if r["success"]:
            assert r["score_recomputed"] is not None
    agg = json.load(open(out / "aggregate.json"))
    assert len(agg["overall"]["per_seed_scores"]) == 3
    assert (out / "run_meta.json").exists() and (out / "summary.csv").exists()


def test_evaluate_official_score_with_real_toolkit(tmp_path):
    """Runs the real arc_agi toolkit and scorecard on a 2-level mini game (offline)."""
    import arc_agi
    import evaluate
    from pathlib import Path
    os.environ.pop("ONLY_RESET_LEVELS", None)
    games_dir = str(Path(__file__).parent / "fixtures" / "games")
    factory = lambda: arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,  # noqa: E731
                                     environments_dir=games_dir)
    cfg = tmp_path / "eval.yaml"
    cfg.write_text("games: {dev: [tt01-v1]}\neval: {seeds: [0, 1, 2, 3], max_actions_per_game: 300, click_grid: 8}\n")
    out = evaluate.main(["--config", str(cfg), "--out", str(tmp_path / "o")], arcade_factory=factory)
    recs = [json.loads(l) for l in open(out / "episodes.jsonl")]
    assert any(r["game_overs"] > 0 for r in recs)              # level resets were exercised
    for r in recs:
        assert r["score_official"] is not None                 # the bug this test guards against
        assert r["full_resets_after_start"] == 0 and r["scorecard_runs"] == 1
        assert abs(r["score_official"] - r["score_recomputed"]) < 1e-6
