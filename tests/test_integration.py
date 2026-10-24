"""Regression checks for checkpoint semantics and shared evaluation failure modes."""
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.checkpoint_encoding import GridHistory
from agents.dqn_agent import DQNPolicy
from agents.ppo_agent import PPOPolicy
from arc_env.encoding import ActionMap
from DQN.action_codec import valid_action_mask
from DQN.q_network import observation_to_tensor
from tests.fake_arc import FakeArcade


def test_history_advances_once_padding_and_reset():
    h = GridHistory(4)
    grid = np.full((64, 64), 2, dtype=np.uint8)
    grid[0, 0] = 16
    first = h.push(grid).reshape(4, 16, 64, 64)
    assert first[:, :, 0, 0].sum() == 0
    assert first[:, 2, 1, 1].tolist() == [1, 1, 1, 1]
    second = h.push(np.full((64, 64), 3, dtype=np.uint8)).reshape(4, 16, 64, 64)
    assert second[:, :, 1, 1].argmax(axis=1).tolist() == [2, 2, 2, 3]
    h.clear()
    reset = h.push(np.full((64, 64), 4, dtype=np.uint8)).reshape(4, 16, 64, 64)
    assert reset[:, :, 1, 1].argmax(axis=1).tolist() == [4, 4, 4, 4]


def test_action_indices_reject_negative_and_out_of_range():
    amap = ActionMap(8)
    for value in (-1, 70, 1.5):
        with pytest.raises(ValueError):
            amap.decode(value)


@pytest.mark.parametrize("variant,frames", [("v0", 1), ("v1", 1), ("v2", 4)])
def test_dqn_full_grid_preserves_native_greedy_decision(variant, frames):
    from arcengine import GameAction
    checkpoint = Path(f"DQN/checkpoints/dqn_{variant}_ls20_seed42.pt")
    if not checkpoint.exists():
        pytest.skip("Local DQN checkpoints are distributed separately")
    p = DQNPolicy(checkpoint, click_grid=64)
    assert p.num_frames == frames
    np.testing.assert_array_equal(p.indices.cpu(), np.arange(4102))
    grid = np.full((64, 64), 3, dtype=np.uint8)
    legal = [GameAction.ACTION1, GameAction.ACTION6, GameAction.ACTION7]
    mask = valid_action_mask(legal)
    native = np.stack([grid] * frames)
    with torch.inference_mode():
        expected = p.model(observation_to_tensor(native))[0].masked_fill(~mask, -torch.inf).argmax().item()
    before = {k: v.clone() for k, v in p.model.state_dict().items()}
    assert p.act(grid, {"action_mask": mask.numpy()}) == expected
    p.observe(grid, expected, 123, grid, True, False, {})
    assert not p.history.frames
    for key, value in p.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_ppo_mask_order_clicks_and_game_seed():
    checkpoint = "PPO/logs/baseline/ppo_baseline_seed1.zip"
    p = PPOPolicy(checkpoint, click_grid=64)
    grid = np.zeros((64, 64), dtype=np.uint8)
    mask = ActionMap(64).mask([6])
    calls = []
    def predict(obs, deterministic, action_masks):
        calls.append(action_masks.copy())
        assert obs.shape == (16, 64, 64) and deterministic
        return np.array(5), None  # native ACTION6, not shared ACTION7 at index 5
    p.model.predict = predict
    p.reset_game("fake", 123)
    actions = [p.act(grid, {"action_mask": mask}) for _ in range(5)]
    p.reset_game("fake", 123)
    assert actions == [p.act(grid, {"action_mask": mask}) for _ in range(5)]
    assert calls[0].tolist() == [False, False, False, False, False, True, False]
    assert all(mask[a] and a >= 6 for a in actions)


def test_evaluator_aborts_illegal_policy_instead_of_hanging():
    from evaluate import run_game
    from agents.base import Policy
    class Illegal(Policy):
        def act(self, obs, info):
            return 2  # ACTION3 is unavailable in FakeArcade
    with pytest.raises(RuntimeError, match="illegal action"):
        run_game(FakeArcade(), Illegal(), "fake-0001", 0, 0, 10, 8, None, None, [])


def test_competition_bridge_clears_history_at_game_over():
    from types import SimpleNamespace
    from arcengine import GameAction, GameState
    from agents.base import Policy
    from agents.kaggle_bridge import PolicyBridge
    class Tracking(Policy):
        def __init__(self):
            self.games, self.episodes = [], 0
        def reset_game(self, game_id, seed):
            self.games.append(game_id)
        def reset_episode(self):
            self.episodes += 1
    p = Tracking()
    bridge = PolicyBridge(p)
    frame = SimpleNamespace(game_id="fake", state=GameState.GAME_OVER)
    assert bridge.choose_action(frame) is GameAction.RESET
    assert p.games == ["fake"] and p.episodes == 1
    assert bridge.choose_action(frame) is GameAction.RESET
    assert p.games == ["fake"] and p.episodes == 2
