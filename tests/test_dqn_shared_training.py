"""Shared training regression checks without touching historical experiments."""
import sys
from types import SimpleNamespace

import numpy as np
import torch
from arcengine import GameAction

from agents.checkpoint_encoding import GridHistory
from DQN.dqn_agent import DQNAgent
from DQN.q_network import encode_frames
from DQN.shared_environment import SharedDQNEnvironment
from tests.fake_arc import FakeArcade


def test_shared_native_action_identity_reward_and_reset():
    env = SharedDQNEnvironment("fake-0001", FakeArcade(), seed=42)
    initial = env.reset()
    assert initial.frame[0].shape == (64, 64)
    for _ in range(3):
        response = env.step(GameAction.ACTION1)
    assert response.task_reward == 1 and response.levels_completed == 1
    terminal = env.step(GameAction.ACTION2)
    assert terminal.terminated and not terminal.truncated
    reset = env.reset()
    assert reset.levels_completed == 1
    assert not reset.info["full_reset"]
    assert env.shared.official_actions == 4 and env.shared.resets_sent == 1


def test_dqn_training_padding_matches_frozen_policy_and_replay():
    frames = np.full((1, 64, 64), 2, dtype=np.uint8)
    frames[0, 0, 0] = 16
    encoded = encode_frames(frames)
    expected = GridHistory(1).push(frames[0])
    np.testing.assert_array_equal(encoded.numpy()[0], expected)
    assert encoded[0, :, 0, 0].sum() == 0
    a = DQNAgent(seed=42, batch_size=1)
    a.remember(frames, 0, 1, frames, False, [GameAction.ACTION1])
    assert a.replay.storage[0].next_valid_mask.any()
    assert np.isfinite(a.learn())
    a.remember(frames, 0, 1, frames, True, [])
    assert not a.replay.storage[1].next_valid_mask.any()


def test_shared_training_saves_distinct_prefix_and_final(monkeypatch, tmp_path):
    import DQN.train as train
    monkeypatch.setattr(train.arc_agi, "Arcade", lambda **kw: FakeArcade())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train", "--backend", "shared", "--steps", "12",
        "--episode-limit", "0", "--warmup", "2", "--train-every", "2", "--batch-size", "2",
        "--checkpoint-steps", "6", "--run-suffix", "sharedtest"])
    train.main()
    prefix = torch.load("DQN/checkpoints/dqn_v0_ls20_seed42_sharedtest_step6.pt", weights_only=True)
    final = torch.load("DQN/checkpoints/dqn_v0_ls20_seed42_sharedtest.pt", weights_only=True)
    assert (prefix["global_step"], prefix["gradient_steps"]) == (6, 3)
    assert (final["global_step"], final["gradient_steps"]) == (12, 6)
    assert final["training_backend"] == "shared"
    assert final["config"]["episode_limit"] == 0
