"""Scientific contract and real learner continuation, without heldout games."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from DQN.extended_train import check_resume, train, replay_load
from DQN.replay_buffer import ReplayBuffer
from tests.fake_arc import FakeArcade


def tiny_config():
    c = json.loads(Path("Configs/dqn_extended_v1.json").read_text())
    return {**c, "training_games": ["a-0001", "b-0001", "c-0001"], "total_actions": 30,
            "warmup": 4, "train_every": 2, "batch_size": 2, "buffer_capacity": 7,
            "checkpoint_actions": [12, 30], "resume_every": 12, "log_every": 30,
            "target_update_interval": 2}


@pytest.mark.parametrize("variant", ["v0", "v1", "v2_history_only", "v2"])
def test_multigame_resume_is_identical_to_uninterrupted(variant, tmp_path):
    c = tiny_config()
    full = tmp_path / "full"
    prefix = tmp_path / "prefix"
    restored = tmp_path / "restored"
    train(c, variant, 42, "cpu", full, arcade_factory=FakeArcade)
    train(c, variant, 42, "cpu", prefix, stop_at=12, arcade_factory=FakeArcade)
    report = train(c, variant, 42, "cpu", restored, resume=prefix / "resume/step000012", arcade_factory=FakeArcade)
    assert report["gradient_steps"] == 14
    assert [r["actions"] for r in report["per_game"]] == [10, 10, 10]
    a, b = [check_resume(p / "resume/step000030") for p in (full, restored)]
    assert a["journal"] == b["journal"]
    assert a["agent_rng"] == b["agent_rng"]
    for model in ("online_state_dict", "target_state_dict"):
        for key, tensor in a[model].items():
            torch.testing.assert_close(tensor, b[model][key], rtol=0, atol=0)
    for i, j in zip(a["contexts"], b["contexts"]):
        assert i["visits"] == j["visits"] and i["action_counts"] == j["action_counts"]
        torch.testing.assert_close(i["history"], j["history"], rtol=0, atol=0)
    ra, rb = ReplayBuffer(7), ReplayBuffer(7)
    replay_load(ra, full / "resume/step000030/replay.npz", a["replay_position"])
    replay_load(rb, restored / "resume/step000030/replay.npz", b["replay_position"])
    assert ra.position == rb.position
    for x, y in zip(ra.storage, rb.storage):
        np.testing.assert_array_equal(x.state, y.state)
        np.testing.assert_array_equal(x.next_valid_mask, y.next_valid_mask)
        assert x.reward == y.reward and x.action == y.action
    cpu = torch.load(full / "frozen/step000030.pt", map_location="cpu", weights_only=True)
    assert cpu["config"]["num_frames"] == (4 if variant.startswith("v2") else 1)
    rows = list(csv.DictReader((full / "actions.csv").open()))
    restored_rows = list(csv.DictReader((restored / "actions.csv").open()))
    assert len(restored_rows) == 30
    assert [r["global_step"] for r in restored_rows] == [str(i) for i in range(1,31)]
    assert [r["game"] for r in rows] == c["training_games"] * 10
    assert all(r["truncated"] == "False" for r in rows)
    if variant in ("v0", "v2_history_only"):
        assert all(float(r["intrinsic_reward"]) == 0 for r in rows)


def test_reject_corrupt_or_mismatched_resume_and_preserve_output(tmp_path):
    c = tiny_config()
    train(c, "v2", 42, "cpu", tmp_path / "prefix", 12, arcade_factory=FakeArcade)
    resume = tmp_path / "prefix/resume/step000012"
    with pytest.raises(ValueError, match="protocol/variant"):
        train(c, "v1", 42, "cpu", tmp_path / "bad", resume=resume, arcade_factory=FakeArcade)
    with pytest.raises(FileExistsError):
        train(c, "v2", 42, "cpu", tmp_path / "prefix", 12, arcade_factory=FakeArcade)
    with (resume / "replay.npz").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="corrupt"):
        check_resume(resume)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("variant", ["v0", "v2"])
def test_cuda_continuation(variant, tmp_path):
    c = tiny_config()
    train(c, variant, 42, "cuda", tmp_path / "full", arcade_factory=FakeArcade)
    train(c, variant, 42, "cuda", tmp_path / "prefix", 12, arcade_factory=FakeArcade)
    train(c, variant, 42, "cuda", tmp_path / "resume", resume=tmp_path / "prefix/resume/step000012", arcade_factory=FakeArcade)
    a, b = [check_resume(tmp_path / x / "resume/step000030") for x in ("full", "resume")]
    assert a["journal"] == b["journal"]
    for key, tensor in a["online_state_dict"].items():
        torch.testing.assert_close(tensor, b["online_state_dict"][key], rtol=0, atol=0)


def test_persistence_and_pilot_validation(tmp_path):
    from Experiments.validate_dqn_pilot import validate
    c = tiny_config()
    train(c, "v0", 42, "cpu", tmp_path / "run", persistent=tmp_path / "drive", arcade_factory=FakeArcade)
    result = validate(tmp_path / "run", required_actions=30, require_cuda=False)
    assert result["valid"] and result["expected_updates"] == 14
    remote = tmp_path / "drive/run/resume/step000030"
    assert check_resume(remote)["global_step"] == 30
    assert len(list((tmp_path / "run/resume").glob("step*"))) == 2


def test_missing_pilot_blocks_matrix(tmp_path):
    from Experiments.run_dqn_extended import matrix_gate
    with pytest.raises(ValueError, match="Both 75K"):
        matrix_gate(tmp_path, tiny_config())


def test_training_seeds_are_aggregation_unit():
    from Experiments.evaluate_dqn_extended import aggregate
    records = []
    for seed, value in [(42, 0), (123, 1), (456, 2)]:
        for evaluation_seed in range(3):
            records.append(dict(variant="v0",checkpoint_step=75000,training_seed=seed,
                game_id="game",score_official=value,levels_completed=value,success=bool(value),action_entropy_nats=0))
    result = aggregate(records)["across_training_seed_summary"][0]
    assert result["independent_training_seeds"] == 3
    assert result["score_official"]["mean"] == 1 and result["score_official"]["sd"] == 1
