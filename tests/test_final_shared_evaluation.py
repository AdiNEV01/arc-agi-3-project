import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from Experiments.evaluate_final_shared import FrozenGuard,validate_records,validate_protocol,aggregate


def records():
    return [dict(game_id=g,seed=s,game_seed=s,action_budget=1000,scorecard_runs=1,
                 full_resets_after_start=0,illegal_actions=0,score_official=0.,score_recomputed=0.,
                 official_actions=1000,success=False,stop_reason="budget")
            for g in ("ft09-0d8bbf25","vc33-5430563c","cd82-fb555c5d") for s in (0,1,2)]


def test_fixed_matrix_and_early_stop_validation():
    rs=records();validate_records(rs,[0,1,2],1000)
    with pytest.raises(ValueError):validate_records(rs[:-1],[0,1,2],1000)
    rs[0]["official_actions"]=129
    with pytest.raises(ValueError,match="stopped before"):validate_records(rs,[0,1,2],1000)


def test_score_and_game_seed_mismatches_rejected():
    rs=records();rs[0]["game_seed"]=99
    with pytest.raises(ValueError):validate_records(rs,[0,1,2],1000)
    rs=records();rs[0]["score_recomputed"]=1
    with pytest.raises(ValueError):validate_records(rs,[0,1,2],1000)


def test_learning_guard_blocks_optimizer_and_detects_weight_change():
    net=torch.nn.Linear(2,2);net.optimizer=torch.optim.Adam(net.parameters())
    p=SimpleNamespace(model=SimpleNamespace(policy=net,learn=lambda:None))
    guard=FrozenGuard(p,"PPO")
    assert guard.check()["optimizer_state_unchanged"]
    with pytest.raises(RuntimeError,match="forbidden"):net.optimizer.step()
    with pytest.raises(RuntimeError,match="forbidden"):p.model.learn()
    with torch.no_grad():net.weight.add_(1)
    with pytest.raises(ValueError,match="weights changed"):guard.check()


def test_aggregate_keeps_training_seeds_as_independent_units():
    rows=[]
    for s in (42,123,456):
        for r in records():
            r.update(algorithm="DQN",variant="v0",training_seed=s,levels_completed=0,runtime_s=1,action_entropy_nats=0)
            rows.append(r)
    result=aggregate(rows)
    assert len(result["seed_level_means"])==3
    assert result["across_training_seed_summary"][0]["independent_training_seeds"]==3


def test_reserved_game_protocol_is_rejected(tmp_path):
    p=Path("Configs/final_shared_development.yaml")
    validate_protocol(p)
    bad=tmp_path/"bad.yaml";bad.write_text(p.read_text().replace("ft09-0d8bbf25","sb26-7fbdac44"))
    with pytest.raises(ValueError,match="protocol changed"):validate_protocol(bad)
