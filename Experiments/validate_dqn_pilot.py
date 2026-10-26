"""Validate actual pilot evidence; zero reward is never an implementation failure."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from DQN.dqn_agent import DQNAgent
from DQN.extended_train import check_resume, hash_file, json_write


def validate(run_dir, required_actions=75000, require_cuda=True):
    root = Path(run_dir)
    report = json.loads((root / "summary.json").read_text())
    cfg = report["protocol"]
    rows = list(csv.DictReader((root / "actions.csv").open()))
    initial = report["initial_step"]
    assert initial == 0, "Pilot validation requires a complete fresh pilot log"
    assert report["global_step"] == required_actions and len(rows) == required_actions
    assert [int(r["global_step"]) for r in rows] == list(range(1, required_actions+1))
    if require_cuda:
        assert "T4" in report["device_name"] and report["device"].startswith("cuda")
    expected = sum(s >= cfg["warmup"] and s % cfg["train_every"] == 0 for s in range(1,required_actions+1))
    losses = [float(r["loss"]) for r in rows if r["loss"]]
    assert len(losses) == expected == report["gradient_steps"]
    assert all(np.isfinite(losses)) and report["finite_losses"]
    assert all(r["truncated"] == "False" for r in rows)
    assert all(np.isfinite(float(r[k])) for r in rows for k in ("task_reward", "intrinsic_reward", "epsilon"))
    counts = Counter(r["game"] for r in rows)
    assert [r["game"] for r in rows] == [cfg["training_games"][i % len(cfg["training_games"])] for i in range(required_actions)]
    assert max(counts.values()) - min(counts.values()) <= 1
    if report["variant"] in ("v0", "v2_history_only"):
        assert all(float(r["intrinsic_reward"]) == 0 for r in rows)
    resume = root / "resume" / f"step{required_actions:06d}"
    state = check_resume(resume)
    assert state["global_step"] == required_actions
    assert len(state["journal"]) >= required_actions
    assert len(state["contexts"]) == len(cfg["training_games"])
    for g, ctx in zip(cfg["training_games"], state["contexts"]):
        assert ctx["actions"] == counts[g]
    frozen = root / "frozen" / f"step{required_actions:06d}.pt"
    model = torch.load(frozen, map_location="cpu", weights_only=True)
    assert model["gradient_steps"] == expected and model["training_backend"] == "shared"
    cpu = DQNAgent(num_frames=report["num_frames"])
    cpu.online.load_state_dict(model["online_state_dict"])
    cpu.target.load_state_dict(model["target_state_dict"])
    cpu.optimizer.load_state_dict(model["optimizer_state_dict"])
    for net in (cpu.online, cpu.target):
        assert all(torch.isfinite(p).all().item() for p in net.parameters())
    with np.load(resume / "replay.npz", allow_pickle=False) as d:
        assert d["states"].shape == (min(cfg["buffer_capacity"],required_actions),report["num_frames"],64,64)
        assert d["masks"].shape[0] == d["states"].shape[0]
    result = {"valid": True, "required_actions": required_actions, "variant": report["variant"],
        "seed": report["seed"], "device_name": report["device_name"], "expected_updates": expected,
        "per_game_actions": dict(counts), "frozen_sha256": hash_file(frozen),
        "protocol": cfg, "source_sha256": report["source_sha256"], "runtime_assets": report["runtime_assets"],
        "actions_per_second_including_persistence": report["actions_per_second_including_persistence"],
        "training_quality_note": "zero task reward does not fail infrastructure validation",
        "gpu_samples": sum(1 for line in (root / "gpu_samples.csv").open())-1}
    if require_cuda:
        assert result["gpu_samples"] > 0 and report["peak_cuda_memory_bytes"] > 0
        samples = list(csv.DictReader((root / "gpu_samples.csv").open()))
        utilization = [float(r["utilization_percent"]) for r in samples]
        assert all(0 <= u <= 100 for u in utilization)
        assert all(float(r["memory_used_MiB"]) > 0 for r in samples)
        result["sampled_gpu_utilization_percent"] = {"mean":statistics.fmean(utilization),
            "min":min(utilization),"max":max(utilization),
            "note":"periodic instantaneous samples; not a continuous time-weighted utilization estimate"}
    json_write(root / "pilot_validation.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    print(json.dumps(validate(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
