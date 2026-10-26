"""V0 development diagnostic only, not the complete 24-model final evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/dqn_budget_assessment_20261024/development_eval")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    models = [
        ("native_3k_v0", "DQN/checkpoints/dqn_v0_ls20_seed42.pt", 3000),
        ("shared_3k_v0", "DQN/checkpoints/dqn_v0_ls20_seed42_shared10k_20261024_step3000.pt", 3000),
        ("shared_10k_v0", "DQN/checkpoints/dqn_v0_ls20_seed42_shared10k_20261024.pt", 10000),
    ]
    for _, checkpoint, _ in models:
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)
    records = []
    for label, checkpoint, step in models:
        sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        target = out / label
        evaluate.main(["--config", "Configs/dqn_pilot_development.yaml",
                       "--agent", "agents.dqn_agent:DQNPolicy", "--agent-version", label,
                       "--checkpoint-step", str(step), "--agent-kwargs", json.dumps({"checkpoint": checkpoint}),
                       "--split", "dev", "--out", str(target), "--save-traj"])
        rs = [json.loads(row) for row in (target / "episodes.jsonl").read_text().splitlines()]
        for rec in rs:
            assert rec["illegal_actions"] == 0 and rec["full_resets_after_start"] == 0
            assert rec["scorecard_runs"] == 1 and rec["score_official"] is not None
            assert abs(rec["score_official"] - rec["score_recomputed"]) < 1e-6
            traj = [json.loads(row) for row in Path(rec["trajectory_file"]).read_text().splitlines()]
            counts = Counter(row["action"] for row in traj)
            probabilities = np.array(list(counts.values()), dtype=float) / len(traj)
            rec["policy_behavior"] = {
                "action_counts": dict(counts), "distinct_actions": len(counts),
                "action_entropy_nats": float(-(probabilities * np.log(probabilities)).sum()),
                "repeated_action_fraction": sum(a["action"] == b["action"] for a, b in zip(traj, traj[1:])) / max(1, len(traj)-1),
                "task_reward_sum": sum(row["reward"] for row in traj),
            }
            rec["model"] = label
            records.append(rec)
        assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == sha
        (out / "comparison.json").write_text(json.dumps(records, indent=2))
    behavior_changes = []
    for game in {r["game_id"] for r in records}:
        for seed in (0, 1, 2):
            subset = {r["model"]: r for r in records if r["game_id"] == game and r["seed"] == seed}
            paths = {k: [json.loads(x)["action"] for x in Path(r["trajectory_file"]).read_text().splitlines()]
                     for k, r in subset.items()}
            for left, right in (("native_3k_v0", "shared_3k_v0"), ("shared_3k_v0", "shared_10k_v0")):
                a, b = paths[left], paths[right]
                behavior_changes.append({"game": game, "evaluation_seed": seed, "left": left, "right": right,
                                         "action_sequence_difference_fraction": sum(x != y for x, y in zip(a, b)) / max(1, min(len(a), len(b)))})
    (out / "behavior_changes.json").write_text(json.dumps(behavior_changes, indent=2))
    print("Completed V0 diagnostic only: 3 checkpoint snapshots x 2 exposed games x 3 evaluation seeds.")


if __name__ == "__main__":
    main()
