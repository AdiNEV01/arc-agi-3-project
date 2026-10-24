"""Offline end-to-end compatibility smoke, not a comparative experiment.

Run from the repository root: python -m Experiments.integration_smoke
Requires local DQN checkpoints and the three committed PPO checkpoints.
It never trains, reads hidden game source, or runs untouched-heldout games.
"""
import argparse
import hashlib
import json
from pathlib import Path

import arc_agi
import torch

import evaluate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/integration_smoke_20261024")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; use a new --out")
    torch.set_num_threads(1)
    policies = [("random", "random", None, None)]
    policies += [(f"dqn_{v}", "agents.dqn_agent:DQNPolicy",
                  f"DQN/checkpoints/dqn_{v}_ls20_seed42.pt", 3000)
                 for v in ("v0", "v1", "v2")]
    policies += [
        ("ppo_baseline", "agents.ppo_agent:PPOPolicy", "PPO/logs/baseline/ppo_baseline_seed1.zip", 12288),
        ("ppo_curiosity", "agents.ppo_agent:PPOPolicy", "PPO/logs/improvement1_curiosity/ppo_curiosity_seed0.zip", 12288),
        ("ppo_memory_curiosity", "agents.ppo_agent:PPOPolicy", "PPO/logs/improvement2_memory_curiosity/ppo_memory_seed0.zip", 12288),
    ]
    for _, _, checkpoint, _ in policies:
        if checkpoint and not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    hashes = {c: digest(c) for _, _, c, _ in policies if c}
    cases = [("fixture", "tt01-v1", "tests/fixtures/games", 30),
             ("ls20", "ls20-9607627b", "environment_files", 140)]
    summary = []
    for case, game, directory, budget in cases:
        factory = lambda directory=directory: arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=directory)
        for name, spec, checkpoint, step in policies:
            target = out / case / name
            argv = ["--config", "Configs/integration_smoke.yaml", "--agent", spec,
                    "--agent-version", name, "--games", game, "--seeds", "0",
                    "--budget", str(budget), "--out", str(target), "--save-traj"]
            if checkpoint:
                argv += ["--agent-kwargs", json.dumps({"checkpoint": checkpoint}),
                         "--checkpoint-step", str(step)]
            evaluate.main(argv, arcade_factory=factory)
            record = json.loads((target / "episodes.jsonl").read_text().strip())
            assert record["illegal_actions"] == 0
            assert record["score_official"] is not None
            assert record["full_resets_after_start"] == 0
            assert record["scorecard_runs"] == 1
            assert record["official_actions"] <= budget
            assert record["score_recomputed"] is None or abs(
                record["score_official"] - record["score_recomputed"]) < 1e-6
            summary.append({"case": case, "policy": name, "checkpoint": checkpoint,
                            **{k: record[k] for k in ("levels_completed", "score_official",
                               "official_actions", "level_resets", "scorecard_runs",
                               "full_resets_after_start", "runtime_s")}})
    assert all(digest(c) == h for c, h in hashes.items())
    (out / "verification.json").write_text(json.dumps({
        "purpose": "compatibility smoke only; one evaluation seed, unequal training budgets",
        "checkpoint_sha256_before_and_after": hashes,
        "checkpoint_files_unchanged": True, "records": summary,
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
