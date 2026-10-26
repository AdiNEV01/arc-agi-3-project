"""Summarize measured pilot cost, archived PPO training cost and validation actions."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import zipfile

import numpy as np
import torch


def main():
    out = Path("results/dqn_budget_assessment_20261024")
    cp = torch.load("DQN/checkpoints/dqn_v0_ls20_seed42_shared10k_20261024.pt", weights_only=True)
    rows = list(csv.DictReader(Path("Logs/DQN/dqn_v0_ls20_seed42_shared10k_20261024.csv").open()))
    seconds = cp["training_seconds"]
    old = json.loads((out / "historical_runtimes.json").read_text())
    medians = {r["variant"]: r["median_seconds"] for r in old}
    estimates = []
    for actions in (10000, 30000, 75000, 450000):
        per_variant = {v: seconds * actions / 10000 * median / medians["v0"] for v, median in medians.items()}
        estimates.append({"selected_actions_per_model": actions,
                          "v0_seconds_measured_or_linear": seconds * actions / 10000,
                          "per_variant_estimated_seconds": per_variant,
                          "all_4_configurations_3_seeds_seconds_serial": 3 * sum(per_variant.values()),
                          "assumption": "linear V0 extrapolation; other variants scaled by historical 3K median runtime ratios; not measured new history training"})
    vr = json.loads(Path("results/ppo_finals_verified_20261024_complete/verification.json").read_text())
    validations = []
    for r in vr["checkpoints"]:
        name = r["path"].replace("final_model.zip", "evaluations.npz")
        raw = subprocess.check_output(["git", "show", f"{r['source_ref']}:{name}"])
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            lengths = data["ep_lengths"]
            vals = {"validation_calls": int(lengths.shape[0]), "episodes_per_call": int(lengths.shape[1]),
                    "validation_actions_excluded_from_num_timesteps": int(lengths.sum()),
                    "validation_callback_local_timesteps": data["timesteps"].tolist()}
        with zipfile.ZipFile(r["path"]) as archive:
            optim = torch.load(io.BytesIO(archive.read("policy.optimizer.pth")), weights_only=True, map_location="cpu")
        optimizer_steps = sorted({int(s["step"].item()) for s in optim["state"].values() if "step" in s})
        validations.append({"experiment": r["experiment"], "training_seed": r["training_seed"],
                            "actual_training_transitions": r["actual_timesteps"],
                            "actual_optimizer_steps_from_saved_state": optimizer_steps, **vals})
    source_files = ["DQN/train.py", "DQN/shared_environment.py", "DQN/dqn_agent.py", "DQN/q_network.py", "DQN/action_codec.py", "arc_env/gym_adapter.py"]
    report = {
        "pilot": {"selected_actions": cp["global_step"], "gradient_updates": cp["gradient_steps"],
                  "seconds": seconds, "actions_per_second": cp["global_step"] / seconds,
                  "max_levels_completed": max(int(r["levels_completed"]) for r in rows),
                  "task_reward_sum": sum(float(r["task_reward"]) for r in rows),
                  "last_official_actions": int(rows[-1]["official_actions"]),
                  "last_controller_resets": int(rows[-1]["controller_resets"]),
                  "truncations": sum(r["truncated"] == "True" for r in rows),
                  "finite_losses": all(np.isfinite(float(r["loss"])) for r in rows if r["loss"]),
                  "device": "CPU", "torch_threads": 1, "fresh_run_not_resume": True,
                  "config": cp["config"]},
        "runtime_estimates": estimates,
        "ppo_validation_budget": validations,
        "ppo_recorded_gpu_training_hours_total": sum(r["resources"]["wall_clock_sec"] for r in vr["checkpoints"]) / 3600,
        "ppo_transitions_per_training_game": 503808 // 3,
        "ppo_to_legacy_dqn_total_transition_ratio": 503808 / 3000,
        "ppo_to_pilot_dqn_total_transition_ratio": 503808 / 10000,
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
        "full_24_model_evaluation_run": False,
    }
    baseline = json.loads((out / "preservation_before.json").read_text())
    changed = [p for p, h in baseline["sha256"].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert not changed, changed
    report["historical_artifacts_preserved"] = len(baseline["sha256"])
    (out / "assessment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"pilot": report["pilot"], "estimates": estimates, "ppo_validation_first": validations[0],
                      "ppo_recorded_training_hours": report["ppo_recorded_gpu_training_hours_total"]}, indent=2))


if __name__ == "__main__":
    main()
