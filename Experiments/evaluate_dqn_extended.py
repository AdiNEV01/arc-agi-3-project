"""Common Task3 development curves, with training seeds as the replicate unit."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

import torch
import evaluate
from DQN.extended_train import hash_file, json_write


def aggregate(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["variant"],r["checkpoint_step"],r["training_seed"])].append(r)
    seed_means = []
    for (variant, step, seed), rs in sorted(grouped.items()):
        # Evaluation seeds are repeated measurements, NOT independent trained models.
        row = {"variant":variant,"checkpoint_step":step,"training_seed":seed}
        for key in ("score_official","levels_completed","success","action_entropy_nats"):
            games = defaultdict(list)
            for r in rs:
                games[r["game_id"]].append(float(r[key]))
            row[key] = statistics.fmean(statistics.fmean(v) for v in games.values())
        seed_means.append(row)
    final = []
    groups = defaultdict(list)
    for r in seed_means:
        groups[(r["variant"],r["checkpoint_step"])].append(r)
    for (variant, step), rs in sorted(groups.items()):
        entry = {"variant":variant,"checkpoint_step":step,"independent_training_seeds":len(rs)}
        for key in ("score_official","levels_completed","success","action_entropy_nats"):
            vals = [r[key] for r in rs]
            entry[key] = {"mean":statistics.fmean(vals),"sd":statistics.stdev(vals) if len(vals)>1 else None,
                          "min":min(vals),"max":max(vals),"individual_training_seed_means":vals}
        final.append(entry)
    return {"seed_level_means":seed_means,"across_training_seed_summary":final,
            "uncertainty":"n=3 training seeds is small; report SD/range and individual points. Repeated evaluation seeds do not increase training n. No narrow confidence claim from all-zero samples."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", nargs="*", type=int)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    all_records = []
    for meta in sorted(args.batch_root.rglob("metadata.json")):
        run = meta.parent
        config = json.loads(meta.read_text())
        for model in sorted((run / "frozen").glob("step*.pt")):
            step = int(model.stem[4:])
            if step not in config["protocol"]["checkpoint_actions"]:
                continue
            if args.steps and step not in args.steps:
                continue
            saved = torch.load(model, map_location="cpu", weights_only=True)
            if saved["global_step"] != step or saved["seed"] != config["seed"]:
                raise ValueError("Checkpoint metadata does not match its declared step/seed")
            if saved["training_backend"] != "shared" or saved["config"]["variant"] != config["variant"]:
                raise ValueError("Unexpected checkpoint training contract")
            label = f"{config['variant']}_s{config['seed']}_step{step}"
            out = args.out / label
            if out.exists():
                raise ValueError("Duplicate variant/training-seed/checkpoint; choose one declared run per seed")
            before = hash_file(model)
            evaluate.main(["--config","Configs/dqn_extended_development.yaml","--split","dev",
                "--agent","agents.dqn_agent:DQNPolicy","--agent-version",label,
                "--checkpoint-step",str(step),"--agent-kwargs",json.dumps({"checkpoint":str(model),"device":args.device}),
                "--out",str(out),"--save-traj"])
            for line in (out / "episodes.jsonl").read_text().splitlines():
                r = json.loads(line)
                assert r["scorecard_runs"]==1 and r["full_resets_after_start"]==0 and r["illegal_actions"]==0
                assert r["score_official"] is not None
                assert abs(r["score_official"]-r["score_recomputed"]) < 1e-6
                trajectory = [json.loads(x) for x in Path(r["trajectory_file"]).read_text().splitlines()]
                counts = Counter(x["action"] for x in trajectory)
                r.update(variant=config["variant"],training_seed=config["seed"],checkpoint_step=step,
                    model_sha256=before,action_counts=dict(counts),
                    action_entropy_nats=-sum((n/len(trajectory))*math.log(n/len(trajectory)) for n in counts.values()))
                all_records.append(r)
            assert hash_file(model)==before
    # One matched random reference, not one per model.
    evaluate.main(["--config","Configs/dqn_extended_development.yaml","--agent","random",
                   "--split","dev","--out",str(args.out / "random"),"--save-traj"])
    json_write(args.out / "records.json",all_records)
    json_write(args.out / "training_seed_aggregate.json",aggregate(all_records))
    assert all_records, "No declared fixed evaluation checkpoints were found"
    json_write(args.out / "evaluation_validation.json",{"valid":True,"model_game_seed_runs":len(all_records),
        "scope":"declared development games only; official score/reset/action invariants passed",
        "reserved_heldout_evaluated":False})


if __name__ == "__main__":
    main()
