"""Curate an audited DQN archive; keep binaries in ignored artifact storage.

Run from the repo root after audit_dqn_503808_archive. All destinations are new;
original archives and historical checkpoints are read-only.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from Experiments.audit_dqn_503808_archive import FINAL, GAMES, SEEDS, VARIANTS


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f: json.dump(data, f, indent=2); f.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_member(z, member, dest, expected):
    data = z.read(member)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Archive member differs from the verified audit: " + member)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as f: f.write(data)


def training_bins(path, variant, seed, bin_size=5000):
    totals = {g: dict(actions=0, task_reward=0., intrinsic_reward=0., wins=0, gameovers=0) for g in GAMES}
    bins = []
    cumulative_task = cumulative_intrinsic = 0.
    cumulative_wins = cumulative_gameovers = updates = 0
    prefix = None
    interval = defaultdict(float)
    start = 1
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["global_step"])
            task, intrinsic = float(row["task_reward"]), float(row["intrinsic_reward"])
            win, over = row["state"] == "WIN", row["state"] == "GAME_OVER"
            game = totals[row["game"]]
            game["actions"] += 1
            game["task_reward"] += task
            game["intrinsic_reward"] += intrinsic
            game["wins"] += win
            game["gameovers"] += over
            cumulative_task += task
            cumulative_intrinsic += intrinsic
            cumulative_wins += win
            cumulative_gameovers += over
            interval["task"] += task
            interval["intrinsic"] += intrinsic
            if row["loss"]:
                interval["loss_sum"] += float(row["loss"])
                interval["updates"] += 1
                updates += 1
            if step == 450000: prefix = {g: dict(v) for g, v in totals.items()}
            if step % bin_size == 0 or step == FINAL:
                bins.append(dict(variant=variant, training_seed=seed, start_action=start,
                    end_action=step, selected_actions=step-start+1,
                    interval_task_reward=interval["task"], cumulative_task_reward=cumulative_task,
                    interval_intrinsic_reward=interval["intrinsic"], cumulative_intrinsic_reward=cumulative_intrinsic,
                    cumulative_wins=cumulative_wins, cumulative_gameovers=cumulative_gameovers,
                    gradient_updates_in_bin=int(interval["updates"]), cumulative_gradient_updates=updates,
                    mean_training_loss=(interval["loss_sum"]/interval["updates"] if interval["updates"] else "")))
                interval.clear()
                start = step+1
    if prefix is None or step != FINAL or updates != 100562:
        raise ValueError("Binning requires the full independently audited history")
    return bins, prefix, totals


def curate(archive, audit_path, repo, out, raw):
    audit = json.loads(audit_path.read_text())
    if (not audit.get("ready_for_full_matrix_import") or audit["errors"] or
            len(audit["archives"]) != 1 or not all(audit["local_training_source_matches"].values())):
        raise ValueError("A complete successful single-archive audit is required")
    with archive.open("rb") as f: digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != audit["archives"][0]["sha256"]:
        raise ValueError("Archive checksum differs from audit")
    expected_raw_parent = (repo / "artifacts/dqn_extended").resolve()
    if not raw.resolve().is_relative_to(expected_raw_parent):
        raise ValueError("Binary destination must be within ignored artifacts/dqn_extended")
    if out.exists() or raw.exists(): raise FileExistsError("Choose unused output and artifact destinations")
    for name in ("dqn_503808_training.json", "dqn_503808_checkpoints.json"):
        if (repo / "Configs" / name).exists(): raise FileExistsError(name)
    cfg = audit["protocol"]
    inventory = audit["inventory"]
    root = audit["result_root"]
    models, comparison_models, by_run, by_game, curves = [], [], [], [], []
    with zipfile.ZipFile(archive) as z:
        def load(name): return json.loads(z.read(root+"/"+name))
        provenance = load("EXTENSION_PROVENANCE.json")
        old_cfg = provenance["old_protocol"]
        changed_keys = sorted(k for k in set(old_cfg) | set(cfg) if old_cfg.get(k) != cfg.get(k))
        if changed_keys != ["checkpoint_actions", "protocol_id", "total_actions"] or provenance["learning_hyperparameters_changed"]:
            raise ValueError("Continuation changed more than the declared budget keys")
        source_jobs = {(r["variant"], r["seed"]): r for r in provenance["source_jobs"]}
        if set(source_jobs) != {(v,s) for v in VARIANTS for s in SEEDS}:
            raise ValueError("Incomplete or duplicate source-job provenance")
        out.mkdir(parents=True, exist_ok=False)
        raw.mkdir(parents=True, exist_ok=False)
        for name in ("QUEUE.json", "extension_config.json", "EXTENSION_PROVENANCE.json", "progress.json"):
            copy_member(z, root+"/"+name, raw / "source" / name, inventory[name]["sha256"])
        for variant in VARIANTS:
            for seed in SEEDS:
                print(f"Importing and summarising {variant} seed{seed}", flush=True)
                base = f"{variant}/seed{seed}/"
                stage = raw / "evaluation_stage" / f"{variant}_seed{seed}"
                copy_member(z, root+"/"+base+"metadata.json", stage/"metadata.json", inventory[base+"metadata.json"]["sha256"])
                receipt = load(base+"RUN_COMPLETE.json")
                old_receipt = load(base+"source_450k/RUN_COMPLETE.json")
                source = source_jobs[(variant,seed)]
                original_frozen = {int(r["step"]): r["sha256"] for r in source["frozen"]}
                for step in cfg["checkpoint_actions"]:
                    name = base+f"frozen/step{step:06d}.pt"
                    expected = receipt["snapshots"][str(step)]["sha256"]
                    if inventory[name]["sha256"] != expected:
                        raise ValueError("Frozen receipt mismatch: " + name)
                    if step <= 450000 and (expected != old_receipt["snapshots"][str(step)]["sha256"] or expected != original_frozen[step]):
                        raise ValueError("An original frozen milestone changed: " + name)
                    if step not in (450000, FINAL): continue
                    dest = stage / "frozen" / f"step{step:06d}.pt"
                    copy_member(z, root+"/"+name, dest, expected)
                    entry = dict(variant=variant, training_seed=seed, checkpoint_step=step,
                        selected_training_actions=step, per_training_game_actions=step//3,
                        gradient_steps=89801 if step==450000 else 100562,
                        num_frames=4 if variant in ("v2", "v2_history_only") else 1,
                        novelty_beta=.01 if variant in ("v1", "v2") else 0.,
                        path=dest.relative_to(repo).as_posix(), sha256=expected,
                        bytes=inventory[name]["bytes"], archive_member=root+"/"+name)
                    (models if step==FINAL else comparison_models).append(entry)
                history = raw / "histories" / f"{variant}_seed{seed}.csv.gz"
                name = base+"actions.csv.gz"
                copy_member(z, root+"/"+name, history, inventory[name]["sha256"])
                bins, prefix, final = training_bins(history, variant, seed)
                curves.extend(bins)
                old_summary = load(base+"source_450k/summary.json")
                summary = load(base+"summary.json")
                for old_game in old_summary["per_game"]:
                    for key in ("actions", "task_reward", "intrinsic_reward", "wins", "gameovers"):
                        if not math.isclose(prefix[old_game["game"]][key], old_game[key], rel_tol=1e-10, abs_tol=1e-9):
                            raise ValueError(f"450K history prefix differs from original summary: {base} {key}")
                for game in GAMES:
                    p, f = prefix[game], final[game]
                    by_game.append(dict(variant=variant, training_seed=seed, game=game,
                        actions_450k=p["actions"], actions_503808=f["actions"],
                        task_reward_450k=p["task_reward"], task_reward_503808=f["task_reward"],
                        extension_task_reward=f["task_reward"]-p["task_reward"],
                        intrinsic_reward_450k=p["intrinsic_reward"], intrinsic_reward_503808=f["intrinsic_reward"],
                        wins_450k=p["wins"], wins_503808=f["wins"],
                        gameovers_450k=p["gameovers"], gameovers_503808=f["gameovers"]))
                by_run.append(dict(variant=variant, training_seed=seed, actions=FINAL,
                    gradient_steps=100562, task_reward_450k=sum(r["task_reward"] for r in prefix.values()),
                    task_reward_503808=sum(r["task_reward"] for r in final.values()),
                    extension_task_reward=sum(final[g]["task_reward"]-prefix[g]["task_reward"] for g in GAMES),
                    wins_503808=sum(r["wins"] for r in final.values()),
                    original_450k_torch=source["software"]["torch"], extension_torch=summary["software"]["torch"],
                    extension_actions=summary["new_actions"], extension_seconds=summary["loop_seconds_including_persistence"],
                    extension_actions_per_second=summary["actions_per_second_including_persistence"],
                    final_model_sha256=models[-1]["sha256"]))
        write_csv(out/"training_by_run.csv", by_run)
        write_csv(out/"training_by_game.csv", by_game)
        write_csv(out/"training_binned_5000.csv", curves)
        write_json(out/"continuation_validation.json", dict(valid=True,
            all_72_original_frozen_milestone_hashes_preserved=True,
            all_12_original_450k_summary_prefixes_match=True,
            changed_protocol_keys=changed_keys, actions_per_model=FINAL,
            total_matrix_actions=12*FINAL, additional_matrix_actions=12*(FINAL-450000),
            historical_bitwise_equivalence=False, frozen_evaluation_status="not yet performed",
            extension_gpu=provenance["bindings"]["hardware"]["gpu"],
            extension_software=provenance["bindings"]["hardware"]["software"],
            numerical_migration_policy=provenance["numerical_migration_policy"]["policy"],
            artifact_archive_sha256=digest, original_archive_modified=False,
            validation_scope="Archive integrity, learner metadata and recorded history; no environment reconstruction executed"))
        write_json(repo/"Configs/dqn_503808_training.json", cfg)
        write_json(repo/"Configs/dqn_503808_checkpoints.json", dict(
            format="dqn_extended_503808_frozen_registry_v1", training_budget_per_model=FINAL,
            checkpoint_selection="All twelve final models, plus their paired450K endpoints; no score-based selection",
            archive=dict(name=archive.name, sha256=digest, bytes=archive.stat().st_size,
                         artifact_url=None, location_note="Retain original ZIP in backed-up Drive storage; add a verified shared URL before publication"),
            ignored_evaluation_staging=(raw/"evaluation_stage").relative_to(repo).as_posix(),
            audit=audit_path.relative_to(repo).as_posix(), models=models, comparison_models=comparison_models,
            independent_training_seeds=list(SEEDS), evaluation_status="not yet performed",
            historical_bitwise_equivalence=False))
    return dict(models=len(models), paired_450k_models=len(comparison_models),
                run_rows=len(by_run), game_rows=len(by_game), curve_rows=len(curves),
                curated_results=out.relative_to(repo).as_posix(),
                ignored_staging=(raw/"evaluation_stage").relative_to(repo).as_posix())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]
    result=curate(args.archive.resolve(), args.audit.resolve(), repo, args.out.resolve(), args.raw.resolve())
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
