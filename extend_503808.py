"""Continue all completed 450K learners to PPO's actual 503,808 interactions.

Additive delivery for the existing cu126/numerical migration environment. The
original learner and all historical artifacts stay unchanged. A derived state
records an explicit budget/cap conversion, plus the already approved CUDA-build
migration for source runs that still carry original cu130 state.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
BASE = Path(__file__).resolve().parent
OLD_STEPS = 450000
FINAL_STEPS = 503808
CAP_KEYS = {"protocol_id", "total_actions", "checkpoint_actions"}

import cu126_recovery as recovery
import continue_numerical as numerical


def gradient_count(config, total):
    return max(0, total // config["train_every"] - (max(1, config["warmup"]) - 1) // config["train_every"])


def extension_config(original, target=FINAL_STEPS):
    if target <= original["total_actions"]:
        raise recovery.RecoveryStopped("The continuation target must exceed the original completed budget")
    result = json.loads(json.dumps(original))
    result.update(protocol_id=original["protocol_id"] + f"_continued_to_{target}", total_actions=target,
                  checkpoint_actions=sorted(set(original["checkpoint_actions"] + [target])))
    if {k for k in original if original[k] != result[k]} != CAP_KEYS:
        raise recovery.RecoveryStopped("Unexpected budget-extension configuration change")
    return result


def queue_definition(config):
    from DQN.extended_train import source_hashes, runtime_hashes
    return {"protocol": config, "source_sha256": source_hashes(),
            "runtime_assets": runtime_hashes(config["training_games"]),
            "queue_sources": {p: recovery.sha(BASE / p) for p in ("Experiments/full_queue.py",
                 "Experiments/full_preflight.py", "Experiments/drive_storage.py")}}


def verify_patch():
    source = numerical.verify_delivery()
    manifest = json.loads((BASE / "EXTENSION_503808_MANIFEST.json").read_text())
    if recovery.sha(BASE / "NUMERICAL_PATCH_MANIFEST.json") != manifest["numerical_patch_manifest_sha256"]:
        raise recovery.RecoveryStopped("The original numerical-continuation patch differs")
    for name, expected in manifest["sha256"].items():
        if recovery.sha(BASE / name) != expected:
            raise recovery.RecoveryStopped(f"Extension delivery changed: {name}")
    return source["protocol"]


def validate_history(checkpoint, config, total):
    games = Counter()
    count = 0
    with gzip.open(Path(checkpoint) / "history.csv.gz", "rt", newline="") as f:
        for count, row in enumerate(csv.DictReader(f), 1):
            expected = config["training_games"][(count - 1) % len(config["training_games"])]
            if int(row["global_step"]) != count or row["game"] != expected or row["truncated"] != "False":
                raise recovery.RecoveryStopped("Original committed action history violates the training contract")
            games[row["game"]] += 1
    if count != total:
        raise recovery.RecoveryStopped("Original full history does not contain the completed action budget")
    return dict(games)


def verified_completed_source(root, config):
    """Read-only verification; never call validate_complete on the source tree."""
    import torch
    from DQN.extended_train import check_resume
    from Experiments.full_queue import verified_generations
    root = Path(root)
    if json.loads((root / "QUEUE.json").read_text()) != queue_definition(config):
        raise recovery.RecoveryStopped("Source queue protocol/code/runtime assets differ")
    progress = json.loads((root / "progress.json").read_text())
    plan = [(v, s) for s in config["seeds"] for v in config["variants"]]
    if progress["status"] != "complete" or len(progress["jobs"]) != len(plan):
        raise recovery.RecoveryStopped("The entire original matrix must be complete before extending it")
    result = []
    for variant, seed in plan:
        key = f"{variant}_seed{seed}"
        entry = progress["jobs"][key]
        if entry["status"] != "complete" or entry["completed_actions"] != config["total_actions"]:
            raise recovery.RecoveryStopped(f"Original job is incomplete: {key}")
        job = root / variant / f"seed{seed}"
        generations, invalid = verified_generations(job, config, variant, seed)
        if invalid or not generations or generations[-1][0] != config["total_actions"]:
            raise recovery.RecoveryStopped(f"Missing or invalid final full learner state: {key}")
        checkpoint = generations[-1][1]
        state = check_resume(checkpoint)
        if state["gradient_steps"] != gradient_count(config, config["total_actions"]):
            raise recovery.RecoveryStopped(f"Wrong original gradient-update count: {key}")
        build = state["torch_version"]
        if build not in (recovery.OLD_TORCH, recovery.NEW_TORCH):
            raise recovery.RecoveryStopped(f"Unsupported original CUDA build: {key}")
        recovery.require_build(state, build)
        receipt = json.loads((job / "RUN_COMPLETE.json").read_text())
        if not receipt.get("validated") or receipt["actions"] != config["total_actions"]:
            raise recovery.RecoveryStopped(f"Missing validated source completion: {key}")
        per_game = validate_history(checkpoint, config, config["total_actions"])
        if receipt["per_game_actions"] != per_game:
            raise recovery.RecoveryStopped(f"Original per-game counts differ: {key}")
        if sum(int(action != -1) for _, action, _ in state["journal"]) != config["total_actions"]:
            raise recovery.RecoveryStopped(f"Original game transcript action count differs: {key}")
        frozen = []
        for step in config["checkpoint_actions"]:
            path = job / "frozen" / f"step{step:06d}.pt"
            expected = receipt["snapshots"][str(step)]["sha256"]
            if recovery.sha(path) != expected:
                raise recovery.RecoveryStopped(f"Original frozen model checksum differs: {path}")
            model = torch.load(path, map_location="cpu", weights_only=True)
            if model["global_step"] != step or model["seed"] != seed or model["config"]["variant"] != variant:
                raise recovery.RecoveryStopped(f"Original frozen model identity differs: {path}")
            if not all(torch.isfinite(t).all().item() for name in ("online_state_dict", "target_state_dict") for t in model[name].values()):
                raise recovery.RecoveryStopped(f"Nonfinite original frozen model: {path}")
            if step == config["total_actions"]:
                for name in ("online_state_dict", "target_state_dict"):
                    recovery.assert_bits(state[name], model[name], name)
            frozen.append({"path": str(path), "sha256": expected, "step": step})
        result.append({"variant": variant, "seed": seed, "checkpoint": str(checkpoint),
                       "complete_sha256": recovery.sha(checkpoint / "COMPLETE.json"),
                       "software": state["software"], "frozen": frozen, "per_game_actions": per_game})
        print(f"VERIFIED ORIGINAL COMPLETE {key}: {config['total_actions']} actions", flush=True)
    return result


def derive_checkpoint(source, destination, old_config, new_config, expected_software, cuda_version):
    """Only extend cap metadata and, explicitly, the authorised CUDA build tag."""
    import torch
    from DQN.extended_train import check_resume
    source, destination = Path(source), Path(destination)
    if destination.resolve().is_relative_to(source.resolve()):
        raise recovery.RecoveryStopped("Continuation output must be outside the original checkpoint")
    original = check_resume(source)
    if original["protocol"] != old_config or original["global_step"] != old_config["total_actions"]:
        raise recovery.RecoveryStopped("Extension must start from the original final full checkpoint")
    if original["software"] != expected_software:
        if (original["torch_version"], expected_software["torch"]) != (recovery.OLD_TORCH, recovery.NEW_TORCH):
            raise recovery.RecoveryStopped("Only the approved cu130-to-cu126 software change is allowed")
        for key in original["software"]:
            if key != "torch" and original["software"][key] != expected_software[key]:
                raise recovery.RecoveryStopped(f"Unapproved software change: {key}")
    permitted = extension_config(old_config, new_config["total_actions"])
    if new_config != permitted:
        raise recovery.RecoveryStopped("Extension changes a training rule beyond the budget and snapshot schedule")
    expected_config = {**old_config, **{k: original[k] for k in ("num_frames", "variant", "novelty_beta", "device")}}
    if original["config"] != expected_config:
        raise recovery.RecoveryStopped("Original model configuration and protocol disagree")
    provenance = {"kind": "explicit_completed_budget_extension", "original_checkpoint": str(source.resolve()),
        "original_complete": json.loads((source / "COMPLETE.json").read_text()),
        "original_software": original["software"], "target_software": expected_software,
        "original_protocol": old_config, "extended_protocol": new_config,
        "configuration_fields_changed": sorted(CAP_KEYS),
        "cuda_build_change": original["software"] != expected_software,
        "numerical_migration_authorised": True, "cross_hardware_bitwise_equivalence_claimed": False,
        "reset_model_optimizer_replay_rng_history_or_novelty": False}
    state = {**original, "protocol": new_config,
             "config": {**original["config"], **{k: new_config[k] for k in CAP_KEYS}},
             "software": expected_software, "torch_version": expected_software["torch"],
             "cuda_version": cuda_version, "budget_extension": provenance}
    for key in recovery.STATE_FIELDS:
        if key not in ("protocol", "config"):
            recovery.assert_bits(original[key], state[key], key)
    destination.mkdir(parents=True, exist_ok=False)
    torch.save(state, destination / "state.pt")
    for name in ("replay.npz", "history.csv.gz"):
        shutil.copy2(source / name, destination / name)
        if recovery.sha(source / name) != recovery.sha(destination / name):
            raise recovery.RecoveryStopped(f"Continuation input copy differs: {name}")
    recovery.write_json(destination / "BUDGET_EXTENSION.json", provenance)
    recovery.write_json(destination / "COMPLETE.json", {"step": original["global_step"], "sha256": {
        name: recovery.sha(destination / name) for name in ("state.pt", "replay.npz", "history.csv.gz")}})
    restored = check_resume(destination)
    for key in recovery.STATE_FIELDS:
        if key not in ("protocol", "config"):
            recovery.assert_bits(original[key], restored[key], key)
    if restored["protocol"] != new_config or restored["config"] != state["config"]:
        raise recovery.RecoveryStopped("Extended budget metadata was not saved correctly")
    return provenance


def prepare_result_tree(source, jobs, old_config, new_config, gpu, root):
    from DQN.extended_train import check_resume
    from Experiments.full_queue import atomic_copy
    source, root = Path(source).resolve(), Path(root).resolve()
    if root.is_relative_to(source):
        raise recovery.RecoveryStopped("The new result tree must be outside the original 450K results")
    estimated = sum(sum(p.stat().st_size for p in Path(j["checkpoint"]).iterdir() if p.is_file()) +
                    sum(Path(p["path"]).stat().st_size for p in j["frozen"]) for j in jobs)
    if shutil.disk_usage(root.parent).free < estimated + 2*2**30:
        raise recovery.RecoveryStopped("Insufficient disk for separate continuation states plus a 2GiB reserve")
    root.mkdir(parents=True, exist_ok=False)
    recovery.write_json(root / "extension_config.json", new_config)
    recovery.write_json(root / "QUEUE.json", queue_definition(new_config))
    for name in ("QUEUE.json", "progress.json"):
        target = root / "extension_provenance" / ("SOURCE_450K_" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(source / name, target)
    prior_migration = source / "numerical_migration"
    if prior_migration.exists(): shutil.copytree(prior_migration, root / "extension_provenance/numerical_migration_450k")
    imported = []
    entries = {}
    for item in jobs:
        variant, seed = item["variant"], item["seed"]
        job = root / variant / f"seed{seed}"
        attempt = job / "attempts" / ("budget_extension_" + uuid.uuid4().hex[:10])
        checkpoint = attempt / "resume" / f"step{old_config['total_actions']:06d}"
        event = derive_checkpoint(item["checkpoint"], checkpoint, old_config, new_config, gpu["software"], gpu["cuda_runtime"])
        state = check_resume(checkpoint)
        names = ("protocol", "variant", "seed", "num_frames", "novelty_beta", "device", "torch_version",
                 "cuda_version", "source_sha256", "platform", "runtime_assets", "software", "checkpoint_kind")
        recovery.write_json(attempt / "metadata.json", {**{k: state[k] for k in names},
            "parent_resume": item["checkpoint"], "initial_step": old_config["total_actions"], "budget_extension": event})
        for frozen in item["frozen"]:
            atomic_copy(Path(frozen["path"]), job / "frozen" / Path(frozen["path"]).name, frozen["sha256"])
        old_job = source / variant / f"seed{seed}"
        for name in ("RUN_COMPLETE.json", "summary.json", "metadata.json"):
            if (old_job / name).exists(): atomic_copy(old_job / name, job / "source_450k" / name)
        key = f"{variant}_seed{seed}"
        entries[key] = {"configuration": variant, "seed": seed, "status": "pending",
                        "completed_actions": old_config["total_actions"], "target_actions": new_config["total_actions"],
                        "latest_verified_checkpoint": str(checkpoint)}
        imported.append({**item, "derived_checkpoint": str(checkpoint),
                         "derived_complete_sha256": recovery.sha(checkpoint / "COMPLETE.json")})
        print(f"READY TO EXTEND {key}: {old_config['total_actions']} -> {new_config['total_actions']}", flush=True)
    recovery.write_json(root / "progress.json", {"status": "ready", "protocol_id": new_config["protocol_id"],
        "target_actions_each": new_config["total_actions"], "remaining_experiments": len(entries), "jobs": entries})
    return imported


def extension_command(config, config_path, variant, seed, run, resume, persistent):
    from DQN.extended_train import check_resume
    if resume is None:
        raise recovery.RecoveryStopped("A completed learner checkpoint is required; extension never starts at zero")
    state = check_resume(Path(resume))
    recovery.require_build(state, recovery.NEW_TORCH)
    if state["protocol"] != config or not OLD_STEPS <= state["global_step"] < FINAL_STEPS:
        raise recovery.RecoveryStopped("Continuation input has the wrong protocol or saved step")
    return [sys.executable, "-u", "-m", "DQN.extended_train", "--config", str(config_path),
            "--variant", variant, "--seed", str(seed), "--device", "cuda", "--run-dir", str(run),
            "--stop-at", str(FINAL_STEPS), "--persistent-root", str(persistent), "--resume", str(resume)]


def status():
    pointer = BASE / "EXTENSION_503808_ACTIVE.json"
    if not pointer.exists():
        print("503808 continuation has not been initialised in this directory."); return
    root = Path(json.loads(pointer.read_text())["root"])
    progress = json.loads((root / "progress.json").read_text())
    print("503808 RESULTS FOLDER:", root)
    print("OVERALL STATUS:", progress["status"])
    done = True
    for name, entry in progress["jobs"].items():
        step = entry.get("completed_actions", 0)
        print(name, entry["status"], f"{step}/{FINAL_STEPS}")
        done &= entry["status"] == "complete" and step == FINAL_STEPS
    print("ALL 12 COMPLETE:", bool(done and len(progress["jobs"]) == 12 and progress["status"] == "complete"))


def run(args, audit, report):
    import torch
    from Experiments.full_queue import run_queue
    old_config = verify_patch()
    if old_config["total_actions"] != OLD_STEPS:
        raise recovery.RecoveryStopped("Expected the original 450K training protocol")
    new_config = extension_config(old_config)
    gpu = recovery.gpu_check(old_config)
    binding = {"hardware": numerical.hardware_binding(gpu), "script_sha256": recovery.sha(Path(__file__)),
               "base_adapter_manifest_sha256": recovery.sha(BASE / "LINUX_ADAPTER_MANIFEST.json"),
               "numerical_patch_manifest_sha256": recovery.sha(BASE / "NUMERICAL_PATCH_MANIFEST.json")}
    pointer = BASE / "EXTENSION_503808_ACTIVE.json"
    if pointer.exists():
        active = json.loads(pointer.read_text())
        root = Path(active["root"])
        manifest_path = root / "EXTENSION_PROVENANCE.json"
        manifest = json.loads(manifest_path.read_text())
        if recovery.sha(manifest_path) != active["manifest_sha256"] or manifest["bindings"] != binding:
            raise recovery.RecoveryStopped("Continuation provenance or verified environment changed")
        if manifest["new_protocol"] != new_config:
            raise recovery.RecoveryStopped("Continuation configuration differs from its fixed target")
        if args.source and Path(args.source).resolve() != Path(manifest["source_450k_root"]):
            raise recovery.RecoveryStopped("Source argument differs from this established extension")
        recovery.scan(root, new_config, recovery.NEW_TORCH)
    else:
        active = json.loads((BASE / "NUMERICAL_ACTIVE_RESULTS.json").read_text())
        proof_path = Path(active["proof"])
        proof = json.loads(proof_path.read_text())
        if recovery.sha(proof_path) != active["proof_sha256"] or not proof.get("restoration_verified"):
            raise recovery.RecoveryStopped("The prior numerical restoration proof is missing or changed")
        if proof["bindings"] != numerical.bindings(gpu):
            raise recovery.RecoveryStopped("The verified numerical environment has changed")
        if not proof["policy"].get("accepted"):
            raise recovery.RecoveryStopped("The numerical migration approval is missing")
        source = Path(args.source).resolve() if args.source else Path(active["root"])
        if source != Path(active["root"]):
            raise recovery.RecoveryStopped("Use the completed result folder bound to NUMERICAL_ACTIVE_RESULTS.json")
        jobs = verified_completed_source(source, old_config)
        root = BASE / ("DQN_503808_CONTINUED_" + audit.name)
        imported = prepare_result_tree(source, jobs, old_config, new_config, gpu, root)
        # Marker hashes capture the exact input states and detect a concurrent
        # source alteration without rewriting or relabelling any source file.
        if any(recovery.sha(Path(j["checkpoint"]) / "COMPLETE.json") != j["complete_sha256"] for j in jobs):
            raise recovery.RecoveryStopped("Original completion states changed during import")
        manifest = {"kind": "450000_to_503808_full_state_continuation", "source_450k_root": str(source),
                    "new_protocol": new_config, "old_protocol": old_config, "bindings": binding,
                    "source_jobs": imported, "additional_actions_per_model": FINAL_STEPS - OLD_STEPS,
                    "additional_actions_matrix": (FINAL_STEPS - OLD_STEPS) * len(imported),
                    "new_gradient_updates_per_model": gradient_count(new_config, FINAL_STEPS) - gradient_count(old_config, OLD_STEPS),
                    "numerical_migration_policy": proof["policy"],
                    "original_450k_files_modified": False, "learning_hyperparameters_changed": False,
                    "ppo_matching_scope": "Training interactions only; training rewards, action spaces and optimisation remain different"}
        manifest_path = root / "EXTENSION_PROVENANCE.json"
        recovery.write_json(manifest_path, manifest)
        recovery.scan(root, new_config, recovery.NEW_TORCH)
        recovery.write_json(pointer, {"root": str(root), "manifest_sha256": recovery.sha(manifest_path)})
    report.update(results_root=str(root), status="ready", target_actions_each=FINAL_STEPS)
    recovery.write_json(audit / "EXTENSION_REPORT.json", report)
    print("503808 RESULTS FOLDER:", root, flush=True)
    if args.verify_only:
        print("SOURCE AND CONTINUATION STATES VERIFIED; no extension training launched.", flush=True); return
    config_path = root / "extension_config.json"
    if json.loads(config_path.read_text()) != new_config:
        raise recovery.RecoveryStopped("Saved continuation configuration changed")
    def command(variant, seed, run, resume, persistent):
        if not torch.cuda.is_available(): raise recovery.RecoveryStopped("CUDA became unavailable")
        return extension_command(new_config, config_path, variant, seed, run, resume, persistent)
    report.update(status="continuing", queue_launch_requested=True)
    recovery.write_json(audit / "EXTENSION_REPORT.json", report)
    run_queue(new_config, root, BASE / "extension_503808_work", command_factory=command, require_cuda=False)
    report["status"] = "complete"
    recovery.write_json(audit / "EXTENSION_REPORT.json", report)
    print("ALL 12 EXTENDED RUNS COMPLETE AT 503808 TRAINING ACTIONS EACH.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    os.chdir(BASE)
    if args.status: status(); return
    audit = BASE / "extension_503808_audits" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    audit.mkdir(parents=True)
    report = {"status": "checking", "queue_launch_requested": False, "original_450k_files_modified": False}
    print("Extension report:", audit / "EXTENSION_REPORT.json", flush=True)
    try:
        if platform.system() != "Linux": raise recovery.RecoveryStopped("Run on the existing Linux RTX cu126 environment")
        from filelock import FileLock
        with FileLock(str(BASE / ".cu126_recovery.lock"), timeout=0): run(args, audit, report)
    except BaseException as exc:
        report.update(status="stopped", error=str(exc), exception=type(exc).__name__)
        recovery.write_json(audit / "EXTENSION_REPORT.json", report)
        print(f"STOPPED: {exc}\nReport: {audit / 'EXTENSION_REPORT.json'}", file=sys.stderr, flush=True)
        if isinstance(exc, KeyboardInterrupt): raise
        raise SystemExit(1)


if __name__ == "__main__": main()
