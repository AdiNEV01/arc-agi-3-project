"""Read-only audit of one or more Google Drive result ZIP parts.

No extraction, training, evaluation, network access or checkpoint conversion.
Missing files produce an incomplete audit, never a training-failure claim.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import zipfile

VARIANTS = ("v0", "v1", "v2_history_only", "v2")
SEEDS = (42, 123, 456)
FINAL = 503808
GAMES = ("ls20-9607627b", "ar25-0c556536", "bp35-0a0ad940")


def sha(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_history(stream, cfg, variant):
    """Validate every action; retain separate task and novelty totals."""
    games = {g: dict(actions=0, task_reward=0., intrinsic_reward=0., wins=0,
                     gameovers=0, extension_task_reward=0.) for g in GAMES}
    losses = 0
    count = 0
    with gzip.GzipFile(fileobj=stream) as gz, io.TextIOWrapper(gz, newline="") as text:
        for count, row in enumerate(csv.DictReader(text), 1):
            expected_game = GAMES[(count - 1) % len(GAMES)]
            if int(row["global_step"]) != count or row["game"] != expected_game:
                raise ValueError(f"Action sequence differs at {count}")
            if row["truncated"] != "False":
                raise ValueError(f"Artificial truncation at {count}")
            game = games[expected_game]
            game["actions"] += 1
            if int(row["game_actions"]) != game["actions"]:
                raise ValueError(f"Per-game counter differs at {count}")
            epsilon = cfg["epsilon_start"] + min(1., (count-1)/cfg["epsilon_decay"]) * (cfg["epsilon_end"]-cfg["epsilon_start"])
            if not math.isclose(float(row["epsilon"]), epsilon, abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Epsilon differs at {count}")
            task, intrinsic = float(row["task_reward"]), float(row["intrinsic_reward"])
            if not all(math.isfinite(x) and x >= 0 for x in (task, intrinsic)):
                raise ValueError(f"Invalid reward at {count}")
            if variant in ("v0", "v2_history_only") and intrinsic != 0:
                raise ValueError(f"Unexpected novelty reward at {count}")
            game["task_reward"] += task
            game["intrinsic_reward"] += intrinsic
            game["extension_task_reward"] += task if count > 450000 else 0
            game["wins"] += row["state"] == "WIN"
            game["gameovers"] += row["state"] == "GAME_OVER"
            update = count >= cfg["warmup"] and count % cfg["train_every"] == 0
            if bool(row["loss"]) != update:
                raise ValueError(f"Gradient schedule differs at {count}")
            if update:
                if not math.isfinite(float(row["loss"])):
                    raise ValueError(f"Nonfinite loss at {count}")
                losses += 1
    if count != cfg["total_actions"]:
        raise ValueError(f"Incomplete history: {count} actions")
    return dict(actions=count, gradient_updates=losses, per_game=games,
                finite_losses=True, round_robin_verified=True, epsilon_verified=True,
                interpretation="Exploratory training history; not frozen evaluation performance")


def check_frozen(data, variant, seed, protocol):
    import torch
    from DQN.q_network import QNetwork
    saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    frames = 4 if variant in ("v2", "v2_history_only") else 1
    cfg = saved["config"]
    beta = .01 if variant in ("v1", "v2") else 0.
    if any(cfg.get(k) != v for k, v in protocol.items() if k != "novelty_beta"):
        raise ValueError("Final model hyperparameters differ from the declared protocol")
    if (saved["global_step"] != FINAL or saved["gradient_steps"] != 100562 or
            saved["seed"] != seed or cfg["variant"] != variant or
            cfg["num_frames"] != frames or cfg["novelty_beta"] != beta or
            cfg["total_actions"] != FINAL or saved["training_backend"] != "shared"):
        raise ValueError("Final model contract differs")
    model = QNetwork(num_frames=frames)
    for key in ("online_state_dict", "target_state_dict"):
        if not all(torch.isfinite(x).all().item() for x in saved[key].values()):
            raise ValueError("Nonfinite model weights")
        model.load_state_dict(saved[key], strict=True)
    model.load_state_dict(saved["online_state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 16*frames, 64, 64))
    if output.shape != (1, 4102) or not torch.isfinite(output).all().item():
        raise ValueError("Invalid model output")
    return dict(global_step=FINAL, gradient_steps=100562, num_frames=frames,
                novelty_beta=beta, finite_online_and_target=True,
                strict_cpu_model_load=True, finite_zero_input_forward=True,
                training_seconds_recorded=saved.get("training_seconds"))


def audit(archives, repo, progress=None):
    progress = progress or (lambda message: None)
    report = dict(format="dqn_503808_archive_audit_v1", archives=[], jobs=[], errors=[],
                  missing_for_matrix_import=[], frozen_evaluation_performed=False,
                  training_or_restoration_performed=False,
                  original_files_modified=False)
    with ExitStack() as stack:
        members, inventory, root_name = {}, {}, None
        for path in archives:
            path = Path(path)
            with path.open("rb") as f: archive_sha = sha(f)
            z = stack.enter_context(zipfile.ZipFile(path))
            file_count = 0
            for info in z.infolist():
                if info.is_dir(): continue
                full = PurePosixPath(info.filename)
                if (full.is_absolute() or ".." in full.parts or "\\" in info.filename or
                        PureWindowsPath(info.filename).drive or len(full.parts) < 2):
                    raise ValueError(f"Unsafe or unexpected archive path: {info.filename}")
                if root_name is None: root_name = full.parts[0]
                if full.parts[0] != root_name: raise ValueError("Mixed result roots")
                name = PurePosixPath(*full.parts[1:]).as_posix()
                with z.open(info) as f: digest = sha(f)  # Also verifies ZIP CRC.
                if name in inventory and inventory[name]["sha256"] != digest:
                    raise ValueError(f"Conflicting duplicate across ZIP parts: {name}")
                members[name] = (z, info)
                inventory[name] = dict(bytes=info.file_size, sha256=digest)
                file_count += 1
            report["archives"].append(dict(name=path.name, bytes=path.stat().st_size,
                                           sha256=archive_sha, files=file_count, crc_verified=True))
            progress(f"Verified CRC/SHA256 of {file_count} members: {path.name}")
        def read(name):
            z, info = members[name]
            return z.read(info)
        def load(name): return json.loads(read(name))
        cfg = load("QUEUE.json")["protocol"]
        if (cfg["total_actions"] != FINAL or tuple(cfg["variants"]) != VARIANTS or
                tuple(cfg["seeds"]) != SEEDS or tuple(cfg["training_games"]) != GAMES):
            raise ValueError("Unexpected matrix protocol")
        report.update(result_root=root_name, inventory=inventory, protocol=cfg)
        queue_progress = load("progress.json")
        report["producer_claims_all_12_complete"] = (queue_progress["status"] == "complete" and
            len(queue_progress["jobs"]) == 12 and all(j["status"] == "complete" and
            j["completed_actions"] == FINAL for j in queue_progress["jobs"].values()))
        for name in ("EXTENSION_PROVENANCE.json", "extension_config.json"):
            if name not in members: report["missing_for_matrix_import"].append(name)
        queue = load("QUEUE.json")
        report["local_training_source_matches"] = {}
        for name, digest in {**queue["source_sha256"], **queue["queue_sources"]}.items():
            path = repo / name
            actual = None
            if path.is_file():
                with path.open("rb") as f: actual = sha(f)
            report["local_training_source_matches"][name] = actual == digest
        report["recovery_receipt_checks"] = []
        for name in sorted(members):
            if not name.endswith("/COMPLETE.json"): continue
            receipt = load(name)
            parent = str(PurePosixPath(name).parent)
            for leaf, digest in receipt["sha256"].items():
                target = parent + "/" + leaf
                present = target in inventory
                match = inventory[target]["sha256"] == digest if present else None
                report["recovery_receipt_checks"].append(dict(path=target, present=present, checksum_matches=match))
                if match is False: report["errors"].append("Recovery checksum mismatch: " + target)
        for variant in VARIANTS:
            for seed in SEEDS:
                progress(f"Checking {variant} seed{seed}: final model and complete training history")
                base = f"{variant}/seed{seed}/"
                job = dict(variant=variant, seed=seed)
                report["jobs"].append(job)
                metadata_name = base + "metadata.json"
                if metadata_name not in members:
                    report["missing_for_matrix_import"].append(metadata_name)
                else:
                    metadata = load(metadata_name)
                    if metadata["protocol"] != cfg or metadata["variant"] != variant or metadata["seed"] != seed:
                        report["errors"].append("Metadata contract differs: " + metadata_name)
                if base + "frozen/step450000.pt" not in members:
                    report["missing_for_matrix_import"].append(base + "frozen/step450000.pt")
                receipt_name = base + "RUN_COMPLETE.json"
                receipt = load(receipt_name) if receipt_name in members else None
                job["completion_receipt_present"] = receipt is not None
                if receipt is None: report["missing_for_matrix_import"].append(receipt_name)
                else:
                    valid = (receipt["validated"] and receipt["actions"] == FINAL and
                             receipt["gradient_steps"] == 100562 and receipt["variant"] == variant and
                             receipt["seed"] == seed and receipt["per_game_actions"] == dict.fromkeys(GAMES, 167936))
                    job["completion_receipt_contract_valid"] = valid
                    if not valid: report["errors"].append("Invalid completion receipt: " + base)
                finals = sorted(n for n in members if n.startswith(base) and n.endswith("/frozen/step503808.pt"))
                # Both primary and attempt copies have the same byte identity.
                if len({inventory[n]["sha256"] for n in finals}) > 1:
                    report["errors"].append("Final model copies disagree: " + base)
                final = base + "frozen/step503808.pt"
                if final not in members: final = finals[0] if finals else None
                job["final_model_path"] = final
                if final is None: report["missing_for_matrix_import"].append(base + "frozen/step503808.pt")
                else:
                    try:
                        job["final_model_validation"] = check_frozen(read(final), variant, seed, cfg)
                        job["final_model_sha256"] = inventory[final]["sha256"]
                    except Exception as exc: report["errors"].append(f"{final}: {type(exc).__name__}: {exc}")
                history = base + "actions.csv.gz"
                histories = sorted(n for n in members if n.startswith(base) and n.endswith("/step503808/history.csv.gz"))
                if history not in members: history = histories[0] if histories else None
                job["full_history_path"] = history
                if history is None: report["missing_for_matrix_import"].append(base + "actions.csv.gz")
                else:
                    try:
                        z, info = members[history]
                        with z.open(info) as f: job["history_validation"] = validate_history(f, cfg, variant)
                    except Exception as exc: report["errors"].append(f"{history}: {type(exc).__name__}: {exc}")
                if receipt:
                    job["frozen_snapshot_checksum_checks"] = []
                    for step, record in receipt["snapshots"].items():
                        name = base + f"frozen/step{int(step):06d}.pt"
                        if int(step) == FINAL and name not in inventory: name = final or name
                        if name in inventory:
                            match = inventory[name]["sha256"] == record["sha256"]
                            job["frozen_snapshot_checksum_checks"].append(dict(path=name, matches=match))
                            if not match: report["errors"].append("Frozen checksum mismatch: " + name)
                        elif int(step) in (450000, FINAL): report["missing_for_matrix_import"].append(name)
                summary_name = base + "summary.json"
                summaries = sorted(n for n in members if n.startswith(base+"attempts/") and n.endswith("/summary.json"))
                if summary_name not in members: summary_name = summaries[0] if summaries else None
                if summary_name:
                    summary = load(summary_name)
                    job["producer_summary"] = {k: summary[k] for k in ("global_step", "initial_step", "new_actions", "gradient_steps", "finite_losses", "new_updates", "device_name", "software", "actions_per_second_including_persistence")}
                    job["producer_summary"]["runtime_scope"] = "53,808-action continuation only; not total training time"
                    if "history_validation" in job:
                        for game in summary["per_game"]:
                            observed = job["history_validation"]["per_game"][game["game"]]
                            for key in ("actions", "task_reward", "intrinsic_reward", "wins", "gameovers"):
                                if not math.isclose(observed[key], game[key], rel_tol=1e-10, abs_tol=1e-9):
                                    report["errors"].append(f"Summary/history differs: {base}{game['game']} {key}")
        report["counts"] = dict(final_models_present=sum(j["final_model_path"] is not None for j in report["jobs"]),
            final_models_cpu_validated=sum("final_model_validation" in j for j in report["jobs"]),
            full_histories_validated=sum("history_validation" in j for j in report["jobs"]),
            completion_receipts_present=sum(j["completion_receipt_present"] for j in report["jobs"]))
        report["ready_for_full_matrix_import"] = (not report["missing_for_matrix_import"] and not report["errors"] and
            all(report["local_training_source_matches"].values()) and
            report["counts"]["final_models_cpu_validated"] == 12 and report["counts"]["full_histories_validated"] == 12)
        import torch
        report["audit_runtime"] = dict(python=platform.python_version(), torch=str(torch.__version__), device="cpu")
        report["limitations"] = ["Producer completion claims are distinct from independently available payloads.",
            "CPU loading is not a CUDA restoration or deterministic continuation test.",
            "Historical T4/cu130 to RTX3050/cu126 numerical migration remains a reproducibility limitation.",
            "No frozen evaluation was run; reserved games remain untouched."]
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archives", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)
    report = audit(args.archives, Path(__file__).resolve().parents[1],
                   progress=lambda message: print(message, flush=True))
    (args.out / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "producer_claims_all_12_complete", "ready_for_full_matrix_import", "errors")}, indent=2), flush=True)


if __name__ == "__main__": main()
