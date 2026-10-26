"""Read-only result collector: one compact archive for analysis/repo handoff.

Run from ~/Downloads/arc_dqn_cu126. No torch, training, network or Drive access.
Both original result trees remain intact. Full recovery payloads stay in backup.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
import zipfile

VARIANTS = ("v0", "v1", "v2_history_only", "v2")
SEEDS = (42, 123, 456)
FINAL = 503808


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(2**20), b""):
            h.update(data)
    return h.hexdigest()


def collect(workspace):
    work = Path(workspace).expanduser().resolve()
    root = Path(json.loads((work / "EXTENSION_503808_ACTIVE.json").read_text())["root"]).resolve()
    cfg = json.loads((root / "QUEUE.json").read_text())["protocol"]
    progress = json.loads((root / "progress.json").read_text())
    if cfg["total_actions"] != FINAL or tuple(cfg["variants"]) != VARIANTS or tuple(cfg["seeds"]) != SEEDS:
        raise ValueError("Expected the declared four-variant, three-seed503808 matrix")
    if progress["status"] != "complete" or len(progress["jobs"]) != 12:
        raise ValueError("The503808 matrix is not complete")
    files = {}
    def add(path, name=None):
        path = Path(path)
        if not path.is_file(): raise FileNotFoundError(path)
        if name is None:
            if not path.resolve().is_relative_to(root): raise ValueError("Unexpected path outside results")
            name = path.relative_to(root).as_posix()
        files["DQN_503808/" + name] = path
    for name in ("QUEUE.json", "progress.json", "extension_config.json", "EXTENSION_PROVENANCE.json"):
        add(root / name)
    for variant in VARIANTS:
        for seed in SEEDS:
            key = f"{variant}_seed{seed}"
            entry = progress["jobs"][key]
            if entry["status"] != "complete" or entry["completed_actions"] != FINAL:
                raise ValueError(f"Incomplete job: {key}")
            job = root / variant / f"seed{seed}"
            receipt = json.loads((job / "RUN_COMPLETE.json").read_text())
            if (not receipt.get("validated") or receipt["actions"] != FINAL or
                    receipt["variant"] != variant or receipt["seed"] != seed or receipt["gradient_steps"] != 100562):
                raise ValueError(f"Completion receipt differs: {key}")
            for name in ("RUN_COMPLETE.json", "metadata.json", "actions.csv.gz"):
                add(job / name)
            if (job / "summary.json").exists(): add(job / "summary.json")
            for step in (450000, FINAL):
                model = job / "frozen" / f"step{step:06d}.pt"
                if sha(model) != receipt["snapshots"][str(step)]["sha256"]:
                    raise ValueError(f"Frozen model checksum differs: {model}")
                add(model)
            markers = sorted(job.glob(f"attempts/*/resume/step{FINAL:06d}/COMPLETE.json"))
            if not markers: raise ValueError(f"No final recovery receipt: {key}")
            # Only the receipt is copied. state.pt and replay.npz are excluded.
            marker = markers[-1]
            record = json.loads(marker.read_text())
            if record["step"] != FINAL or sha(job / "actions.csv.gz") != record["sha256"]["history.csv.gz"]:
                raise ValueError(f"Canonical training history checksum differs: {key}")
            add(marker)
            for name in ("RUN_COMPLETE.json", "metadata.json", "summary.json"):
                old = job / "source_450k" / name
                if old.exists(): add(old)
    provenance = root / "extension_provenance"
    if provenance.exists():
        for p in sorted(provenance.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".txt", ".log"):
                add(p)
    for name in ("FULL_SOURCE_MANIFEST.json", "LINUX_ADAPTER_MANIFEST.json", "NUMERICAL_PATCH_MANIFEST.json", "EXTENSION_503808_MANIFEST.json"):
        add(work / name, "source_manifests/" + name)
    return root, files


def package(workspace, output_directory):
    root, files = collect(workspace)
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    output = output_directory / f"DQN_503808_HANDOFF_{stamp}.zip"
    inventory = {name: {"bytes": path.stat().st_size, "sha256": sha(path)} for name, path in files.items()}
    manifest = {"format": "completed_dqn_503808_handoff_v1", "original_result_root": str(root),
        "actions_per_model": FINAL, "models": 12, "included_model_steps": [450000, FINAL],
        "canonical_full_training_histories_included": True,
        "full_resume_state_and_replay_included": False, "originals_modified": False,
        "evaluation_status": "Training evidence; frozen performance evaluation is a separate stage",
        "files": inventory}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in files.items(): z.write(path, name)
        z.writestr("HANDOFF_MANIFEST.json", json.dumps(manifest, indent=2))
    # Compare archived bytes to the pre-copy inventory and original files.
    with zipfile.ZipFile(output) as z:
        for name, path in files.items():
            h = hashlib.sha256()
            with z.open(name) as f:
                for data in iter(lambda: f.read(2**20), b""): h.update(data)
            if h.hexdigest() != inventory[name]["sha256"] or sha(path) != inventory[name]["sha256"]:
                raise ValueError(f"Handoff or original changed during packaging: {name}")
    return {"archive": str(output), "size_MiB": round(output.stat().st_size / 2**20, 2),
            "sha256": sha(output), "files": len(files), "frozen_models": 24}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path.cwd())
    ap.add_argument("--out", type=Path, default=Path.home() / "Downloads")
    args = ap.parse_args()
    result = package(args.workspace, args.out)
    print(json.dumps(result, indent=2))
    print("Upload the HANDOFF ZIP to this chat. Keep both complete training folders backed up in Drive.")


if __name__ == "__main__": main()
