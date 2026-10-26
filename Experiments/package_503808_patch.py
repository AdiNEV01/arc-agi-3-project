"""Add a budget continuation driver without modifying earlier deliveries."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


def build():
    numerical = BASE / "Notebooks/dqn_continue_numerical_patch.zip"
    expected = "2989e2cca68fe6f1e0673147f6d54c2f33a8e44b270ec86b126c7c99c512ef4a"
    if sha(numerical.read_bytes()) != expected: raise ValueError("Existing numerical patch changed")
    with zipfile.ZipFile(numerical) as z: parent = z.read("NUMERICAL_PATCH_MANIFEST.json")
    names = ("extend_503808.py", "extend_503808.sh", "README_EXTEND_503808.md")
    extras = {n: (BASE / n).read_bytes().replace(b"\r\n", b"\n") for n in names}
    manifest = {"format": "450k_to_actual_ppo_503808_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "numerical_patch_manifest_sha256": sha(parent),
                "sha256": {n: sha(v) for n, v in extras.items()}}
    out = BASE / "Notebooks/dqn_extend_503808_patch.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for n, v in extras.items(): z.writestr(n, v)
        z.writestr("EXTENSION_503808_MANIFEST.json", json.dumps(manifest, indent=2))
    stage = BASE / "results/extension_503808_20261026/packaged_source"
    stage.mkdir(parents=True, exist_ok=True)
    for archive in (BASE / "Notebooks/dqn_linux_cu126_recovery.zip", numerical, out):
        with zipfile.ZipFile(archive) as z: z.extractall(stage)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": sha(out.read_bytes()), "stage": str(stage)}


if __name__ == "__main__": print(json.dumps(build(), indent=2))
