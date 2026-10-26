"""Build an additive patch for the installed cu126 delivery, no core changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


def build():
    original = BASE / "Notebooks/dqn_linux_cu126_recovery.zip"
    expected = "0e1ac8d70467e9610396ec93889b757d4a017e637b0088cc042007684766c290"
    if sha(original.read_bytes()) != expected: raise ValueError("Original cu126 ZIP changed")
    with zipfile.ZipFile(original) as z:
        adapter = z.read("LINUX_ADAPTER_MANIFEST.json")
    names = ("continue_numerical.py", "continue_numerical.sh", "README_NUMERICAL_CONTINUE.md")
    extras = {n: (BASE / n).read_bytes().replace(b"\r\n", b"\n") for n in names}
    manifest = {"format": "approved_numerical_continuation_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "base_zip_sha256": expected, "base_adapter_manifest_sha256": sha(adapter),
                "historical_bitwise_equivalence_claimed": False,
                "sha256": {n: sha(v) for n, v in extras.items()}}
    out = BASE / "Notebooks/dqn_continue_numerical_patch.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for n, v in extras.items(): z.writestr(n, v)
        z.writestr("NUMERICAL_PATCH_MANIFEST.json", json.dumps(manifest, indent=2))
    # Verify the overlay against a real extraction in a unique local directory.
    stage = BASE / "results/numerical_continuation_20261026/packaged_source"
    stage.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(original) as z: z.extractall(stage)
    with zipfile.ZipFile(out) as z: z.extractall(stage)
    return {"path": str(out), "bytes": out.stat().st_size, "sha256": sha(out.read_bytes()),
            "patch_members": list(extras) + ["NUMERICAL_PATCH_MANIFEST.json"], "stage": str(stage)}


if __name__ == "__main__": print(json.dumps(build(), indent=2))
