"""Separate cu126 delivery; preserve both original archives and all core bytes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parents[1]
ORIGINAL_SHA = "3e85adc7cd9af2c61b05bc739d3080ffdcf1e754c11bbd2ed6a6f5a039339b5a"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build():
    original = BASE / "Notebooks/dqn_extended_full_source.zip"
    if sha(original.read_bytes()) != ORIGINAL_SHA:
        raise ValueError("Original Colab source archive changed")
    names = ("linux_resume.py", "start_linux.sh", "README_LINUX.md", "cu126_recovery.py",
             "start_linux_cu126.sh", "README_CU126.md", "tests/test_cu126_recovery.py")
    extras = {name: (BASE / name).read_bytes().replace(b"\r\n", b"\n") for name in names}
    manifest = {"format": "linux_explicit_cu126_recovery_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
                "original_colab_zip_sha256": ORIGINAL_SHA, "original_members_unchanged": True,
                "cuda_build_policy": "explicit cu130 -> cu126, PyTorch 2.11.0 only; bitwise historical and restore gates",
                "sha256": {name: sha(data) for name, data in extras.items()}}
    out = BASE / "Notebooks/dqn_linux_cu126_recovery.zip"
    tmp = out.with_suffix(".zip.partial")
    with zipfile.ZipFile(original) as src, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            if item.filename in extras: raise ValueError("Original member collision")
            dst.writestr(item, src.read(item.filename))
        for name, data in extras.items(): dst.writestr(name, data)
        dst.writestr("LINUX_ADAPTER_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(original) as src, zipfile.ZipFile(tmp) as dst:
        if not all(src.read(n) == dst.read(n) for n in src.namelist()): raise ValueError("Original member changed")
        if dst.testzip() is not None: raise ValueError("Delivery ZIP corrupt")
    tmp.replace(out)
    return {"path": str(out), "sha256": sha(out.read_bytes()), "size": out.stat().st_size,
            "original_colab_zip_sha256": ORIGINAL_SHA}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
