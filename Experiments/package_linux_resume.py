"""Add a Linux adapter without changing any byte of the Colab source members."""
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
        raise ValueError("Original Colab delivery changed; refuse to build a mixed continuation bundle")
    extras = {name: (BASE / name).read_bytes().replace(b"\r\n", b"\n")
              for name in ("linux_resume.py", "start_linux.sh", "README_LINUX.md")}
    manifest = {"format": "linux_resume_adapter_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "original_colab_zip_sha256": ORIGINAL_SHA,
                "original_members_unchanged": True,
                "sha256": {name: sha(data) for name, data in extras.items()}}
    out = BASE / "Notebooks/dqn_linux_resume.zip"
    tmp = out.with_suffix(".zip.partial")
    with zipfile.ZipFile(original) as src, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            if item.filename in extras or item.filename == "LINUX_ADAPTER_MANIFEST.json":
                raise ValueError("Unexpected Linux adapter in original source")
            dst.writestr(item, src.read(item.filename))
        for name, data in extras.items():
            dst.writestr(name, data)
        dst.writestr("LINUX_ADAPTER_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(original) as src, zipfile.ZipFile(tmp) as dst:
        assert all(src.read(name) == dst.read(name) for name in src.namelist())
        assert dst.testzip() is None
    tmp.replace(out)
    return {"path": str(out), "size": out.stat().st_size, "sha256": sha(out.read_bytes()),
            "original_colab_zip_sha256": ORIGINAL_SHA, "added_files": list(extras)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
