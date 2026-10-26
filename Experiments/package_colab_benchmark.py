"""Deliver current working files, including unpushed adapter, without model artifacts.
Run from repository root: python -m Experiments.package_colab_benchmark
Cached ls20 runtime assets are copied opaquely, never used for agent design.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone

root = Path.cwd()
files = []
for directory in ("DQN", "arc_env"):
    files.extend(p for p in (root / directory).glob("*.py"))
files += [root / p for p in ("agents/__init__.py", "agents/checkpoint_encoding.py", "tests/__init__.py",
    "tests/fake_arc.py", "tests/test_dqn_shared_training.py", "tests/test_dqn_devices.py")]
# Include only the already exposed ls20 game; no checkpoint/log/secret directories.
files += [p for p in (root / "environment_files/ls20").rglob("*")
          if p.is_file() and p.suffix in (".py", ".json") and "__pycache__" not in p.parts]
if not any("environment_files" in p.parts for p in files):
    raise RuntimeError("ls20 offline cache missing; obtain it through the toolkit before packaging")
files = sorted(set(files))
manifest = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source": "current working tree (includes uncommitted integration changes)",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
out = root / "Notebooks" / ("dqn_colab_source_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(root).as_posix())
    z.writestr("BENCHMARK_SOURCE_MANIFEST.json", json.dumps(manifest, indent=2))
print(out)
print("sha256:", hashlib.sha256(out.read_bytes()).hexdigest())
