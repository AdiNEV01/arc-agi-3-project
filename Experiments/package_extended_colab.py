"""Package exact working revision + only approved exposed runtime assets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from DQN.extended_train import hash_file


def main():
    cfg = json.loads(Path("Configs/dqn_extended_v1.json").read_text())
    files = [Path("evaluate.py"),Path("Configs/dqn_extended_v1.json"),Path("Configs/dqn_extended_development.yaml")]
    for directory in ("DQN","arc_env","agents","Experiments","tests"):
        files.extend(Path(directory).glob("*.py"))
    for game in cfg["training_games"]+cfg["development_games"]:
        short, version = game.split("-",1)
        folder = Path("environment_files")/short/version
        assets = [p for p in folder.rglob("*") if p.is_file() and p.suffix in (".py",".json")]
        if not assets:
            raise FileNotFoundError(f"Required exposed game cache missing: {game}")
        files.extend(assets)
    files = sorted(set(files))
    manifest = {"base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source":"working revision including uncommitted integration; per-file hashes are authoritative",
        "protocol":cfg,"sha256":{p.as_posix():hash_file(p) for p in files},
        "created_utc":datetime.now(timezone.utc).isoformat(),"heldout_assets_included":False}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path("Notebooks")/f"dqn_extended_source_{stamp}.zip"
    with zipfile.ZipFile(out,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.as_posix())
        z.writestr("EXTENDED_SOURCE_MANIFEST.json",json.dumps(manifest,indent=2))
    print(out)
    print("SHA256:",hash_file(out))


if __name__ == "__main__":
    main()
