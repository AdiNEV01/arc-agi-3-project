"""Build the one-file manual Colab delivery; no remote integration branch."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cfg=json.loads(Path("Configs/dqn_extended_v1.json").read_text())
    files=[Path("evaluate.py"),Path("Configs/dqn_extended_v1.json"),Path("Configs/dqn_extended_development.yaml")]
    for folder in ("DQN","arc_env","agents","Experiments","tests"):
        files.extend(Path(folder).glob("*.py"))
    files.extend(Path("Evidence").glob("*"))
    for game in cfg["training_games"]+cfg["development_games"]:
        short,version=game.split("-",1)
        folder=Path("environment_files")/short/version
        assets=[p for p in folder.rglob("*") if p.is_file() and p.suffix in (".py",".json")]
        if not assets:raise FileNotFoundError(game)
        files.extend(assets)
    files=sorted(set(files))
    manifest={"format":"full450k_manual_v1","base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "created_utc":datetime.now(timezone.utc).isoformat(),"protocol":cfg,
        "source":"Exact integrated working tree; includes uncommitted work",
        "sha256":{p.as_posix():sha(p) for p in files},"heldout_assets_included":False,
        "changes_since_gpu_pilots":"Queue/recovery orchestration plus publish-only checksum hardening; learner AST is checked against pilot source"}
    out=Path("Notebooks/dqn_extended_full_source.zip")
    tmp=out.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.as_posix())
        z.writestr("FULL_SOURCE_MANIFEST.json",json.dumps(manifest,indent=2))
    tmp.replace(out)
    print(out,"bytes",out.stat().st_size,"sha256",sha(out),flush=True)


if __name__=="__main__":main()
