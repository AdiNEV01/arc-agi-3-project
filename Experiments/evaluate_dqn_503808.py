"""Explicit frozen development evaluation of the audited final DQN matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import yaml

from Experiments.audit_dqn_503808_archive import SEEDS, VARIANTS


def materialize(repo, registry, archive):
    """Restore only endpoint weights/sidecars from the SHA-bound archive."""
    with archive.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
    if digest != registry["archive"]["sha256"]: raise ValueError("Wrong result archive checksum")
    stage=(repo/registry["ignored_evaluation_staging"]).resolve()
    if not stage.is_relative_to((repo/"artifacts/dqn_extended").resolve()):
        raise ValueError("Unexpected evaluation staging location")
    planned={}
    with zipfile.ZipFile(archive) as z:
        for r in registry["models"]+registry["comparison_models"]:
            dest=(repo/r["path"]).resolve()
            if not dest.is_relative_to(stage): raise ValueError("Unexpected weight destination")
            data=z.read(r["archive_member"])
            if hashlib.sha256(data).hexdigest()!=r["sha256"]: raise ValueError("Wrong weight checksum")
            planned[dest]=data
            member=r["archive_member"].rsplit("/frozen/",1)[0]+"/metadata.json"
            planned[dest.parent.parent/"metadata.json"]=z.read(member)
    # Validate all existing targets before making new files.
    for dest,data in planned.items():
        if dest.exists() and dest.read_bytes()!=data: raise FileExistsError("Existing artifact differs: " + str(dest))
    for dest,data in planned.items():
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open("xb") as f: f.write(data)


def validate_registry(repo, registry, steps):
    expected = {(v,s) for v in VARIANTS for s in SEEDS}
    checked = []
    stage = (repo / registry["ignored_evaluation_staging"]).resolve()
    if not stage.is_relative_to((repo/"artifacts/dqn_extended").resolve()):
        raise ValueError("Unexpected evaluation staging location")
    for step in steps:
        rows = registry["models"] if step == 503808 else registry["comparison_models"]
        if len(rows) != 12 or {(r["variant"],r["training_seed"]) for r in rows} != expected:
            raise ValueError("The entire four-variant, three-seed matrix is required")
        for r in rows:
            path = (repo/r["path"]).resolve()
            if r["checkpoint_step"] != step or not path.is_relative_to(stage):
                raise ValueError("Checkpoint path or endpoint differs")
            with path.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
            if digest != r["sha256"]: raise ValueError("Checkpoint checksum differs: " + r["path"])
            checked.append(r)
    config = yaml.safe_load((repo/"Configs/dqn_extended_development.yaml").read_text())
    if (config["games"]["dev"] != ["ft09-0d8bbf25", "vc33-5430563c", "cd82-fb555c5d"] or
        config["eval"]["seeds"] != [0,1,2] or config["eval"]["max_actions_per_game"] != 1000 or
        config["eval"]["click_grid"] != 64 or config["eval"]["operation_mode"] != "offline"):
        raise ValueError("Frozen development protocol changed")
    return stage, checked


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, default=Path("Configs/dqn_503808_checkpoints.json"))
    ap.add_argument("--steps", type=int, nargs="+", choices=[450000,503808], default=[503808])
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--materialize-from", type=Path, help="Restore endpoint weights from the original result ZIP")
    a=ap.parse_args()
    if len(set(a.steps)) != len(a.steps): raise ValueError("Duplicate endpoint")
    repo=Path(__file__).resolve().parents[1]
    registry=json.loads((repo/a.registry).read_text())
    if a.materialize_from: materialize(repo,registry,a.materialize_from.resolve())
    stage, checked=validate_registry(repo,registry,a.steps)
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable; no CPU fallback")
    plan=dict(verified_checkpoints=len(checked), endpoints=a.steps, device=a.device,
              model_game_seed_runs=len(checked)*9, matched_random_runs=9,
              selected_actions_per_game=1000, reserved_games_evaluated=False,
              no_learning=True)
    print(json.dumps(plan,indent=2),flush=True)
    if a.validate_only: return
    if a.out is None: ap.error("--out is required unless --validate-only is used")
    out=a.out.resolve()
    if out.exists(): raise FileExistsError("Choose a new output directory")
    subprocess.run([sys.executable,"-m","Experiments.evaluate_dqn_extended",str(stage),
        "--out",str(out),"--device",a.device,"--steps",*[str(s) for s in a.steps]],cwd=repo,check=True)
    validation=json.loads((out/"evaluation_validation.json").read_text())
    if validation["model_game_seed_runs"] != plan["model_game_seed_runs"]:
        raise ValueError("Evaluation matrix is incomplete")
    validate_registry(repo,registry,a.steps)  # Confirm model byte identity after evaluation.
    with (out/"registry_evaluation_plan.json").open("x") as f: json.dump(plan,f,indent=2)


if __name__ == "__main__": main()
