"""Materialize ONLY PPO finals from a fetched ref; validate archives and shared API."""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

import arc_agi
import torch

from agents.ppo_agent import PPOPolicy
from evaluate import run_game, versions, hardware


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def git(*args):
    return subprocess.check_output(["git", *args])


def model_hash(model):
    h = hashlib.sha256()
    for key, value in model.policy.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="origin/PPO")
    ap.add_argument("--out", default="results/ppo_finals_verified_20261024")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    ref = git("rev-parse", args.ref).decode().strip()
    tree = {}
    for row in git("ls-tree", "-rlz", ref).split(b"\0"):
        if row:
            meta, path = row.split(b"\t", 1)
            _, kind, oid, size = meta.decode().split()
            if kind == "blob":
                tree[path.decode()] = (oid, int(size))
    records = []
    source_files = {}
    for name in ("PPO/run_experiments.py", "PPO/arc_env.py", "PPO/arc_env_memory.py", "PPO/curiosity_wrapper.py"):
        raw = git("show", f"{ref}:{name}")
        source_files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode()}
    (out / "source_training_contract.json").write_text(json.dumps(source_files, indent=2))
    attributes = [p for p in tree if p.endswith(".gitattributes") or p == ".lfsconfig"]
    downloaded = []
    for exp in ("baseline", "curiosity", "memory_only", "memory_curiosity"):
        for seed in (0, 1, 2):
            path = Path(f"PPO/logs/{exp}_seed{seed}/final_model.zip")
            name = path.as_posix()
            oid, size = tree[name]
            # Inspect local object availability without invoking lazy fetch.
            env = os.environ.copy()
            env["GIT_NO_LAZY_FETCH"] = "1"
            present = subprocess.run(["git", "cat-file", "-e", oid], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            started = time.perf_counter()
            raw = git("show", f"{ref}:{name}")
            if not present:
                downloaded.append(name)
            if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(f"Unmaterialized LFS pointer: {name}")
            assert len(raw) == size
            sha = hashlib.sha256(raw).hexdigest()
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, f"Refusing overwrite: {name}"
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            resource_name = name.replace("final_model.zip", "resources.json")
            resources = json.loads(git("show", f"{ref}:{resource_name}"))
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                assert archive.testzip() is None
                assert {"data", "policy.pth", "_stable_baselines3_version"} <= set(archive.namelist())
                data = json.loads(archive.read("data"))
                saved_version = archive.read("_stable_baselines3_version").decode()
            p = PPOPolicy(str(path), click_grid=64, seed=0, device="cpu")
            m = p.model
            assert m.seed == seed
            expected_channels = 64 if "memory" in exp else 16
            assert m.observation_space.shape == (expected_channels, 64, 64)
            assert m.action_space.n == 7
            assert all(torch.isfinite(v).all() for v in m.policy.state_dict().values())
            before = model_hash(m)
            smoke = []
            for game, directory, budget in (("tt01-v1", "tests/fixtures/games", 6),
                                            ("ls20-9607627b", "environment_files", 6)):
                arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                                     environments_dir=directory)
                rec = run_game(arc, p, game, 0, 0, budget, 64, 30, None, ["ppo-final-compatibility"])
                assert rec["illegal_actions"] == 0
                assert rec["scorecard_runs"] == 1 and rec["full_resets_after_start"] == 0
                assert rec["score_official"] is not None
                smoke.append(rec)
            assert model_hash(m) == before
            record = {
                "experiment": exp, "training_seed": seed, "path": name,
                "source_ref": ref, "git_blob": oid, "bytes": size, "sha256": sha,
                "git_lfs_pointer": False, "git_lfs_status": "ordinary Git ZIP blob",
                "blob_already_local_before_verification": present,
                "valid_zip_crc": True, "loaded_successfully": True,
                "actual_timesteps": m.num_timesteps, "n_envs": m.n_envs,
                "n_steps": m.n_steps, "n_epochs": m.n_epochs, "n_updates": m._n_updates,
                "batch_size": m.batch_size, "gamma": m.gamma, "gae_lambda": m.gae_lambda,
                "learning_rate": float(m.lr_schedule(1)), "ent_coef": m.ent_coef,
                "clip_range": float(m.clip_range(1)), "vf_coef": m.vf_coef,
                "max_grad_norm": m.max_grad_norm, "normalize_images": m.policy.normalize_images,
                "observation_shape": list(m.observation_space.shape), "action_count": m.action_space.n,
                "policy_class": type(m.policy).__name__, "feature_extractor": type(m.policy.features_extractor).__name__,
                "policy_architecture": str(m.policy), "parameters": sum(v.numel() for v in m.policy.parameters()),
                "weight_sha256": before, "weights_frozen_during_checks": True,
                "saved_sb3_version": saved_version, "resources": resources,
                "small_checks_only": smoke, "verification_seconds": round(time.perf_counter()-started, 3),
            }
            records.append(record)
            (out / "checkpoints.json").write_text(json.dumps(records, indent=2, default=json_default))
            print(f"{exp} seed{seed}: {m.num_timesteps} timesteps, {size} bytes, shape {m.observation_space.shape}, load+shared checks PASS", flush=True)
            del p, m, raw
            gc.collect()
    assert len(records) == 12
    assert len({r["weight_sha256"] for r in records}) == 12, "Duplicate model weights"
    report = {"source_commit": ref, "all_12_valid": True, "all_12_weights_distinct": True,
              "attribute_files": attributes, "fetched_final_blob_names_if_missing": downloaded,
              "intermediate_checkpoint_files_materialized": [], "versions": versions(), "hardware": hardware(),
              "checkpoints": records}
    (out / "verification.json").write_text(json.dumps(report, indent=2, default=json_default))
    print("All 12 final checkpoints verified; no intermediate checkpoint files materialized.", flush=True)


if __name__ == "__main__":
    main()
