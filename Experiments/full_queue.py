"""Sequential, restartable 450K queue. No CPU fallback or session keep-alive.

Learner/algorithm is the pilot implementation; only publication was hardened.
Progress is reconstructed from verified checkpoints, never trusted from a label.
"""
import argparse
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from filelock import FileLock
import torch

from DQN.extended_train import check_resume, hash_file, json_write, source_hashes, runtime_hashes


class UnsafeRecovery(RuntimeError):
    pass


def atomic_copy(src, dst, expected=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    expected = expected or hash_file(src)
    if dst.exists() and hash_file(dst) == expected:
        return
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    if hash_file(tmp) != expected:
        raise IOError(f"Copy checksum mismatch: {dst}")
    tmp.replace(dst)


def verified_generations(job, config, variant, seed):
    candidates = []
    invalid = []
    for marker in job.glob("attempts/*/resume/step*/COMPLETE.json"):
        try:
            state = check_resume(marker.parent)
            if (state["protocol"] != config or state["variant"] != variant or state["seed"] != seed
                    or state["source_sha256"] != source_hashes()
                    or state["runtime_assets"] != runtime_hashes(config["training_games"])):
                raise UnsafeRecovery("Checkpoint source/protocol/game mismatch")
            candidates.append((state["global_step"], marker.parent))
            del state
        except UnsafeRecovery:
            raise
        except (OSError, ValueError, EOFError, RuntimeError) as exc:
            invalid.append({"path":str(marker.parent), "error":str(exc)})
    candidates.sort(key=lambda x:(x[0], str(x[1])))
    if invalid and not candidates:
        raise UnsafeRecovery("Continuation markers exist but no checkpoint verifies; refusing a restart from zero")
    return candidates, invalid


def collect_frozen(job, config, seed, variant, committed_step):
    """Keep all six original snapshots, including snapshots in earlier attempts."""
    result = {}
    for step in config["checkpoint_actions"]:
        if step > committed_step:
            continue
        target = job / "frozen" / f"step{step:06d}.pt"
        choices = sorted(job.glob(f"attempts/*/frozen/step{step:06d}.pt"))
        if not target.exists():
            if not choices:
                raise UnsafeRecovery(f"Missing evaluation snapshot at {step}")
            atomic_copy(choices[0], target)
        saved = torch.load(target, map_location="cpu", weights_only=True)
        if saved["global_step"] != step or saved["seed"] != seed or saved["config"]["variant"] != variant:
            raise UnsafeRecovery("Frozen checkpoint step/seed/variant mismatch")
        if not all(torch.isfinite(v).all().item() for k in ("online_state_dict","target_state_dict") for v in saved[k].values()):
            raise UnsafeRecovery("Nonfinite frozen model")
        result[str(step)] = {"path":str(target),"sha256":hash_file(target)}
    return result


def latest_progress(local_run):
    path = local_run / "actions.csv"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.seek(max(0,path.stat().st_size-8192))
            lines = f.read().decode().splitlines()
        for line in reversed(lines):
            cols = next(csv.reader([line]))
            if len(cols) == 17 and cols[0].isdigit():
                return int(cols[0]),float(cols[-1])
    except (OSError,ValueError,UnicodeDecodeError):
        pass
    return None


def save_progress(root, manifest):
    manifest["updated_utc_epoch"] = time.time()
    if (root / "progress.json").exists():
        atomic_copy(root / "progress.json",root / "progress.previous.json")
    json_write(root / "progress.json",manifest)


def validate_complete(job, config, variant, seed, latest):
    state = check_resume(latest)
    if state["global_step"] != config["total_actions"]:
        return False
    expected = sum(s >= config["warmup"] and s % config["train_every"] == 0
                   for s in range(1,config["total_actions"]+1))
    if state["gradient_steps"] != expected:
        raise UnsafeRecovery("Final gradient update count differs from frozen protocol")
    with gzip.open(latest / "history.csv.gz","rt",newline="") as f:
        rows = csv.DictReader(f)
        count = 0
        games = Counter()
        for count,r in enumerate(rows,1):
            if int(r["global_step"]) != count or r["truncated"] != "False":
                raise UnsafeRecovery("Invalid committed training history")
            expected_game = config["training_games"][(count-1)%len(config["training_games"])]
            if r["game"] != expected_game:
                raise UnsafeRecovery("Training game sequence changed")
            games[r["game"]] += 1
    if count != config["total_actions"]:
        raise UnsafeRecovery("Final committed log is incomplete")
    snapshots = collect_frozen(job,config,seed,variant,state["global_step"])
    if len(snapshots) != len(config["checkpoint_actions"]):
        raise UnsafeRecovery("Not all six milestone models are accessible")
    json_write(job / "RUN_COMPLETE.json",dict(variant=variant,seed=seed,actions=count,
        gradient_steps=expected,per_game_actions=dict(games),latest_checkpoint=str(latest),
        snapshots=snapshots,validated=True))
    atomic_copy(latest / "history.csv.gz",job / "actions.csv.gz")
    attempt = latest.parent.parent
    atomic_copy(attempt / "metadata.json",job / "metadata.json")
    if (attempt / "summary.json").exists():
        atomic_copy(attempt / "summary.json",job / "summary.json")
    return True


def prune_verified_generations(job, generations, active_attempt=None):
    # The learner retains two per attempt. Retain two across all attempts too.
    root = job.resolve()
    for _, path in generations[:-2]:
        # The active learner prunes its own generation directory after publish.
        # Do not race that operation; prune only older attempts while it runs.
        if active_attempt and path.parent.parent.name == active_attempt:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved.parent.name != "resume":
            raise UnsafeRecovery("Unsafe retention path")
        shutil.rmtree(resolved)


def classify_failure(text):
    fatal = ("protocol/variant/seed/source differs", "reconstruction diverged", "same recorded Python",
             "same torch version", "runtime assets differ", "Nonfinite", "nonfinite", "illegal action",
             "Unexpected full reset", "Unexpected artificial truncation", "CUDA requested but unavailable",
             "no CUDA-capable", "quota", "No space left", "Disk quota", "Permission denied",
             "deterministic implementation", "No module named", "ImportError", "AssertionError")
    return "fatal" if any(x.lower() in text.lower() for x in fatal) else "recoverable"


def run_queue(config, root, local, *, max_retries=2, command_factory=None, require_cuda=True, poll_seconds=5):
    root,local = Path(root),Path(local)
    root.mkdir(parents=True,exist_ok=True)
    local.mkdir(parents=True,exist_ok=True)
    if require_cuda and (not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0)):
        raise UnsafeRecovery("A T4 CUDA runtime is required; no CPU fallback")
    definition = {"protocol":config,"source_sha256":source_hashes(),
                  "runtime_assets":runtime_hashes(config["training_games"]),
                  "queue_sources":{p:hash_file(Path(p)) for p in ("Experiments/full_queue.py","Experiments/full_preflight.py","Experiments/drive_storage.py")}}
    if (root / "QUEUE.json").exists():
        if json.loads((root / "QUEUE.json").read_text()) != definition:
            raise UnsafeRecovery("Queue definition/source differs; cannot mix experiments")
    else:
        json_write(root / "QUEUE.json",definition)
    plan = [(v,s) for s in config["seeds"] for v in config["variants"]]
    manifest = {"protocol_id":config["protocol_id"],"target_actions_each":config["total_actions"],
                "jobs":{f"{v}_seed{s}":{"configuration":v,"seed":s,"completed_actions":0,"status":"pending"}
                        for v,s in plan},"status":"running","owner_pid":os.getpid(),"remaining_experiments":len(plan)}
    # OS-level local lock prevents duplicate Run All / Resume within this VM.
    with FileLock(str(local / "queue.lock"),timeout=0):
        for variant,seed in plan:
            key = f"{variant}_seed{seed}"
            job = root / variant / f"seed{seed}"
            job.mkdir(parents=True,exist_ok=True)
            retries = 0
            while True:
                gens,invalid = verified_generations(job,config,variant,seed)
                completed_step,checkpoint = gens[-1] if gens else (0,None)
                entry = {"configuration":variant,"seed":seed,"completed_actions":completed_step,
                    "target_actions":config["total_actions"],"latest_verified_checkpoint":str(checkpoint) if checkpoint else None,
                    "invalid_checkpoints":invalid,"status":"pending","retry":retries}
                manifest["jobs"][key] = entry
                if checkpoint and validate_complete(job,config,variant,seed,checkpoint):
                    entry["status"] = "complete"
                    if (job / "summary.json").exists():
                        entry["actions_per_second_last_attempt"] = json.loads((job / "summary.json").read_text()).get("actions_per_second_including_persistence")
                    manifest["remaining_experiments"] = len(plan)-sum(v["status"]=="complete" for v in manifest["jobs"].values())
                    save_progress(root,manifest)
                    print(f"SKIP COMPLETE {key}: {completed_step}",flush=True)
                    break
                if require_cuda and not torch.cuda.is_available():
                    raise UnsafeRecovery("GPU became unavailable; progress preserved")
                if require_cuda:
                    from Experiments.drive_storage import require_space
                    entry["drive_storage"] = require_space()
                if shutil.disk_usage(root).free < 512*2**20:
                    raise UnsafeRecovery("Less than512MiB reported free; stopping before a new attempt")
                attempt = f"a_{uuid.uuid4().hex[:12]}"
                attempt_local = local / key / attempt
                if checkpoint:
                    # Recover from disk verified independently of progress.json.
                    cache = local / "resume_cache" / key / checkpoint.name
                    cache.mkdir(parents=True,exist_ok=True)
                    marker = json.loads((checkpoint / "COMPLETE.json").read_text())
                    for name,h in marker["sha256"].items():
                        atomic_copy(checkpoint / name,cache / name,h)
                    atomic_copy(checkpoint / "COMPLETE.json",cache / "COMPLETE.json")
                    check_resume(cache)
                    resume_arg = cache
                else:
                    resume_arg = None
                if command_factory:
                    cmd = command_factory(variant,seed,attempt_local,resume_arg,job / "attempts")
                else:
                    cmd = [sys.executable,"-u","-m","DQN.extended_train","--variant",variant,"--seed",str(seed),
                        "--device","cuda","--run-dir",str(attempt_local),"--stop-at",str(config["total_actions"]),
                        "--persistent-root",str(job / "attempts")]
                    if resume_arg:
                        cmd += ["--resume",str(resume_arg)]
                entry.update(status="running",attempt=attempt,restored_from_actions=completed_step)
                save_progress(root,manifest)
                print(f"TRAIN {key}: {completed_step}/{config['total_actions']} -> {attempt}",flush=True)
                env = {**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONHASHSEED":str(seed),"CUBLAS_WORKSPACE_CONFIG":":4096:8"}
                log_path = job / (attempt + ".log")
                started = time.monotonic()
                with log_path.open("x") as log:
                    process = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
                    last_line = ""
                    last_check = completed_step
                    try:
                        while process.poll() is None:
                            time.sleep(poll_seconds)
                            progress = latest_progress(attempt_local)
                            if progress:
                                step,elapsed = progress
                                entry["observed_actions"] = step
                                entry["actions_per_second_current_attempt"] = (step-completed_step)/max(0.01,time.monotonic()-started)
                                line = f"{key}: {step}/{config['total_actions']} | durable={entry['completed_actions']}"
                                if line != last_line:
                                    print(line,flush=True);last_line=line
                                boundary = (step//config["resume_every"])*config["resume_every"]
                                if boundary > last_check:
                                    found,_ = verified_generations(job,config,variant,seed)
                                    if found and found[-1][0] > last_check:
                                        last_check,cp = found[-1]
                                        entry.update(completed_actions=last_check,latest_verified_checkpoint=str(cp))
                                        collect_frozen(job,config,seed,variant,last_check)
                                        prune_verified_generations(job,found,active_attempt=attempt)
                                        print(f"DRIVE COMPLETE {key}: {last_check}",flush=True)
                            save_progress(root,manifest)
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill();process.wait(timeout=10)
                        raise
                found,_ = verified_generations(job,config,variant,seed)
                if found:
                    collect_frozen(job,config,seed,variant,found[-1][0])
                    prune_verified_generations(job,found)
                if process.returncode == 0:
                    if not found or not validate_complete(job,config,variant,seed,found[-1][1]):
                        raise UnsafeRecovery("Process exited without a verified final continuation checkpoint")
                    # Rescan once to mark completion, then automatically advance.
                    continue
                error = log_path.read_text(errors="replace")[-12000:]
                entry.update(status="failed",failure=error[-3000:],returncode=process.returncode)
                save_progress(root,manifest)
                if classify_failure(error)=="fatal" or retries >= max_retries:
                    raise UnsafeRecovery(f"Stopped {key}; progress preserved. See {log_path}")
                retries += 1
                print(f"RECOVER {key}: retry{retries}/{max_retries}; independently verifying latest checkpoint",flush=True)
        manifest["status"] = "complete"
        manifest["remaining_experiments"] = 0
        save_progress(root,manifest)
        return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--local",type=Path,required=True)
    ap.add_argument("--config",default="Configs/dqn_extended_v1.json")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    preflight = args.root / "PREFLIGHT_PASSED.json"
    if not preflight.exists():
        raise UnsafeRecovery("Run the notebook Recovery and preflight cell first")
    proof = json.loads(preflight.read_text())
    if not proof.get("valid") or proof["source_sha256"] != source_hashes():
        raise UnsafeRecovery("Preflight proof is missing or does not match the current source")
    try:
        run_queue(config,args.root,args.local)
    except BaseException as exc:
        args.root.mkdir(parents=True,exist_ok=True)
        json_write(args.root / "LAST_FAILURE.json",{"error":str(exc),"time":time.time(),
            "next_step":"Reconnect a T4 runtime, rerun setup/Drive/preflight, then RESUME TRAINING. No CPU fallback."})
        if (args.root / "progress.json").exists():
            progress=json.loads((args.root / "progress.json").read_text())
            progress.update(status="stopped",failure=str(exc))
            json_write(args.root / "progress.json",progress)
        raise


if __name__ == "__main__":
    main()
