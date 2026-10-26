"""Explicit CUDA-build migration with a bitwise, historical checkpoint gate.

Original code and checkpoint files remain untouched. Conversion changes build
metadata in NEW checkpoint files only; a provenance receipt preserves its origin.
Matrix training is conditional on actual RTX CUDA execution and comparisons
against historical cu130 checkpoints. A finite comparison is not a guarantee
that future trajectories on different hardware will stay bitwise identical.
"""
import argparse
from datetime import datetime, timezone
import csv
import gc
import gzip
import hashlib
from importlib import metadata
from itertools import zip_longest
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import time
import uuid
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
BASE = Path(__file__).resolve().parent
OLD_TORCH = "2.11.0+cu130"
NEW_TORCH = "2.11.0+cu126"
PINNED = {"python": "3.13.16", "numpy": "2.5.4", "arc-agi": "0.9.9",
          "arcengine": "0.9.3", "gymnasium": "1.4.0"}
STATE_FIELDS = (
    "online_state_dict", "target_state_dict", "optimizer_state_dict", "seed",
    "global_step", "gradient_steps", "config", "training_backend", "protocol",
    "variant", "num_frames", "novelty_beta", "source_sha256", "runtime_assets",
    "replay_position", "agent_rng", "python_rng", "numpy_rng", "torch_rng",
    "cuda_rng", "device_type", "journal", "initial_digests", "contexts")


class RecoveryStopped(RuntimeError):
    pass


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(2**20), b""):
            h.update(data)
    return h.hexdigest()


def require_build(state, build):
    expected = {**PINNED, "torch": build}
    if state["software"] != expected or state["torch_version"] != build:
        raise RecoveryStopped(f"Unexpected checkpoint software. Required {expected}; found {state['software']}")
    if state["device_type"] != "cuda":
        raise RecoveryStopped("The original checkpoint must be a CUDA checkpoint")
    if len(state["cuda_rng"]) != 1:
        raise RecoveryStopped("This migration supports exactly one recorded CUDA RNG device")


def assert_bits(a, b, path="state"):
    """Compare every tensor byte, including signed zeros, and nested RNG state."""
    import torch
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
            raise RecoveryStopped(f"Equivalence failed: tensor layout at {path}")
        av = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        bv = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        if av != bv:
            raise RecoveryStopped(f"Equivalence failed: tensor bits at {path}")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise RecoveryStopped(f"Equivalence failed: mapping keys at {path}")
        for key in a:
            assert_bits(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise RecoveryStopped(f"Equivalence failed: sequence at {path}")
        for i, (av, bv) in enumerate(zip(a, b)):
            assert_bits(av, bv, f"{path}[{i}]")
    elif isinstance(a, float):
        if not isinstance(b, float) or struct.pack("!d", a) != struct.pack("!d", b):
            raise RecoveryStopped(f"Equivalence failed: floating point value at {path}")
    elif type(a) is not type(b) or a != b:
        raise RecoveryStopped(f"Equivalence failed: value at {path}")


def replay_digests(path):
    # Stream uncompressed .npy members; avoid loading two full replay arrays.
    result = {}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise RecoveryStopped("Duplicate replay array names")
        required = {x + ".npy" for x in ("states", "next_states", "actions", "rewards", "terminals", "masks")}
        if set(names) != required:
            raise RecoveryStopped("Replay arrays are incomplete")
        for name in sorted(names):
            h = hashlib.sha256()
            with z.open(name) as f:
                for data in iter(lambda: f.read(2**20), b""):
                    h.update(data)
            result[name] = h.hexdigest()
    return result


def compare_checkpoints(reference, candidate):
    from DQN.extended_train import check_resume
    a, b = check_resume(Path(reference)), check_resume(Path(candidate))
    for key in STATE_FIELDS:
        if key not in a or key not in b:
            raise RecoveryStopped(f"Incomplete learner state: missing {key}")
        assert_bits(a[key], b[key], key)
    arrays = replay_digests(Path(reference) / "replay.npz")
    if arrays != replay_digests(Path(candidate) / "replay.npz"):
        raise RecoveryStopped("Equivalence failed: replay arrays, actions, rewards, terminals or masks")
    count = 0
    with gzip.open(Path(reference) / "history.csv.gz", "rt", newline="") as af, \
            gzip.open(Path(candidate) / "history.csv.gz", "rt", newline="") as bf:
        ar, br = csv.DictReader(af), csv.DictReader(bf)
        if ar.fieldnames != br.fieldnames or "elapsed_seconds" not in ar.fieldnames:
            raise RecoveryStopped("Training history columns differ")
        for count, (av, bv) in enumerate(zip_longest(ar, br), 1):
            if av is None or bv is None:
                raise RecoveryStopped("Training history lengths differ")
            av.pop("elapsed_seconds"); bv.pop("elapsed_seconds")
            if av != bv:
                changed = [k for k in av if av[k] != bv[k]]
                raise RecoveryStopped(f"Equivalence failed: history row {count}, fields {changed}")
    if count != a["global_step"]:
        raise RecoveryStopped("Training history is not complete for the saved step")
    return {"passed": True, "step": a["global_step"], "history_rows": count,
            "state_fields_bitwise_equal": list(STATE_FIELDS), "replay_array_sha256": arrays,
            "excluded": ["wall-clock times", "platform/build metadata", "attempt/parent paths"],
            "reference_state_sha256": sha(Path(reference) / "state.pt"),
            "candidate_state_sha256": sha(Path(candidate) / "state.pt")}


def convert_copy(source, destination, target=NEW_TORCH):
    """Explicit, allowlisted conversion into a NEW directory; never patch originals."""
    import torch
    from DQN.extended_train import check_resume
    source, destination = Path(source), Path(destination)
    if destination.resolve().is_relative_to(source.resolve()):
        raise RecoveryStopped("Conversion output must be outside the original checkpoint directory")
    original = check_resume(source)
    require_build(original, OLD_TORCH)
    if target != NEW_TORCH:
        raise RecoveryStopped("Only cu130 -> cu126 with PyTorch 2.11.0 is permitted")
    destination.mkdir(parents=True, exist_ok=False)
    origin = {"kind": "explicit_cuda_build_conversion", "source": str(source.resolve()),
              "original_complete": json.loads((source / "COMPLETE.json").read_text()),
              "original_software": original["software"], "original_cuda_version": original["cuda_version"],
              "target_software": {**PINNED, "torch": NEW_TORCH},
              "reason": "Driver 565 cannot run CUDA 13; verify cu126 equivalence before any matrix training",
              "tensor_optimizer_replay_rng_and_game_state_changes": False,
              "verification_status": "conversion alone does not establish equivalence"}
    converted = {**original, "software": {**PINNED, "torch": NEW_TORCH},
                 "torch_version": NEW_TORCH, "cuda_version": "12.6", "cuda_build_conversion": origin}
    for key in STATE_FIELDS:
        assert_bits(original[key], converted[key], key)
    torch.save(converted, destination / "state.pt")
    for name in ("replay.npz", "history.csv.gz"):
        shutil.copy2(source / name, destination / name)
        if sha(source / name) != sha(destination / name):
            raise RecoveryStopped(f"Converted checkpoint copy differs: {name}")
    write_json(destination / "CUDA_BUILD_CONVERSION.json", origin)
    write_json(destination / "COMPLETE.json", {"step": original["global_step"],
        "sha256": {n: sha(destination / n) for n in ("state.pt", "replay.npz", "history.csv.gz")}})
    restored = check_resume(destination)
    require_build(restored, NEW_TORCH)
    for key in STATE_FIELDS:
        assert_bits(original[key], restored[key], key)
    return origin


def gpu_check(config):
    import numpy as np
    import torch
    from DQN.dqn_agent import DQNAgent
    from DQN.action_codec import N_ACTIONS
    software = {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
                **{k: metadata.version(k) for k in ("arc-agi", "arcengine", "gymnasium")}}
    if software != {**PINNED, "torch": NEW_TORCH} or torch.version.cuda != "12.6":
        raise RecoveryStopped(f"Use the separate cu126 environment; found {software}, CUDA {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RecoveryStopped("Exactly one working NVIDIA CUDA device is required; no CPU fallback")
    device = torch.cuda.get_device_properties(0)
    if "RTX 3050" not in device.name:
        raise RecoveryStopped(f"Expected the reported RTX 3050; found {device.name}")
    identity = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,uuid",
                                        "--format=csv,noheader"], text=True).strip()
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Exercise real project encoding, masks, inference, backward and Adam for
    # both one-frame and four-frame models, rather than trusting is_available.
    losses = {}
    for frames in (1, 4):
        agent = DQNAgent(seed=42, batch_size=8, buffer_capacity=8, device="cuda", num_frames=frames)
        grid = np.zeros((frames, 64, 64), dtype=np.uint8)
        for i in range(8):
            agent.replay.add(grid + i, 1, 0.01, grid + (i + 1), False, np.ones(N_ACTIONS, dtype=bool))
        value = agent.learn()
        torch.cuda.synchronize()
        if value is None or not np.isfinite(value) or agent.gradient_steps != 1:
            raise RecoveryStopped("The project CUDA gradient/optimizer test failed")
        if not all(torch.isfinite(p).all().item() for p in agent.online.parameters()):
            raise RecoveryStopped("Nonfinite project CUDA model")
        losses[str(frames)] = value
        del agent
    gc.collect(); torch.cuda.empty_cache()
    return {"passed": True, "gpu": device.name, "memory_bytes": device.total_memory,
            "compute_capability": [device.major, device.minor], "driver_identity": identity,
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "software": software, "smoke_losses_by_frame_count": losses}


def scan(root, config, target_build=OLD_TORCH):
    from DQN.extended_train import check_resume, source_hashes, runtime_hashes
    from Experiments.full_queue import verified_generations
    root = Path(root)
    definition = {"protocol": config, "source_sha256": source_hashes(),
                  "runtime_assets": runtime_hashes(config["training_games"]),
                  "queue_sources": {p: sha(BASE / p) for p in ("Experiments/full_queue.py",
                      "Experiments/full_preflight.py", "Experiments/drive_storage.py")}}
    if json.loads((root / "QUEUE.json").read_text()) != definition:
        raise RecoveryStopped("Queue protocol/source/assets differ from the original Colab delivery")
    previous = json.loads((root / "progress.json").read_text())["jobs"]
    jobs = []
    for seed in config["seeds"]:
        for variant in config["variants"]:
            key = f"{variant}_seed{seed}"
            if key not in previous:
                raise RecoveryStopped(f"Missing queue entry {key}; download the whole experiment folder")
            directory = root / variant / f"seed{seed}"
            generations, invalid = verified_generations(directory, config, variant, seed)
            if invalid:
                raise RecoveryStopped(f"Invalid original checkpoint(s) for {key}: {invalid}")
            latest = generations[-1][0] if generations else 0
            recorded = previous[key].get("completed_actions", 0)
            observed = previous[key].get("observed_actions", 0)
            attempted = previous[key].get("attempt") or previous[key].get("status") in ("running", "failed")
            if latest < recorded or (latest == 0 and (observed > 0 or attempted)):
                raise RecoveryStopped(f"Missing full recovery state for {key}; refusing to restart from zero or discard saved progress")
            pairs = {}
            for step, path in generations:
                state = check_resume(path)
                # A migrated copy retains old cu130 generations alongside new
                # cu126 states, including completed jobs from either runtime.
                build = state["torch_version"] if target_build == NEW_TORCH else OLD_TORCH
                if build not in (OLD_TORCH, NEW_TORCH): raise RecoveryStopped("Unapproved CUDA build in result copy")
                require_build(state, build)
                pairs[step] = path
            if latest and latest < config["total_actions"]:
                require_build(check_resume(pairs[latest]), target_build)
            item = {"variant": variant, "seed": seed, "step": latest,
                    "generations": [(s, str(p)) for s, p in sorted(pairs.items())],
                    "latest": str(pairs[latest]) if latest else None,
                    "complete_marker_sha256": {str(p): sha(p / "COMPLETE.json") for _, p in generations}}
            # In a migrated copy, old immutable cu130 generations coexist with
            # the new conversion. Check the selected latest only on repeat runs.
            jobs.append(item)
            print(f"VERIFIED {key}: {latest}/{config['total_actions']}", flush=True)
    return jobs


def run_probe(variant, seed, resume, stop, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    cmd = [sys.executable, "-u", "-m", "DQN.extended_train", "--variant", variant, "--seed", str(seed),
           "--device", "cuda", "--run-dir", str(out), "--resume", str(resume), "--stop-at", str(stop)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    started = time.monotonic()
    with log.open("x") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                time.sleep(10)
                print(f"VERIFYING {variant} seed{seed}: game reconstruction / reference replay, {time.monotonic()-started:.0f}s; log {log}", flush=True)
        except BaseException:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if process.returncode:
        raise RecoveryStopped(f"Actual checkpoint restoration/replay failed. See {log}:\n{log.read_text(errors='replace')[-4000:]}")
    return out / "resume" / f"step{stop:06d}"


def prove_job(job, config, audit):
    generations = job["generations"]
    if len(generations) < 2:
        raise RecoveryStopped(f"Two distinct original checkpoints are required to test cross-build equivalence for {job['variant']} seed{job['seed']}")
    (start, earlier), (end, later) = generations[-2:]
    if end <= start or end % config["resume_every"] != 0:
        raise RecoveryStopped("Historical reference interval is not a normal saved checkpoint interval")
    stop = min(end + 10, config["total_actions"])
    if stop <= end:
        raise RecoveryStopped("No unfinished interval is available for an actual latest-checkpoint restoration test")
    audit = Path(audit)
    convert_copy(earlier, audit / "converted_earlier")
    convert_copy(later, audit / "converted_latest")
    # Path A reconstructs the earlier games and retrains a KNOWN historical
    # interval, saving at the old end and continuing 10 diagnostic actions.
    # Path B restores the ACTUAL latest original checkpoint and continues those
    # same actions. Comparing A/B tests full same-GPU restoration, including RNG,
    # optimizer, replay, masks, frame history and novelty.
    a = run_probe(job["variant"], job["seed"], audit / "converted_earlier", stop, audit / "from_earlier")
    historical = compare_checkpoints(later, a.parent / f"step{end:06d}")
    write_json(audit / "HISTORICAL_COMPARISON.json", historical)
    print(f"HISTORICAL BUILD EQUIVALENCE PASSED: {start} -> {end}", flush=True)
    b = run_probe(job["variant"], job["seed"], audit / "converted_latest", stop, audit / "from_latest")
    restored = compare_checkpoints(a, b)
    write_json(audit / "LATEST_RESTORE_COMPARISON.json", restored)
    print(f"ACTUAL LATEST CHECKPOINT RESTORATION PASSED: {end} -> {stop}", flush=True)
    return {"variant": job["variant"], "seed": job["seed"], "earlier_step": start,
            "latest_step": end, "historical_equivalence": historical, "latest_restore_equivalence": restored,
            "verification_selected_actions": (stop-start)+(stop-end),
            "counted_as_matrix_training": False,
            "original_checkpoint": later, "original_complete_sha256": sha(Path(later) / "COMPLETE.json")}


def locate_original(explicit):
    from linux_resume import locate_queue, find_download
    if explicit:
        return locate_queue(explicit, BASE / "cu126_imports")
    old = Path.home() / "Downloads/arc_dqn_linux/LINUX_CHECKPOINT_LOCATION.json"
    if old.exists():
        path = Path(json.loads(old.read_text())["root"])
        if (path / "QUEUE.json").exists(): return path.resolve()
    return locate_queue(find_download(), BASE / "cu126_imports")


def fork_results(original, jobs, audit):
    destination = BASE / ("CU126_450K_RESULTS_" + audit.name)
    original = Path(original).resolve()
    if destination.resolve().is_relative_to(original):
        raise RecoveryStopped("New results must be outside the historical checkpoint folder")
    size = 0
    for p in original.rglob("*"):
        if p.is_symlink(): raise RecoveryStopped("Checkpoint folder contains a symlink; refusing an ambiguous copy")
        if p.is_file(): size += p.stat().st_size
    if shutil.disk_usage(BASE).free < size + 2*2**30:
        raise RecoveryStopped("Insufficient disk for a separate result copy plus 2GiB reserve")
    shutil.copytree(original, destination)
    for job in jobs:
        if not 0 < job["step"] < 450000: continue
        # Sorting after original a_* names selects the explicit conversion at
        # the same step. All original attempts remain unchanged in the source.
        attempt = destination / job["variant"] / f"seed{job['seed']}" / "attempts" / ("zz_cu126_conversion_" + uuid.uuid4().hex[:8])
        cp = attempt / "resume" / f"step{job['step']:06d}"
        convert_copy(job["latest"], cp)
        from DQN.extended_train import check_resume
        state = check_resume(cp)
        fields = ("protocol", "variant", "seed", "num_frames", "novelty_beta", "device", "torch_version",
                  "cuda_version", "source_sha256", "platform", "runtime_assets", "software", "checkpoint_kind")
        write_json(attempt / "metadata.json", {**{k: state[k] for k in fields},
                   "parent_resume": job["latest"], "initial_step": job["step"],
                   "cuda_build_conversion": state["cuda_build_conversion"]})
    return destination


def resume_queue(root, config):
    import torch
    from DQN.extended_train import check_resume
    from Experiments.full_queue import run_queue
    def command(variant, seed, run, resume, persistent):
        if not torch.cuda.is_available(): raise RecoveryStopped("CUDA disappeared")
        if resume:
            require_build(check_resume(Path(resume)), NEW_TORCH)
        else:
            # A durable one-time marker also blocks retries/restarts before
            # the first 25K checkpoint or progress-log flush exists.
            marker = root / "cuda126_migration" / "zero_starts" / f"{variant}_seed{seed}.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            if marker.exists():
                raise RecoveryStopped(f"{variant} seed{seed} was already launched without a checkpoint. No zero-step restart is permitted.")
            with marker.open("x") as f:
                json.dump({"variant": variant, "seed": seed, "first_launch_utc": datetime.now(timezone.utc).isoformat(),
                           "run_directory": str(run), "policy": "No repeat start at zero without separate user approval"}, f)
            print(f"STARTING PREVIOUSLY UNSTARTED QUEUE ENTRY: {variant} seed{seed}", flush=True)
        cmd = [sys.executable, "-u", "-m", "DQN.extended_train", "--variant", variant, "--seed", str(seed),
               "--device", "cuda", "--run-dir", str(run), "--stop-at", str(config["total_actions"]),
               "--persistent-root", str(persistent)]
        if resume: cmd += ["--resume", str(resume)]
        return cmd
    print("VERIFIED CU126 MATRIX RESUME: completed jobs are checked and skipped.", flush=True)
    print("Local persistent results:", root, flush=True)
    return run_queue(config, root, BASE / "cu126_work", command_factory=command, require_cuda=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    os.chdir(BASE)
    audit = BASE / "cu126_audits" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    audit.mkdir(parents=True)
    report_path = audit / "RECOVERY_REPORT.json"
    report = {"status": "checking", "verification_succeeded": False, "matrix_started": False,
              "historical_checkpoints_modified": False, "audit_directory": str(audit),
              "limitations": "Bitwise equality is checked on saved historical intervals and a short restoration continuation. Future equivalence on different CUDA/GPU hardware is not guaranteed."}
    write_json(report_path, report)
    print("Recovery report:", report_path, flush=True)
    try:
        if platform.system() != "Linux": raise RecoveryStopped("Run this on the Linux RTX 3050 PC")
        from filelock import FileLock
        from linux_resume import verify_source
        source = verify_source()
        config = source["protocol"]
        with FileLock(str(BASE / ".cu126_recovery.lock"), timeout=0):
            report["gpu_check"] = gpu_check(config)
            report["adapter_sha256"] = sha(Path(__file__))
            report["original_core_manifest_sha256"] = sha(BASE / "FULL_SOURCE_MANIFEST.json")
            (audit / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True))
            report["pip_freeze_sha256"] = sha(audit / "pip_freeze.txt")
            print("CUDA126 PROJECT FORWARD/BACKWARD/ADAM PASSED:", report["gpu_check"]["gpu"], flush=True)
            write_json(report_path, report)
            active_path = BASE / "CU126_ACTIVE_RESULTS.json"
            if active_path.exists():
                active = json.loads(active_path.read_text())
                proof_path = Path(active["proof"])
                proof = json.loads(proof_path.read_text())
                if sha(proof_path) != active["proof_sha256"] or not proof.get("verification_succeeded"):
                    raise RecoveryStopped("Previous equivalence proof is missing or changed")
                for key in ("gpu_check", "adapter_sha256", "original_core_manifest_sha256", "pip_freeze_sha256"):
                    if proof[key] != report[key]: raise RecoveryStopped(f"Verified recovery environment changed: {key}")
                root = Path(active["root"])
                if args.checkpoints and locate_original(args.checkpoints) not in (Path(proof["original_checkpoint_root"]), root):
                    raise RecoveryStopped("--checkpoints differs from the previously verified original/result copy")
                report["verification_succeeded"] = True
                report["previous_proof"] = str(proof_path)
                report["jobs"] = scan(root, config, NEW_TORCH)
            else:
                original = locate_original(args.checkpoints)
                report["original_checkpoint_root"] = str(original)
                jobs = scan(original, config)
                report["jobs"] = jobs
                incomplete = [j for j in jobs if 0 < j["step"] < config["total_actions"]]
                if not incomplete:
                    raise RecoveryStopped("No interrupted job with historical checkpoint pairs is available. No experiment will start from zero.")
                report["comparisons"] = []
                for job in incomplete:
                    report["comparisons"].append(prove_job(job, config, audit / f"{job['variant']}_seed{job['seed']}"))
                    write_json(report_path, report)
                # Rescan to prevent a simultaneous Colab/other local run from
                # changing the source while proof was being computed.
                if scan(original, config) != jobs:
                    raise RecoveryStopped("Original progress changed during verification; stop the other queue first")
                root = fork_results(original, jobs, audit)
                report.update(verification_succeeded=True, results_root=str(root),
                    status="historical_interval_and_latest_restore_equivalence_passed")
                write_json(report_path, report)
                proof_path = root / "cuda126_migration/EQUIVALENCE_PASSED.json"
                write_json(proof_path, report)
                shutil.copy2(audit / "pip_freeze.txt", proof_path.parent / "pip_freeze.txt")
                for log in audit.rglob("*.log"):
                    target = proof_path.parent / "logs" / log.relative_to(audit)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(log, target)
                write_json(active_path, {"root": str(root), "proof": str(proof_path), "proof_sha256": sha(proof_path)})
            report["results_root"] = str(root)
            print("Local persistent results:", root, flush=True)
            if args.verify_only:
                report["status"] = "verified_no_matrix_launch_requested"
                write_json(report_path, report)
                print("VERIFICATION PASSED; --verify-only kept matrix training stopped.", flush=True)
                return
            report.update(status="resuming_matrix", matrix_started=True)
            write_json(report_path, report)
            resume_queue(root, config)
            report["status"] = "matrix_complete"
            write_json(report_path, report)
    except BaseException as exc:
        report.update(status="stopped", error=str(exc), exception=type(exc).__name__)
        write_json(report_path, report)
        print(f"STOPPED: {exc}\nNo original checkpoint was overwritten. Report: {report_path}", file=sys.stderr, flush=True)
        if isinstance(exc, KeyboardInterrupt): raise
        raise SystemExit(1)


if __name__ == "__main__":
    main()
