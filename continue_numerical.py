"""User-authorised numerical migration; exact restoration on the new GPU.

Install alongside the existing cu126 recovery delivery. Its learner, strict
source/build guards, environment and historical checkpoint files stay intact.
Cross-hardware equivalence is explicitly NOT required or claimed in this mode.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
BASE = Path(__file__).resolve().parent
import cu126_recovery as recovery


def verify_delivery():
    from linux_resume import verify_source
    source = verify_source()
    manifest = json.loads((BASE / "NUMERICAL_PATCH_MANIFEST.json").read_text())
    if recovery.sha(BASE / "LINUX_ADAPTER_MANIFEST.json") != manifest["base_adapter_manifest_sha256"]:
        raise recovery.RecoveryStopped("This patch requires the supplied cu126 recovery delivery")
    for name, expected in manifest["sha256"].items():
        if recovery.sha(BASE / name) != expected:
            raise recovery.RecoveryStopped(f"Numerical continuation patch changed: {name}")
    return source


def hardware_binding(gpu):
    return {k: v for k, v in gpu.items() if k != "smoke_losses_by_frame_count"}


def original_root(explicit):
    if explicit:
        return recovery.locate_original(explicit)
    # Reuse the original source selected by the failed audit, not its diagnostic
    # from_earlier/400010 output. Only a real queue root can be accepted by scan.
    reports = sorted((BASE / "cu126_audits").glob("*/RECOVERY_REPORT.json"), reverse=True)
    for path in reports:
        data = json.loads(path.read_text())
        candidate = Path(data.get("original_checkpoint_root", "/nonexistent"))
        if (candidate / "QUEUE.json").exists():
            return candidate.resolve()
    return recovery.locate_original(None)


def prove_latest_restore(job, config, out, runner=None):
    """Actual latest -> 10 vs latest -> 5 -> 10, all on the new GPU."""
    from DQN.extended_train import check_resume
    runner = runner or recovery.run_probe
    start = job["step"]
    end = min(start + 10, config["total_actions"])
    middle = start + (end - start) // 2
    if not start < middle < end:
        raise recovery.RecoveryStopped("Not enough remaining actions for the split-restoration check")
    out = Path(out)
    converted = out / "original_latest_converted"
    recovery.convert_copy(job["latest"], converted)
    # This comparison verifies exact input serialization; no numerical update
    # has happened, so every model/optimizer/RNG/replay/history value must match.
    input_check = recovery.compare_checkpoints(job["latest"], converted)
    print(f"CHECKING ACTUAL LATEST RESTORE: {job['variant']} seed{job['seed']} from {start}", flush=True)
    uninterrupted = runner(job["variant"], job["seed"], converted, end, out / "uninterrupted")
    prefix = runner(job["variant"], job["seed"], converted, middle, out / "prefix")
    restored = runner(job["variant"], job["seed"], prefix, end, out / "restored")
    comparison = recovery.compare_checkpoints(uninterrupted, restored)
    state = check_resume(restored)
    recovery.require_build(state, recovery.NEW_TORCH)
    for name in ("online_state_dict", "target_state_dict"):
        import torch
        if not all(torch.isfinite(t).all().item() for t in state[name].values()):
            raise recovery.RecoveryStopped("Nonfinite model after latest-state restoration")
    result = {"passed": True, "variant": job["variant"], "seed": job["seed"],
              "original_checkpoint": job["latest"], "original_step": start,
              "prefix_step": middle, "verified_step": end,
              "input_state_preserved": input_check, "same_gpu_split_restore": comparison,
              "interruption_kind": "controlled stop at a complete checkpoint, then a new training process",
              "historical_cross_hardware_bitwise_equivalence": False,
              "adopted_checkpoint": str(restored), "matrix_actions_adopted": end - start,
              "extra_diagnostic_actions_excluded_from_matrix": end - start}
    recovery.write_json(out / "LATEST_RESTORE_PASSED.json", result)
    print(f"SAME-GPU FULL RESTORATION PASSED: {start} -> {middle} -> {end}", flush=True)
    return result


def copy_verified_attempt(root, result, approval):
    """Adopt the verified 10-action continuation; do not train those actions twice."""
    from DQN.extended_train import check_resume
    source = Path(result["adopted_checkpoint"])
    state = check_resume(source)
    recovery.require_build(state, recovery.NEW_TORCH)
    job = root / result["variant"] / f"seed{result['seed']}"
    attempt = job / "attempts" / ("zz_numerical_verified_" + uuid.uuid4().hex[:10])
    target = attempt / "resume" / source.name
    shutil.copytree(source, target)
    recovery.compare_checkpoints(source, target)
    run = source.parent.parent
    for name in ("metadata.json", "summary.json"):
        if (run / name).exists(): shutil.copy2(run / name, attempt / name)
    recovery.write_json(attempt / "NUMERICAL_MIGRATION.json", approval)
    return {"variant": result["variant"], "seed": result["seed"], "step": result["verified_step"],
            "checkpoint": str(target), "complete_sha256": recovery.sha(target / "COMPLETE.json")}


def bindings(gpu):
    return {"hardware": hardware_binding(gpu),
            "patch_sha256": recovery.sha(Path(__file__)),
            "base_adapter_manifest_sha256": recovery.sha(BASE / "LINUX_ADAPTER_MANIFEST.json"),
            "core_manifest_sha256": recovery.sha(BASE / "FULL_SOURCE_MANIFEST.json")}


def run(args, audit, report):
    from DQN.extended_train import check_resume
    source = verify_delivery()
    config = source["protocol"]
    if config["total_actions"] != 450000:
        raise recovery.RecoveryStopped("This continuation preserves the original 450K protocol")
    gpu = recovery.gpu_check(config)
    report["gpu_check"] = gpu
    binding = bindings(gpu)
    pointer = BASE / "NUMERICAL_ACTIVE_RESULTS.json"
    if pointer.exists():
        active = json.loads(pointer.read_text())
        proof_path = Path(active["proof"])
        proof = json.loads(proof_path.read_text())
        if recovery.sha(proof_path) != active["proof_sha256"] or not proof.get("restoration_verified"):
            raise recovery.RecoveryStopped("Numerical restoration proof is missing or changed")
        if proof["bindings"] != binding:
            raise recovery.RecoveryStopped("Verified environment changed; refusing an unchecked continuation")
        root = Path(active["root"])
        if args.checkpoints and original_root(args.checkpoints) not in (Path(proof["original_checkpoint_root"]), root):
            raise recovery.RecoveryStopped("Checkpoint argument differs from the verified migration")
        recovery.scan(root, config, recovery.NEW_TORCH)
        report.update(results_root=str(root), restoration_verified=True, cached_proof=str(proof_path))
    else:
        if not args.accept_numerical_migration:
            raise recovery.RecoveryStopped("Use --accept-numerical-migration to record the approved change of numerical environment")
        original = original_root(args.checkpoints)
        jobs = recovery.scan(original, config)
        incomplete = [j for j in jobs if 0 < j["step"] < config["total_actions"]]
        if not incomplete:
            raise recovery.RecoveryStopped("No interrupted checkpoint found; refusing an unintended fresh matrix")
        approval = {"policy": "user_authorised_numerical_migration", "accepted": True,
                    "accepted_utc": datetime.now(timezone.utc).isoformat(),
                    "previous_environment": "T4 / PyTorch 2.11.0+cu130",
                    "new_environment": gpu,
                    "historical_bitwise_equivalence": False,
                    "known_difference": "Historical CUDA13/T4 replay differed at online_state_dict.encoder.0.weight",
                    "requirement": "Exact original checkpoint content plus verified full restoration on the new GPU",
                    "future_cross_hardware_identity_claimed": False}
        report.update(original_checkpoint_root=str(original), jobs=jobs, approval=approval, bindings=binding)
        recovery.write_json(audit / "CONTINUATION_REPORT.json", report)
        checks = []
        for job in incomplete:
            # No 25K historical replay. Use the actual latest original state.
            checks.append(prove_latest_restore(job, config, audit / f"{job['variant']}_seed{job['seed']}"))
            report["restoration_checks"] = checks
            recovery.write_json(audit / "CONTINUATION_REPORT.json", report)
        if recovery.scan(original, config) != jobs:
            raise recovery.RecoveryStopped("Original checkpoint source changed while checking restoration")
        # Legacy fork copies the entire source into a NEW, uniquely named tree.
        # audit.name starts NUMERICAL_ so the result directory labels the policy.
        root = recovery.fork_results(original, jobs, audit)
        adopted = [copy_verified_attempt(root, item, approval) for item in checks]
        # Record the numerical policy on every new build-metadata conversion.
        for path in root.glob("*/seed*/attempts/zz_cu126_conversion_*"):
            recovery.write_json(path / "NUMERICAL_MIGRATION.json", approval)
        recovery.scan(root, config, recovery.NEW_TORCH)
        proof = {"restoration_verified": True, "policy": approval, "bindings": binding,
                 "original_checkpoint_root": str(original), "results_root": str(root),
                 "restoration_checks": checks, "adopted_checkpoints": adopted,
                 "historical_bitwise_equivalence": False,
                 "original_checkpoints_modified": False}
        proof_path = root / "numerical_migration/RESTORATION_PASSED.json"
        recovery.write_json(proof_path, proof)
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
        (proof_path.parent / "pip_freeze.txt").write_text(freeze)
        for path in audit.rglob("*.log"):
            dest = proof_path.parent / "verification_logs" / path.relative_to(audit)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        recovery.write_json(pointer, {"root": str(root), "proof": str(proof_path),
                                      "proof_sha256": recovery.sha(proof_path)})
        report.update(results_root=str(root), restoration_verified=True, adopted_checkpoints=adopted)
    print("Local persistent results:", root, flush=True)
    report["status"] = "verified_no_queue_launch" if args.verify_only else "continuing_matrix"
    recovery.write_json(audit / "CONTINUATION_REPORT.json", report)
    if args.verify_only:
        print("RESTORATION VERIFIED; --verify-only kept the queue stopped.", flush=True)
        return
    report["matrix_launch_requested"] = True
    recovery.write_json(audit / "CONTINUATION_REPORT.json", report)
    recovery.resume_queue(root, config)
    report["status"] = "matrix_complete"
    recovery.write_json(audit / "CONTINUATION_REPORT.json", report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-numerical-migration", action="store_true")
    ap.add_argument("--checkpoints", type=Path)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    os.chdir(BASE)
    audit = BASE / "numerical_audits" / ("NUMERICAL_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    audit.mkdir(parents=True)
    report = {"status": "checking", "restoration_verified": False, "matrix_launch_requested": False,
              "historical_bitwise_equivalence": False, "original_checkpoints_modified": False}
    print("Continuation report:", audit / "CONTINUATION_REPORT.json", flush=True)
    try:
        if platform.system() != "Linux": raise recovery.RecoveryStopped("Run this on the existing Linux cu126 environment")
        from filelock import FileLock
        # Same lock as the strict launcher prevents simultaneous local queues.
        with FileLock(str(BASE / ".cu126_recovery.lock"), timeout=0):
            run(args, audit, report)
    except BaseException as exc:
        report.update(status="stopped", error=str(exc), exception=type(exc).__name__)
        recovery.write_json(audit / "CONTINUATION_REPORT.json", report)
        print(f"STOPPED: {exc}\nReport: {audit / 'CONTINUATION_REPORT.json'}", file=sys.stderr, flush=True)
        if isinstance(exc, KeyboardInterrupt): raise
        raise SystemExit(1)


if __name__ == "__main__":
    main()
