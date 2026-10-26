"""Real CPU restoration checks for the new driver; GPU runs on the Linux PC."""
from argparse import Namespace
import json
from pathlib import Path
import shutil

import pytest
import torch

import continue_numerical as migration
import cu126_recovery as recovery
from DQN.extended_train import check_resume, train
from tests.fake_arc import FakeArcade
from tests.test_dqn_extended import tiny_config


@pytest.fixture
def latest_cpu(tmp_path, monkeypatch):
    cfg = tiny_config()
    cfg.update(variants=["v2"], seeds=[42])
    source = tmp_path / "original"
    train(cfg, "v2", 42, "cpu", source, stop_at=12, arcade_factory=FakeArcade)
    cp = source / "resume/step000012"
    # Production build-conversion checks are covered by test_cu126_recovery.
    # Here the real CPU trainer must see the real CPU build metadata.
    monkeypatch.setattr(recovery, "convert_copy", lambda a, b: shutil.copytree(a, b))
    monkeypatch.setattr(recovery, "require_build", lambda state, build: None)
    return cfg, cp


def test_actual_latest_split_restoration_and_adoption(latest_cpu, tmp_path):
    cfg, cp = latest_cpu
    before = {p.name: recovery.sha(p) for p in cp.iterdir() if p.is_file()}
    calls = []
    def runner(variant, seed, resume, stop, out):
        calls.append((check_resume(Path(resume))["global_step"], stop))
        train(cfg, variant, seed, "cpu", out, stop_at=stop, resume=resume, arcade_factory=FakeArcade)
        return Path(out) / "resume" / f"step{stop:06d}"
    job = {"variant": "v2", "seed": 42, "step": 12, "latest": str(cp)}
    result = migration.prove_latest_restore(job, cfg, tmp_path / "probe", runner)
    assert calls == [(12, 22), (12, 17), (17, 22)]
    assert result["passed"] and result["verified_step"] == 22
    assert result["matrix_actions_adopted"] == 10
    assert result["extra_diagnostic_actions_excluded_from_matrix"] == 10
    assert not result["historical_cross_hardware_bitwise_equivalence"]
    adopted = migration.copy_verified_attempt(tmp_path / "results", result, {"accepted": True})
    assert check_resume(Path(adopted["checkpoint"]))["global_step"] == 22
    assert recovery.compare_checkpoints(result["adopted_checkpoint"], adopted["checkpoint"])["passed"]
    assert before == {p.name: recovery.sha(p) for p in cp.iterdir() if p.is_file()}


def test_failed_same_gpu_restore_does_not_pass(latest_cpu, tmp_path):
    cfg, cp = latest_cpu
    def runner(variant, seed, resume, stop, out):
        train(cfg, variant, seed, "cpu", out, stop_at=stop, resume=resume, arcade_factory=FakeArcade)
        result = Path(out) / "resume" / f"step{stop:06d}"
        if Path(out).name == "restored":
            state = check_resume(result)
            next(iter(state["optimizer_state_dict"]["state"].values()))["exp_avg"].reshape(-1)[0] += 1
            torch.save(state, result / "state.pt")
            marker = json.loads((result / "COMPLETE.json").read_text())
            marker["sha256"]["state.pt"] = recovery.sha(result / "state.pt")
            recovery.write_json(result / "COMPLETE.json", marker)
        return result
    with pytest.raises(recovery.RecoveryStopped, match="optimizer_state_dict"):
        migration.prove_latest_restore({"variant": "v2", "seed": 42, "step": 12, "latest": str(cp)}, cfg, tmp_path / "probe", runner)
    assert not (tmp_path / "probe/LATEST_RESTORE_PASSED.json").exists()


def test_restoration_failure_blocks_copy_and_queue(tmp_path, monkeypatch):
    (tmp_path / "LINUX_ADAPTER_MANIFEST.json").write_text("{}")
    (tmp_path / "FULL_SOURCE_MANIFEST.json").write_text("{}")
    cfg = tiny_config(); cfg["total_actions"] = 450000
    monkeypatch.setattr(migration, "BASE", tmp_path)
    monkeypatch.setattr(migration, "verify_delivery", lambda: {"protocol": cfg})
    monkeypatch.setattr(recovery, "gpu_check", lambda _: {"passed": True})
    monkeypatch.setattr(migration, "bindings", lambda _: {})
    monkeypatch.setattr(migration, "original_root", lambda _: tmp_path / "original")
    monkeypatch.setattr(recovery, "scan", lambda *a: [{"variant": "v2", "seed": 42, "step": 400000}])
    def fail(*a): raise recovery.RecoveryStopped("same-GPU restoration failed")
    monkeypatch.setattr(migration, "prove_latest_restore", fail)
    monkeypatch.setattr(recovery, "fork_results", lambda *a: pytest.fail("No result fork after failed restoration"))
    monkeypatch.setattr(recovery, "resume_queue", lambda *a: pytest.fail("No queue after failed restoration"))
    report = {"matrix_launch_requested": False}
    args = Namespace(accept_numerical_migration=True, checkpoints=None, verify_only=False)
    with pytest.raises(recovery.RecoveryStopped, match="same-GPU restoration failed"):
        migration.run(args, tmp_path / "audit", report)
    assert not report["matrix_launch_requested"]
    assert not (tmp_path / "NUMERICAL_ACTIVE_RESULTS.json").exists()


def test_cached_proof_and_progress_resume_without_repeating_probes(tmp_path, monkeypatch):
    cfg = tiny_config(); cfg["total_actions"] = 450000
    monkeypatch.setattr(migration, "BASE", tmp_path)
    monkeypatch.setattr(migration, "verify_delivery", lambda: {"protocol": cfg})
    monkeypatch.setattr(recovery, "gpu_check", lambda _: {"passed": True})
    monkeypatch.setattr(migration, "bindings", lambda _: {"same": True})
    proof_path = tmp_path / "results/numerical_migration/RESTORATION_PASSED.json"
    recovery.write_json(proof_path, {"restoration_verified": True, "bindings": {"same": True},
                                   "original_checkpoint_root": str(tmp_path / "original")})
    recovery.write_json(tmp_path / "NUMERICAL_ACTIVE_RESULTS.json", {"root": str(tmp_path / "results"),
        "proof": str(proof_path), "proof_sha256": recovery.sha(proof_path)})
    monkeypatch.setattr(recovery, "scan", lambda *a: [])
    monkeypatch.setattr(migration, "prove_latest_restore", lambda *a: pytest.fail("Do not repeat verified restoration"))
    launches = []
    monkeypatch.setattr(recovery, "resume_queue", lambda root, config: launches.append(root))
    args = Namespace(accept_numerical_migration=False, checkpoints=None, verify_only=False)
    report = {"matrix_launch_requested": False}
    migration.run(args, tmp_path / "audit", report)
    assert launches == [tmp_path / "results"]
    assert report["restoration_verified"] and report["status"] == "matrix_complete"
