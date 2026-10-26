"""CPU checks of migration safety and real learner-state comparison.

Synthetic CUDA metadata is used only for conversion-policy tests; these tests
make no claim that CUDA or a historical Colab trajectory was verified locally.
"""
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

import cu126_recovery as recovery
from DQN.extended_train import check_resume, train, source_hashes, runtime_hashes
from tests.fake_arc import FakeArcade
from tests.test_dqn_extended import tiny_config


@pytest.fixture(scope="module")
def real_states(tmp_path_factory):
    root = tmp_path_factory.mktemp("cu126_real_cpu_states")
    cfg = tiny_config()
    cfg.update(variants=["v2"], seeds=[42])
    train(cfg, "v2", 42, "cpu", root / "full", arcade_factory=FakeArcade)
    train(cfg, "v2", 42, "cpu", root / "prefix", stop_at=12, arcade_factory=FakeArcade)
    train(cfg, "v2", 42, "cpu", root / "restored", resume=root / "prefix/resume/step000012", arcade_factory=FakeArcade)
    return cfg, root


def commit_state(path, state):
    torch.save(state, path / "state.pt")
    recovery.write_json(path / "COMPLETE.json", {"step": state["global_step"], "sha256": {
        n: recovery.sha(path / n) for n in ("state.pt", "replay.npz", "history.csv.gz")}})


def synthetic_cuda_copy(source, destination):
    shutil.copytree(source, destination)
    state = check_resume(destination)
    state.update(software={**recovery.PINNED, "torch": recovery.OLD_TORCH},
                 torch_version=recovery.OLD_TORCH, cuda_version="13.0", device_type="cuda",
                 cuda_rng=[torch.zeros(16, dtype=torch.uint8)])
    commit_state(destination, state)


def test_real_full_state_resume_comparison(real_states):
    _, root = real_states
    result = recovery.compare_checkpoints(root / "full/resume/step000030", root / "restored/resume/step000030")
    assert result["passed"] and result["history_rows"] == 30
    assert "optimizer_state_dict" in result["state_fields_bitwise_equal"]
    assert "numpy_rng" in result["state_fields_bitwise_equal"]
    assert len(result["replay_array_sha256"]) == 6


@pytest.mark.parametrize("field", ["optimizer_state_dict", "agent_rng"])
def test_optimizer_and_rng_drift_cannot_pass(real_states, tmp_path, field):
    _, root = real_states
    original = root / "full/resume/step000030"
    candidate = tmp_path / "changed"
    shutil.copytree(original, candidate)
    state = check_resume(candidate)
    if field == "optimizer_state_dict":
        next(iter(state[field]["state"].values()))["exp_avg"].reshape(-1)[0] += 1
    else:
        state[field] = (4, *state[field][1:])
    commit_state(candidate, state)
    with pytest.raises(recovery.RecoveryStopped, match=field):
        recovery.compare_checkpoints(original, candidate)


def test_replay_mask_drift_cannot_pass(real_states, tmp_path):
    _, root = real_states
    original = root / "full/resume/step000030"
    candidate = tmp_path / "changed"
    shutil.copytree(original, candidate)
    with np.load(candidate / "replay.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays["masks"][0, 0] = ~arrays["masks"][0, 0]
    np.savez_compressed(candidate / "replay.npz", **arrays)
    commit_state(candidate, check_resume(original))
    with pytest.raises(recovery.RecoveryStopped, match="replay arrays"):
        recovery.compare_checkpoints(original, candidate)


def test_only_wall_clock_history_is_ignored(real_states, tmp_path):
    _, root = real_states
    original = root / "full/resume/step000030"
    candidate = tmp_path / "changed"
    shutil.copytree(original, candidate)
    with gzip.open(candidate / "history.csv.gz", "rt", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["elapsed_seconds"] = "999999.0"
    def save():
        with gzip.open(candidate / "history.csv.gz", "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        commit_state(candidate, check_resume(original))
    save()
    assert recovery.compare_checkpoints(original, candidate)["passed"]
    rows[3]["loss"] = "123.0"
    save()
    with pytest.raises(recovery.RecoveryStopped, match="history row 4.*loss"):
        recovery.compare_checkpoints(original, candidate)


def test_bitwise_gate_rejects_signed_zero():
    with pytest.raises(recovery.RecoveryStopped, match="tensor bits"):
        recovery.assert_bits(torch.tensor([0.0]), torch.tensor([-0.0]))


def test_explicit_conversion_preserves_originals_and_does_not_spoof_torch(real_states, tmp_path):
    _, root = real_states
    original = tmp_path / "original"
    synthetic_cuda_copy(root / "prefix/resume/step000012", original)
    before = {p.name: recovery.sha(p) for p in original.iterdir() if p.is_file()}
    actual_torch = str(torch.__version__)
    destination = tmp_path / "converted"
    origin = recovery.convert_copy(original, destination)
    assert before == {p.name: recovery.sha(p) for p in original.iterdir() if p.is_file()}
    assert str(torch.__version__) == actual_torch
    converted = check_resume(destination)
    assert converted["torch_version"] == recovery.NEW_TORCH
    assert origin["original_software"]["torch"] == recovery.OLD_TORCH
    assert recovery.compare_checkpoints(original, destination)["passed"]
    with pytest.raises(FileExistsError):
        recovery.convert_copy(original, destination)
    with pytest.raises(recovery.RecoveryStopped, match="Only cu130"):
        recovery.convert_copy(original, tmp_path / "disallowed", target="2.11.0+cu128")
    changed = check_resume(original)
    changed["software"]["numpy"] = "2.0.0"
    commit_state(original, changed)
    with pytest.raises(recovery.RecoveryStopped, match="Unexpected checkpoint software"):
        recovery.convert_copy(original, tmp_path / "wrong_numpy")


def definition(cfg):
    return {"protocol": cfg, "source_sha256": source_hashes(),
        "runtime_assets": runtime_hashes(cfg["training_games"]),
        "queue_sources": {p: recovery.sha(Path(p)) for p in ("Experiments/full_queue.py",
            "Experiments/full_preflight.py", "Experiments/drive_storage.py")}}


def test_migrated_scan_retains_old_generations_and_selects_new_build(real_states, tmp_path):
    cfg, root = real_states
    queue = tmp_path / "matrix"
    old = queue / "v2/seed42/attempts/a_original/resume/step000012"
    old.parent.mkdir(parents=True)
    synthetic_cuda_copy(root / "prefix/resume/step000012", old)
    converted = queue / "v2/seed42/attempts/zz_cu126/resume/step000012"
    recovery.convert_copy(old, converted)
    recovery.write_json(queue / "QUEUE.json", definition(cfg))
    recovery.write_json(queue / "progress.json", {"jobs": {"v2_seed42": {"completed_actions": 12}}})
    jobs = recovery.scan(queue, cfg, recovery.NEW_TORCH)
    assert jobs[0]["latest"] == str(converted) and jobs[0]["step"] == 12
    assert old.exists()


def test_activity_without_checkpoint_is_not_restarted(real_states, tmp_path):
    cfg, _ = real_states
    recovery.write_json(tmp_path / "QUEUE.json", definition(cfg))
    recovery.write_json(tmp_path / "progress.json", {"jobs": {"v2_seed42": {
        "completed_actions": 0, "observed_actions": 12000}}})
    with pytest.raises(recovery.RecoveryStopped, match="refusing to restart from zero"):
        recovery.scan(tmp_path, cfg)
    with pytest.raises(recovery.RecoveryStopped, match="Two distinct original checkpoints"):
        recovery.prove_job({"generations": [(25000, "old")], "variant": "v2", "seed": 42}, cfg, tmp_path)


def test_failed_cuda_check_never_converts_or_launches_matrix(monkeypatch, tmp_path):
    import linux_resume
    cfg = tiny_config()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recovery, "BASE", tmp_path)
    monkeypatch.setattr(recovery.platform, "system", lambda: "Linux")
    monkeypatch.setattr(linux_resume, "verify_source", lambda: {"protocol": cfg})
    def no_cuda(_): raise recovery.RecoveryStopped("CUDA unavailable")
    monkeypatch.setattr(recovery, "gpu_check", no_cuda)
    monkeypatch.setattr(recovery, "convert_copy", lambda *a, **k: pytest.fail("No conversion before CUDA succeeds"))
    monkeypatch.setattr(recovery, "resume_queue", lambda *a, **k: pytest.fail("No matrix before equivalence"))
    monkeypatch.setattr(sys, "argv", ["cu126_recovery.py"])
    with pytest.raises(SystemExit) as failure:
        recovery.main()
    assert failure.value.code == 1
    reports = list((tmp_path / "cu126_audits").glob("*/RECOVERY_REPORT.json"))
    report = json.loads(reports[0].read_text())
    assert report["status"] == "stopped" and not report["verification_succeeded"] and not report["matrix_started"]
    assert not (tmp_path / "CU126_ACTIVE_RESULTS.json").exists()


def test_new_pending_job_cannot_be_retried_from_zero(monkeypatch, tmp_path):
    import Experiments.full_queue as queue
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def controller(cfg, root, local, *, command_factory, **kwargs):
        cmd = command_factory("v2", 42, tmp_path / "first", None, tmp_path / "attempts")
        assert "--resume" not in cmd and cmd[cmd.index("--device") + 1] == "cuda"
        with pytest.raises(recovery.RecoveryStopped, match="No zero-step restart"):
            command_factory("v2", 42, tmp_path / "second", None, tmp_path / "attempts")
        return "checked"
    monkeypatch.setattr(queue, "run_queue", controller)
    assert recovery.resume_queue(tmp_path, tiny_config()) == "checked"
