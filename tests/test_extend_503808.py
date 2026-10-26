"""Full-state budget extension and a real fail/resume queue; CPU fixtures only."""
import json
from pathlib import Path
import sys

import pytest
import torch

import extend_503808 as extension
import cu126_recovery as recovery
from DQN.extended_train import check_resume, train
from Experiments.full_queue import validate_complete, run_queue
from tests.fake_arc import FakeArcade
from tests.test_dqn_extended import tiny_config


def config(variant="v0"):
    value = tiny_config()
    value.update(variants=[variant], seeds=[42], epsilon_decay=12)
    return value


@pytest.mark.parametrize("variant", ["v0", "v1", "v2_history_only", "v2"])
def test_extension_equals_uninterrupted_full_learner_for_all_variants(tmp_path, variant):
    old = config(variant)
    new = extension.extension_config(old, 45)
    train(old, variant, 42, "cpu", tmp_path / "old", arcade_factory=FakeArcade)
    source = tmp_path / "old/resume/step000030"
    before = {p.name: recovery.sha(p) for p in source.iterdir() if p.is_file()}
    state = check_resume(source)
    derived = tmp_path / "derived"
    event = extension.derive_checkpoint(source, derived, old, new, state["software"], None)
    assert not event["reset_model_optimizer_replay_rng_history_or_novelty"]
    assert not event["cuda_build_change"]
    assert check_resume(derived)["replay_position"] == state["replay_position"]
    train(new, variant, 42, "cpu", tmp_path / "resumed", resume=derived, arcade_factory=FakeArcade)
    train(new, variant, 42, "cpu", tmp_path / "baseline", arcade_factory=FakeArcade)
    final = tmp_path / "resumed/resume/step000045"
    assert recovery.compare_checkpoints(tmp_path / "baseline/resume/step000045", final)["passed"]
    saved = check_resume(final)
    assert saved["global_step"] == 45 and saved["gradient_steps"] == 21
    assert saved["replay_position"] == 45 % old["buffer_capacity"]
    assert before == {p.name: recovery.sha(p) for p in source.iterdir() if p.is_file()}
    assert (tmp_path / "old/frozen/step000030.pt").exists()


def test_only_budget_fields_change_and_actual_counts():
    old = json.loads(Path("Configs/dqn_extended_v1.json").read_text())
    new = extension.extension_config(old)
    assert new["total_actions"] == 503808
    assert [s for s in new["checkpoint_actions"] if s not in old["checkpoint_actions"]] == [503808]
    assert {k for k in old if old[k] != new[k]} == extension.CAP_KEYS
    assert extension.gradient_count(old, 450000) == 89801
    assert extension.gradient_count(new, 503808) == 100562
    assert 503808 - 450000 == 53808 and 12 * 53808 == 645696
    assert 503808 // 3 == 167936


def test_rule_changes_and_zero_start_are_rejected(tmp_path):
    old = config()
    train(old, "v0", 42, "cpu", tmp_path / "old", arcade_factory=FakeArcade)
    source = tmp_path / "old/resume/step000030"
    software = check_resume(source)["software"]
    changed = extension.extension_config(old, 45)
    changed["gamma"] = 0.9
    with pytest.raises(recovery.RecoveryStopped, match="beyond the budget"):
        extension.derive_checkpoint(source, tmp_path / "bad", old, changed, software, None)
    with pytest.raises(recovery.RecoveryStopped, match="never starts at zero"):
        extension.extension_command({}, "missing.json", "v0", 42, tmp_path, None, tmp_path)


def test_completed_jobs_extend_then_resume_failure_and_skip_final(tmp_path, monkeypatch):
    old = config()
    new = extension.extension_config(old, 45)
    original = tmp_path / "o"
    job = original / "v0/seed42"
    train(old, "v0", 42, "cpu", tmp_path / "a", persistent=job / "attempts", arcade_factory=FakeArcade)
    cp = job / "attempts/a/resume/step000030"
    assert validate_complete(job, old, "v0", 42, cp)
    recovery.write_json(original / "QUEUE.json", extension.queue_definition(old))
    recovery.write_json(original / "progress.json", {"status": "complete", "jobs": {
        "v0_seed42": {"status": "complete", "completed_actions": 30}}})
    # Only the CPU fixture's software layer is substituted. Real history,
    # receipts, checkpoints, model bits, source and snapshot checks still run.
    monkeypatch.setattr(recovery, "NEW_TORCH", str(torch.__version__))
    monkeypatch.setattr(recovery, "require_build", lambda *a: None)
    before = {str(p.relative_to(original)): recovery.sha(p) for p in original.rglob("*") if p.is_file()}
    jobs = extension.verified_completed_source(original, old)
    state = check_resume(cp)
    root = tmp_path / "n"
    extension.prepare_result_tree(original, jobs, old, new, {"software": state["software"], "cuda_runtime": None}, root)
    assert not (root / "v0/seed42/RUN_COMPLETE.json").exists()
    assert (root / "v0/seed42/source_450k/RUN_COMPLETE.json").exists()
    assert recovery.sha(root / "v0/seed42/frozen/step000030.pt") == recovery.sha(job / "frozen/step000030.pt")
    sentinel = tmp_path / "sentinel"
    calls = []
    def command(variant, seed, run, resume, persistent):
        assert resume is not None
        calls.append(check_resume(Path(resume))["global_step"])
        return [sys.executable, "-m", "tests.extension_queue_worker", "--config", str(root / "extension_config.json"),
            "--variant", variant, "--seed", str(seed), "--run", str(run), "--persistent", str(persistent),
            "--resume", str(resume), "--sentinel", str(sentinel)]
    final = run_queue(new, root, tmp_path / "w", command_factory=command, require_cuda=False, poll_seconds=0.05)
    assert final["status"] == "complete" and calls == [30, 36]
    complete = json.loads((root / "v0/seed42/RUN_COMPLETE.json").read_text())
    assert complete["actions"] == 45 and len(complete["snapshots"]) == 3
    calls.clear()
    run_queue(new, root, tmp_path / "w", command_factory=command, require_cuda=False)
    assert calls == []
    assert before == {str(p.relative_to(original)): recovery.sha(p) for p in original.rglob("*") if p.is_file()}


def test_incomplete_source_is_not_extended(tmp_path):
    old = config()
    recovery.write_json(tmp_path / "QUEUE.json", extension.queue_definition(old))
    recovery.write_json(tmp_path / "progress.json", {"status": "stopped", "jobs": {}})
    with pytest.raises(recovery.RecoveryStopped, match="entire original matrix must be complete"):
        extension.verified_completed_source(tmp_path, old)


def test_explicit_build_metadata_conversion_preserves_all_dynamic_state(tmp_path):
    # Synthetic CUDA build tags on a real CPU state exercise metadata policy
    # only; actual CUDA execution is still checked by the Linux launcher.
    old = config()
    train(old, "v0", 42, "cpu", tmp_path / "o", arcade_factory=FakeArcade)
    source = tmp_path / "o/resume/step000030"
    state = check_resume(source)
    state.update(torch_version=recovery.OLD_TORCH, cuda_version="13.0",
                 software={**state["software"], "torch": recovery.OLD_TORCH})
    torch.save(state, source / "state.pt")
    marker = json.loads((source / "COMPLETE.json").read_text())
    marker["sha256"]["state.pt"] = recovery.sha(source / "state.pt")
    recovery.write_json(source / "COMPLETE.json", marker)
    before = {p.name: recovery.sha(p) for p in source.iterdir() if p.is_file()}
    new = extension.extension_config(old, 45)
    software = {**state["software"], "torch": recovery.NEW_TORCH}
    event = extension.derive_checkpoint(source, tmp_path / "d", old, new, software, "12.6")
    converted = check_resume(tmp_path / "d")
    assert event["cuda_build_change"] and converted["torch_version"] == recovery.NEW_TORCH
    for key in recovery.STATE_FIELDS:
        if key not in ("protocol", "config"):
            recovery.assert_bits(state[key], converted[key], key)
    with pytest.raises(recovery.RecoveryStopped, match="Unapproved software change"):
        extension.derive_checkpoint(source, tmp_path / "bad", old, new, {**software, "numpy": "different"}, "12.6")
    assert before == {p.name: recovery.sha(p) for p in source.iterdir() if p.is_file()}
