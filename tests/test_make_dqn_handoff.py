"""Integrity/selection tests with synthetic receipts and byte payloads, no ML claims."""
import gzip
import json
from pathlib import Path
import zipfile
import pytest
import make_dqn_handoff as handoff


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def completed(tmp_path):
    work = tmp_path / "work"
    root = work / "finished"
    write(work / "EXTENSION_503808_ACTIVE.json", {"root": str(root)})
    write(root / "QUEUE.json", {"protocol": {"total_actions": 503808,
        "variants": list(handoff.VARIANTS), "seeds": list(handoff.SEEDS)}})
    write(root / "extension_config.json", {})
    write(root / "EXTENSION_PROVENANCE.json", {})
    progress = {"status": "complete", "jobs": {}}
    for variant in handoff.VARIANTS:
        for seed in handoff.SEEDS:
            job = root / variant / f"seed{seed}"
            snapshots = {}
            for step in (450000, 503808):
                path = job / "frozen" / f"step{step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"synthetic model {variant} {seed} {step}".encode())
                snapshots[str(step)] = {"sha256": handoff.sha(path)}
            write(job / "RUN_COMPLETE.json", {"validated": True, "actions": 503808,
                "gradient_steps": 100562, "variant": variant, "seed": seed, "snapshots": snapshots})
            write(job / "metadata.json", {"variant": variant, "seed": seed})
            (job / "actions.csv.gz").write_bytes(gzip.compress(b"synthetic canonical history"))
            cp = job / "attempts/a/resume/step503808"
            write(cp / "COMPLETE.json", {"step": 503808, "sha256": {
                "state.pt": "excluded", "replay.npz": "excluded", "history.csv.gz": handoff.sha(job / "actions.csv.gz")}})
            (cp / "state.pt").write_bytes(b"full resume must not be exported")
            (cp / "replay.npz").write_bytes(b"large replay must not be exported")
            progress["jobs"][f"{variant}_seed{seed}"] = {"status": "complete", "completed_actions": 503808}
    write(root / "progress.json", progress)
    for name in ("FULL_SOURCE_MANIFEST.json", "LINUX_ADAPTER_MANIFEST.json", "NUMERICAL_PATCH_MANIFEST.json", "EXTENSION_503808_MANIFEST.json"):
        write(work / name, {})
    return work, root


def test_single_handoff_has_both_endpoints_and_full_logs_without_replays(completed, tmp_path):
    work, root = completed
    before = {str(p.relative_to(work)): handoff.sha(p) for p in work.rglob("*") if p.is_file()}
    result = handoff.package(work, tmp_path / "out")
    with zipfile.ZipFile(result["archive"]) as z:
        names = z.namelist()
        assert len([n for n in names if n.endswith(".pt")]) == 24
        assert len([n for n in names if n.endswith("actions.csv.gz")]) == 12
        assert not any(n.endswith(("state.pt", "replay.npz")) for n in names)
        manifest = json.loads(z.read("HANDOFF_MANIFEST.json"))
        assert manifest["included_model_steps"] == [450000, 503808]
        assert not manifest["full_resume_state_and_replay_included"]
    assert before == {str(p.relative_to(work)): handoff.sha(p) for p in work.rglob("*") if p.is_file()}


def test_corrupt_frozen_snapshot_is_rejected(completed, tmp_path):
    work, root = completed
    (root / "v0/seed42/frozen/step503808.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Frozen model checksum differs"):
        handoff.package(work, tmp_path / "out")


def test_incomplete_matrix_is_rejected(completed, tmp_path):
    work, root = completed
    progress = json.loads((root / "progress.json").read_text())
    progress["jobs"]["v2_seed456"]["completed_actions"] = 500000
    write(root / "progress.json", progress)
    with pytest.raises(ValueError, match="Incomplete job"):
        handoff.package(work, tmp_path / "out")
