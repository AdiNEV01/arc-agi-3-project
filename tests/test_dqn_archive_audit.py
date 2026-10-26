"""Missing Drive parts and conflicting copies must never imply validated training."""
import io
import json
from pathlib import Path
import zipfile
import pytest
from Experiments.audit_dqn_503808_archive import audit, validate_history

REPO = Path(__file__).resolve().parents[1]


def minimal(path, extra=None):
    cfg = json.loads((REPO / "Configs/dqn_extended_v1.json").read_text())
    cfg["total_actions"] = 503808
    jobs = {f"{v}_seed{s}": dict(status="complete", completed_actions=503808)
            for v in cfg["variants"] for s in cfg["seeds"]}
    files = {"QUEUE.json": json.dumps(dict(protocol=cfg, source_sha256={}, queue_sources={})),
             "progress.json": json.dumps(dict(status="complete", jobs=jobs))}
    files.update(extra or {})
    with zipfile.ZipFile(path, "w") as z:
        for name, value in files.items(): z.writestr("result/"+name, value)


def test_complete_label_with_missing_payloads_stays_incomplete(tmp_path):
    part = tmp_path / "001.zip"
    minimal(part)
    result = audit([part], REPO)
    assert result["producer_claims_all_12_complete"]
    assert not result["ready_for_full_matrix_import"]
    assert result["counts"]["final_models_present"] == 0
    assert "v0/seed42/frozen/step503808.pt" in result["missing_for_matrix_import"]
    assert "EXTENSION_PROVENANCE.json" in result["missing_for_matrix_import"]


def test_multiple_parts_accept_identical_duplicates_and_reject_conflict(tmp_path):
    first, second = tmp_path / "001.zip", tmp_path / "002.zip"
    minimal(first)
    minimal(second)
    assert len(audit([first, second], REPO)["archives"]) == 2
    minimal(second, {"QUEUE.json": "different"})
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        audit([first, second], REPO)


def test_unsafe_zip_names_rejected(tmp_path):
    part = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(part, "w") as z: z.writestr("result/../../outside", "data")
    with pytest.raises(ValueError, match="Unsafe"):
        audit([part], REPO)


def test_incomplete_history_rejected():
    import gzip
    cfg = json.loads((REPO / "Configs/dqn_extended_v1.json").read_text())
    with pytest.raises(ValueError, match="Incomplete history"):
        validate_history(io.BytesIO(gzip.compress(b"global_step\n")), cfg, "v0")
