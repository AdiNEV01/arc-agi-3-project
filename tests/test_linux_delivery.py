"""Migration checks: exact source delivery, safe import, no lost-job restart."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

import linux_resume as launcher


@pytest.fixture
def delivery(tmp_path):
    source = Path(__file__).resolve().parents[1] / "Notebooks"
    if not all((source/name).is_file() for name in ("dqn_linux_resume.zip", "dqn_extended_full_source.zip")):
        pytest.skip("Historical delivery ZIPs are external artifacts; see docs/ARTIFACT_ACCESS.md")
    dest = tmp_path / "delivery"
    with zipfile.ZipFile(source / "dqn_linux_resume.zip") as z:
        z.extractall(dest)
    return source, dest


def test_original_source_preserved_and_tampering_rejected(delivery):
    source, dest = delivery
    with zipfile.ZipFile(source / "dqn_extended_full_source.zip") as old:
        for name in old.namelist():
            assert (dest / name).read_bytes() == old.read(name)
    assert launcher.verify_source(dest)["protocol"]["total_actions"] == 450000
    shell = (dest / "start_linux.sh").read_bytes()
    assert b"\r" not in shell
    (dest / "DQN/q_network.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Training source changed"):
        launcher.verify_source(dest)


def test_complete_download_and_path_escape(tmp_path):
    archive = tmp_path / "download.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("FULL_450K/QUEUE.json", "{}")
    cache = tmp_path / "imports"
    root = launcher.locate_queue(archive, cache)
    assert root.name == "FULL_450K" and (root / "QUEUE.json").exists()
    assert launcher.locate_queue(archive, cache) == root
    with zipfile.ZipFile(tmp_path / "unsafe.zip", "w") as z:
        z.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe checkpoint ZIP"):
        launcher.locate_queue(tmp_path / "unsafe.zip", cache)
    assert not (cache / "escape.txt").exists()
    for name in ("one", "two"):
        directory = tmp_path / "ambiguous" / name
        directory.mkdir(parents=True)
        (directory / "QUEUE.json").write_text("{}")
    with pytest.raises(ValueError, match="found 2"):
        launcher.locate_queue(tmp_path / "ambiguous", cache)


def test_partial_drive_download_cannot_restart_trained_job(delivery, monkeypatch, tmp_path):
    import sys
    import torch
    from DQN.extended_train import source_hashes
    import Experiments.full_queue as queue

    _, dest = delivery
    monkeypatch.chdir(dest)
    source = launcher.verify_source(dest)
    root = tmp_path / "partial_download"
    root.mkdir()
    (root / "QUEUE.json").write_text(json.dumps({"protocol": source["protocol"],
        "source_sha256": source_hashes(), "queue_sources": {}}))
    (root / "progress.json").write_text(json.dumps({"jobs": {"v0_seed42": {"completed_actions": 225000}}}))
    monkeypatch.setattr(launcher, "BASE", dest)
    monkeypatch.setattr(launcher, "verify_source", lambda: source)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test GPU")
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **k: "test driver")
    monkeypatch.setattr(queue, "run_queue", lambda *a, **k: pytest.fail("A partial import must never launch training"))
    monkeypatch.setattr(sys, "argv", ["linux_resume.py", "--checkpoints", str(root)])
    with pytest.raises(ValueError, match="Missing downloaded checkpoints.*225000.*found 0"):
        launcher.main()
