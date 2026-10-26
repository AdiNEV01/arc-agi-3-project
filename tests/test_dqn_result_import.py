"""Preserve original evidence; reject unaudited imports and tampered binaries."""
import hashlib
import json
import zipfile
import pytest
from Experiments.import_dqn_503808_results import copy_member, curate


def test_unaudited_import_writes_nothing(tmp_path):
    audit=tmp_path/"audit.json"
    audit.write_text(json.dumps({"ready_for_full_matrix_import":False}))
    out,raw=tmp_path/"out",tmp_path/"raw"
    with pytest.raises(ValueError, match="complete successful"):
        curate(tmp_path/"missing.zip",audit,tmp_path,out,raw)
    assert not out.exists() and not raw.exists()


def test_member_checksum_and_no_overwrite(tmp_path):
    archive=tmp_path/"data.zip"
    with zipfile.ZipFile(archive,"w") as z: z.writestr("model.pt",b"synthetic")
    dest=tmp_path/"copy.pt"
    with zipfile.ZipFile(archive) as z:
        with pytest.raises(ValueError,match="differs"):
            copy_member(z,"model.pt",dest,"wrong")
        assert not dest.exists()
        expected=hashlib.sha256(b"synthetic").hexdigest()
        copy_member(z,"model.pt",dest,expected)
        with pytest.raises(FileExistsError): copy_member(z,"model.pt",dest,expected)
    assert dest.read_bytes()==b"synthetic"
