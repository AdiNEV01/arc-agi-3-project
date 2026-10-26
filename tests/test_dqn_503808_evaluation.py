import hashlib
import json
from pathlib import Path
import pytest
import yaml
from Experiments.evaluate_dqn_503808 import validate_registry, materialize
from Experiments.audit_dqn_503808_archive import VARIANTS, SEEDS


@pytest.fixture
def registry(tmp_path):
    stage=tmp_path/"artifacts/dqn_extended/test/evaluation_stage"
    rows=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            path=stage/f"{variant}_seed{seed}/frozen/step503808.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"synthetic")
            rows.append(dict(variant=variant,training_seed=seed,checkpoint_step=503808,
                             path=path.relative_to(tmp_path).as_posix(),sha256=hashlib.sha256(b"synthetic").hexdigest()))
    cfg=tmp_path/"Configs/dqn_extended_development.yaml"
    cfg.parent.mkdir()
    cfg.write_text((Path(__file__).resolve().parents[1]/"Configs/dqn_extended_development.yaml").read_text())
    return tmp_path,dict(ignored_evaluation_staging=stage.relative_to(tmp_path).as_posix(),models=rows)


def test_full_matrix_and_all_hashes_required(registry):
    repo,reg=registry
    assert len(validate_registry(repo,reg,[503808])[1])==12
    (repo/reg["models"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError,match="checksum differs"): validate_registry(repo,reg,[503808])


def test_partial_matrix_and_reserved_protocol_rejected(registry):
    repo,reg=registry
    last=reg["models"].pop()
    with pytest.raises(ValueError,match="entire"): validate_registry(repo,reg,[503808])
    reg["models"].append(last)
    p=repo/"Configs/dqn_extended_development.yaml"
    cfg=yaml.safe_load(p.read_text());cfg["games"]["dev"][0]="sb26-7fbdac44"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match="protocol changed"): validate_registry(repo,reg,[503808])


def test_wrong_archive_rejected_before_materializing(registry,tmp_path):
    repo,reg=registry
    archive=tmp_path/"wrong.zip";archive.write_bytes(b"not the archive")
    reg["archive"]={"sha256":"wrong"}
    with pytest.raises(ValueError,match="archive checksum"): materialize(repo,reg,archive)
