import json
from pathlib import Path
import sys
import pytest

from DQN.extended_train import hash_file,json_write,publish
from Experiments.full_queue import run_queue,verified_generations,classify_failure,UnsafeRecovery
from tests.test_dqn_extended import tiny_config


def test_real_subprocess_failure_recovery_and_skip(tmp_path):
    c=tiny_config()
    c.update(variants=["v0"],seeds=[42],checkpoint_actions=[12,24,30])
    cfg=tmp_path/"tiny.json";cfg.write_text(json.dumps(c))
    sentinel=tmp_path/"failed_once"
    launches=[]
    def command(variant,seed,run,resume,persistent):
        launches.append(resume)
        result=[sys.executable,"-m","tests.queue_smoke_worker","--config",str(cfg),"--run",str(run),
            "--persistent",str(persistent),"--sentinel",str(sentinel),"--variant",variant,"--seed",str(seed)]
        if resume:result += ["--resume",str(resume)]
        return result
    root=tmp_path/"drive/full"
    result=run_queue(c,root,tmp_path/"local",command_factory=command,require_cuda=False,poll_seconds=0.05)
    assert result["status"]=="complete"
    assert len(launches)==2 and launches[0] is None and launches[1] is not None
    job=root/"v0/seed42"
    complete=json.loads((job/"RUN_COMPLETE.json").read_text())
    assert complete["actions"]==30 and complete["gradient_steps"]==14
    assert len(complete["snapshots"])==3
    generations,invalid=verified_generations(job,c,"v0",42)
    assert [v[0] for v in generations]==[24,30] and not invalid
    launches.clear()
    result=run_queue(c,root,tmp_path/"local",command_factory=command,require_cuda=False,poll_seconds=0.05)
    assert result["status"]=="complete" and not launches
    changed={**c,"epsilon_decay":100}
    with pytest.raises(UnsafeRecovery,match="definition"):
        run_queue(changed,root,tmp_path/"local",require_cuda=False)


def test_corrupt_upload_does_not_publish_complete(tmp_path,monkeypatch):
    import shutil
    src=tmp_path/"run/resume/step000012";src.mkdir(parents=True)
    for name in ("state.pt","replay.npz","history.csv.gz"):(src/name).write_bytes(b"payload")
    json_write(src/"COMPLETE.json",{"step":12,"sha256":{n:hash_file(src/n) for n in ("state.pt","replay.npz","history.csv.gz")}})
    original=shutil.copy2
    def corrupt(a,b,*args,**kw):
        original(a,b,*args,**kw)
        if str(b).endswith("history.csv.gz.partial"):
            with open(b,"ab") as f:f.write(b"corrupt")
    monkeypatch.setattr(shutil,"copy2",corrupt)
    with pytest.raises(IOError,match="checksum"):
        publish(tmp_path/"run",tmp_path/"drive")
    assert not (tmp_path/"drive/run/resume/step000012/COMPLETE.json").exists()


def test_fatal_errors_do_not_retry_from_zero():
    assert classify_failure("Environment reconstruction diverged; continuation rejected")=="fatal"
    assert classify_failure("CUDA requested but unavailable")=="fatal"
    assert classify_failure("No space left on device")=="fatal"
    assert classify_failure("process killed by signal9")=="recoverable"


def test_retention_does_not_race_active_learner(tmp_path):
    from Experiments.full_queue import prune_verified_generations
    job=tmp_path/"job"
    generations=[]
    for step in (12,24,30):
        folder=job/"attempts/active/resume"/f"step{step:06d}"
        folder.mkdir(parents=True)
        generations.append((step,folder))
    prune_verified_generations(job,generations,active_attempt="active")
    assert all(p.exists() for _,p in generations)
    prune_verified_generations(job,generations)
    assert not generations[0][1].exists() and generations[1][1].exists()
