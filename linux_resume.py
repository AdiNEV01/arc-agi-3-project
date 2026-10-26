"""Resume the exact Colab experiment on a local Linux NVIDIA GPU.

No original learner/controller files are modified. The local platform supplies
its own CUDA and disk checks in place of Colab's T4/Drive-quota checks; children
still explicitly train on CUDA and retain all source/protocol/checksum gates.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
BASE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()


def verify_source(base=BASE):
    m=json.loads((base/"FULL_SOURCE_MANIFEST.json").read_text())
    for name,expected in m["sha256"].items():
        if sha(base/name)!=expected:raise ValueError(f"Training source changed: {name}")
    adapter=json.loads((base/"LINUX_ADAPTER_MANIFEST.json").read_text())
    for name,expected in adapter["sha256"].items():
        if sha(base/name)!=expected:raise ValueError(f"Linux launcher changed: {name}")
    return m


def locate_queue(path,cache):
    path=Path(path).expanduser().resolve()
    if path.is_file():
        if not zipfile.is_zipfile(path):raise ValueError("Checkpoint download must be a ZIP or extracted folder")
        dest=Path(cache)/sha(path)[:12]
        marker=dest/"IMPORT_COMPLETE.json"
        if not marker.exists():
            dest.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(path) as z:
                if z.testzip() is not None:raise ValueError("The Drive download ZIP is corrupt")
                needed=sum(p.file_size for p in z.infolist())+2*2**30
                if shutil.disk_usage(dest).free < needed:
                    raise OSError("Not enough local disk to extract checkpoints and keep a 2GiB working reserve")
                for name in z.namelist():
                    if not (dest/name).resolve().is_relative_to(dest.resolve()):raise ValueError("Unsafe checkpoint ZIP path")
                z.extractall(dest)
            marker.write_text(json.dumps({"archive":str(path),"sha256":sha(path)}))
        path=dest
    choices=[path] if (path/"QUEUE.json").exists() else [p.parent for p in path.rglob("QUEUE.json")]
    if len(choices)!=1:
        raise ValueError(f"Expected one complete experiment folder with QUEUE.json; found {len(choices)}. Download the whole FULL_450K folder, not just model .pt files.")
    return choices[0].resolve()


def find_download(base=BASE):
    saved=base/"LINUX_CHECKPOINT_LOCATION.json"
    if saved.exists():
        path=Path(json.loads(saved.read_text())["root"])
        if (path/"QUEUE.json").exists():return path
    local=base/"checkpoints"
    if local.exists():return local
    downloads=Path.home()/"Downloads"
    paths=list(downloads.glob("FULL_450K_dqn_extended_v1_20261024*.zip"))
    paths += [p for p in downloads.glob("FULL_450K_dqn_extended_v1_20261024*") if p.is_dir()]
    if not paths:
        raise ValueError("Download MyDrive/ARC_DQN_extended/FULL_450K_dqn_extended_v1_20261024 as a complete ZIP into Downloads, then rerun this command. Or use --checkpoints /path/to/folder-or-zip.")
    return max(paths,key=lambda p:p.stat().st_mtime)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoints",type=Path)
    ap.add_argument("--inspect-only",action="store_true")
    ap.add_argument("--cuda-build",choices=("cu130","cu126"),default="cu130")
    a=ap.parse_args()
    os.chdir(BASE)
    if a.cuda_build=="cu126":
        cmd=["bash",str(BASE/"start_linux_cu126.sh")]
        if a.checkpoints:cmd += ["--checkpoints",str(a.checkpoints.resolve())]
        if a.inspect_only:cmd += ["--verify-only"]
        os.execvp("bash",cmd)
    if platform.system()!="Linux":raise RuntimeError("Use this launcher on the Linux NVIDIA PC")
    source=verify_source()
    import torch
    import numpy as np
    from importlib import metadata
    from DQN.extended_train import check_resume,source_hashes,json_write
    from Experiments.full_queue import verified_generations,run_queue
    if not torch.cuda.is_available():raise RuntimeError("NVIDIA CUDA is unavailable. No CPU training will start.")
    gpu=torch.cuda.get_device_name(0)
    software={"python":platform.python_version(),"numpy":np.__version__,"torch":str(torch.__version__),
        **{k:metadata.version(k) for k in ("arc-agi","arcengine","gymnasium")}}
    print("GPU:",gpu,"CUDA:",torch.version.cuda,"Software:",software,flush=True)
    nvidia=subprocess.check_output(["nvidia-smi"],text=True)
    gpu_details=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,uuid","--format=csv,noheader"],text=True).strip()
    print(nvidia,flush=True)
    root=locate_queue(a.checkpoints or find_download(),BASE/"checkpoint_imports")
    cfg=source["protocol"]
    definition=json.loads((root/"QUEUE.json").read_text())
    if definition["protocol"]!=cfg or definition["source_sha256"]!=source_hashes():
        raise ValueError("Downloaded queue does not match the exact training code/protocol")
    for name,expected in definition.get("queue_sources",{}).items():
        if sha(BASE/name)!=expected:raise ValueError(f"Original queue source differs: {name}")
    progress_path=root/"progress.json"
    if not progress_path.exists():raise ValueError("progress.json is missing. Download the entire Drive experiment folder.")
    previous=json.loads(progress_path.read_text())["jobs"]
    status=[]
    total_saved=0
    for seed in cfg["seeds"]:
        for variant in cfg["variants"]:
            job=root/variant/f"seed{seed}"
            generations,invalid=verified_generations(job,cfg,variant,seed) if job.exists() else ([],[])
            step=generations[-1][0] if generations else 0
            recorded=previous.get(f"{variant}_seed{seed}",{}).get("completed_actions",0)
            if step < recorded and not invalid:
                raise ValueError(f"Missing downloaded checkpoints for {variant} seed{seed}: Drive recorded {recorded}, found {step}. If Drive split the download, extract ALL parts into one folder and pass --checkpoints /path/to/that/folder.")
            if generations:
                state=check_resume(generations[-1][1])
                if state["software"]!=software:
                    raise ValueError(f"Exact continuation needs {state['software']}; current venv is {software}. Run bash start_linux.sh to prepare the pinned environment.")
                if state["device_type"]!="cuda":raise ValueError("Expected an original CUDA continuation checkpoint")
                del state
            total_saved+=step
            status.append({"variant":variant,"seed":seed,"saved_actions":step,
                "latest":str(generations[-1][1]) if generations else None,"invalid":invalid})
            print(f"VERIFIED {variant} seed{seed}: {step}/{cfg['total_actions']}",flush=True)
    if total_saved==0:raise ValueError("No complete recovery checkpoint found. Refusing to restart the experiment from zero.")
    if shutil.disk_usage(root).free < 2*2**30:raise OSError("At least 2GiB free local disk is required for checkpoint copying")
    json_write(BASE/"LINUX_CHECKPOINT_LOCATION.json",{"root":str(root)})
    migration={"time_utc":datetime.now(timezone.utc).isoformat(),"previous_hardware":"Colab T4",
        "current_gpu":gpu,"cuda_version":torch.version.cuda,"nvidia_smi":nvidia,"gpu_driver_identity":gpu_details,
        "operating_system":platform.platform(),"software":software,"queue_root":str(root),
        "saved_progress":status,"adapter_sha256":sha(Path(__file__)),
        "platform":"Linux CUDA + local persistent disk; original queue and learner bytes unchanged",
        "reproducibility":"New GPU model/driver recorded; cross-hardware bitwise equivalence is not claimed",
        "cloud_sync":"Not configured by this adapter; local checkpoints must be backed up to Drive separately"}
    records=root/"linux_migrations";records.mkdir(exist_ok=True)
    json_write(records/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:6]+".json"),migration)
    print("Local persistent results:",root,flush=True)
    if a.inspect_only:
        print("INSPECTION PASSED; no training launched.",flush=True);return
    work=BASE/"linux_work";work.mkdir(exist_ok=True)
    receipt=work/"GPU_RECOVERY_PASSED.json"
    hardware={"gpu":gpu,"gpu_driver_identity":gpu_details,"cuda_version":torch.version.cuda,
        "software":software,"source_sha256":source_hashes(),"adapter_sha256":sha(Path(__file__))}
    if not receipt.exists() or json.loads(receipt.read_text()).get("hardware")!=hardware:
        from Experiments.full_preflight import forced_recovery_test
        probe=work/("preflight_"+uuid.uuid4().hex[:8])
        print("Checking same-GPU forced interruption and restoration before resuming...",flush=True)
        result=forced_recovery_test(probe,root/"linux_preflight"/probe.name,"cuda")
        json_write(receipt,{"valid":True,"hardware":hardware,"result":result})
    def command(variant,seed,run,resume,persistent):
        # Hardware and local disk have been checked here. Child always uses CUDA.
        if not torch.cuda.is_available():raise RuntimeError("GPU disappeared; stopping with saved progress")
        if shutil.disk_usage(root).free < 512*2**20:raise OSError("Insufficient local checkpoint disk reserve")
        cmd=[sys.executable,"-u","-m","DQN.extended_train","--variant",variant,"--seed",str(seed),
             "--device","cuda","--run-dir",str(run),"--stop-at",str(cfg["total_actions"]),
             "--persistent-root",str(persistent)]
        if resume:cmd += ["--resume",str(resume)]
        return cmd
    print("RESUMING QUEUE: completed jobs are skipped; latest valid states are restored.",flush=True)
    print("Progress is saved to LOCAL DISK. Sync/upload this experiment folder back to Drive for an off-PC backup.",flush=True)
    run_queue(cfg,root,work/"training",command_factory=command,require_cuda=False)


if __name__=="__main__":
    try:main()
    except KeyboardInterrupt:raise
    except Exception as exc:
        print("STOPPED SAFELY:",exc,file=sys.stderr,flush=True)
        raise SystemExit(1)
