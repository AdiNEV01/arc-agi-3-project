"""Separate, explicit Task3 development workflow after the full queue completes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from Experiments.full_queue import atomic_copy


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--queue-root",required=True,type=Path)
    ap.add_argument("--staging",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    cfg=json.loads((a.queue_root/"QUEUE.json").read_text())["protocol"]
    for variant in cfg["variants"]:
        for seed in cfg["seeds"]:
            job=a.queue_root/variant/f"seed{seed}"
            record=json.loads((job/"RUN_COMPLETE.json").read_text())
            if not record["validated"] or record["actions"]!=450000:
                raise RuntimeError("All twelve final runs must verify before full development curves")
            dest=a.staging/f"{variant}_seed{seed}"
            atomic_copy(job/"metadata.json",dest/"metadata.json")
            for step,model in record["snapshots"].items():
                atomic_copy(Path(model["path"]),dest/"frozen"/f"step{int(step):06d}.pt",model["sha256"])
    subprocess.run([sys.executable,"-m","Experiments.evaluate_dqn_extended",str(a.staging),"--out",str(a.out),"--device","cuda"],check=True)


if __name__=="__main__":main()
