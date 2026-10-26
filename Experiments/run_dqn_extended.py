"""Default: two CUDA pilots only. Full matrix requires separately validated evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from DQN.extended_train import json_write, publish, source_hashes, runtime_hashes
from Experiments.validate_dqn_pilot import validate


def matrix_gate(pilot_root, config):
    runs = [p.parent for p in Path(pilot_root).rglob("summary.json")]
    good = {}
    for root in runs:
        r = json.loads((root / "summary.json").read_text())
        if r["variant"] not in ("v0", "v2") or r["seed"] != 42 or r["global_step"] != 75000:
            continue
        evidence = validate(root)
        if evidence["protocol"] != config or evidence["source_sha256"] != source_hashes() or evidence["runtime_assets"] != runtime_hashes(config["training_games"]):
            raise ValueError("Pilot source/protocol/assets do not match the proposed matrix")
        good[r["variant"]] = root
    if set(good) != {"v0", "v2"}:
        raise ValueError("Both 75K V0 and V2 seed42 CUDA pilots must validate before matrix launch")
    ev = Path(pilot_root) / "development_evaluation/evaluation_validation.json"
    if not ev.exists() or not json.loads(ev.read_text()).get("valid"):
        raise ValueError("Shared development evaluation must complete without invariant failures")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilots", "matrix"], default="pilots")
    ap.add_argument("--config", default="Configs/dqn_extended_v1.json")
    ap.add_argument("--output-root", type=Path, default=Path("outputs/extended"))
    ap.add_argument("--persistent-root", type=Path, required=True)
    ap.add_argument("--validated-pilots", type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    c = json.loads(Path(args.config).read_text())
    if args.phase == "matrix":
        if not args.validated_pilots:
            ap.error("matrix requires --validated-pilots; pilots are the default")
        matrix_gate(args.validated_pilots, c)
    jobs = [(v,42) for v in ("v0","v2")] if args.phase == "pilots" else [(v,s) for s in c["seeds"] for v in c["variants"]]
    runid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    batch = args.output_root / (args.phase + "_" + runid)
    batch.mkdir(parents=True, exist_ok=False)
    json_write(batch / "batch_plan.json", {"phase":args.phase, "jobs":jobs,"protocol":c})
    if args.phase == "pilots":
        # No GPU hours until the real same-device environment restore preflight passes.
        subprocess.run([sys.executable,"-m","Experiments.check_extended_reconstruction",
                        "--device",args.device,"--out",str(batch / "reconstruction")],check=True)
    results = []
    for variant, seed in jobs:
        run = batch / f"{variant}_seed{seed}_{runid}"
        cmd = [sys.executable,"-m","DQN.extended_train","--config",args.config,"--variant",variant,
               "--seed",str(seed),"--device",args.device,"--run-dir",str(run),
               "--stop-at",str(75000 if args.phase == "pilots" else c["total_actions"]),
               "--persistent-root",str(args.persistent_root / batch.name)]
        env = {**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONHASHSEED":str(seed),"CUBLAS_WORKSPACE_CONFIG":":4096:8"}
        print("Starting",variant,seed,flush=True)
        with (batch / f"{variant}_seed{seed}_console.txt").open("x") as log:
            process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        if returncode:
            print((batch / f"{variant}_seed{seed}_console.txt").read_text()[-6000:])
            raise RuntimeError("Training failed; matrix progression stopped")
        if args.phase == "pilots":
            validate(run)
        results.append(json.loads((run / "summary.json").read_text()))
        json_write(batch / "batch_summary.json",results)
        publish(batch,args.persistent_root)
    if args.phase == "pilots":
        rates = {r["variant"]:r["actions_per_second_including_persistence"] for r in results}
        # V1 approximated by V0 and history-only by V2 until separately timed.
        base = 3 * 450000 * (2/rates["v0"] + 2/rates["v2"])
        json_write(batch / "matrix_cost_estimate.json", {"estimated_training_hours":base/3600,
            "with_25_percent_reserve_hours":base*1.25/3600,
            "assumption":"V1~V0, history-only~V2; multi-game 75K rate extrapolated, not measured 450K",
            "matrix_launched":False})
        publish(batch,args.persistent_root)
    print("Completed:",batch,flush=True)


if __name__ == "__main__":
    main()
