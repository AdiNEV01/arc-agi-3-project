"""Verify actual 75K evidence and forced subprocess recovery before full runs."""
import argparse
import ast
from collections import Counter
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import torch
from DQN.extended_train import check_resume, hash_file, json_write, source_hashes, publish
from DQN.q_network import QNetwork


def without_publish(text):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!="publish"]
    return ast.dump(tree,include_attributes=False)


def verify_pilot_evidence():
    cfg = json.loads(Path("Configs/dqn_extended_v1.json").read_text())
    proof = json.loads(Path("Evidence/pilot_provenance.json").read_text())
    archive = Path("Evidence/verified_75k_evidence.zip")
    if hash_file(archive) != proof["evidence_sha256"]:
        raise ValueError("Pilot evidence ZIP changed")
    if without_publish(Path("DQN/extended_train.py").read_text()) != without_publish(Path("Evidence/pilot_extended_train.py.txt").read_text()):
        raise ValueError("Learner differs from the validated pilot implementation")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Pilot evidence CRC failure")
        source = json.loads(z.read("EXTENDED_SOURCE_MANIFEST.json"))["sha256"]
        for path, expected in source.items():
            if path.startswith(("DQN/","arc_env/")) and path != "DQN/extended_train.py" and hash_file(Path(path)) != expected:
                raise ValueError(f"Scientific implementation changed: {path}")
        variants = set()
        for name in z.namelist():
            if not name.endswith("/pilot_validation.json"):
                continue
            root = name.rsplit("/",1)[0]
            receipt = json.loads(z.read(name))
            if receipt["protocol"] != cfg or receipt["seed"] !=42 or not receipt["valid"]:
                raise ValueError("Pilot protocol/seed/validation differs")
            rows = list(csv.DictReader(io.StringIO(z.read(root+"/actions.csv").decode())))
            if len(rows)!=75000 or any(int(r["global_step"])!=i for i,r in enumerate(rows,1)):
                raise ValueError("Pilot action history incomplete")
            if Counter(r["game"] for r in rows) != Counter({g:25000 for g in cfg["training_games"]}):
                raise ValueError("Pilot exposure differs")
            losses = [float(r["loss"]) for r in rows if r["loss"]]
            if len(losses)!=14801 or not all(math.isfinite(v) for v in losses):
                raise ValueError("Pilot update/loss validation failed")
            raw = z.read(root+"/frozen/step075000.pt")
            import hashlib
            if hashlib.sha256(raw).hexdigest()!=receipt["frozen_sha256"]:
                raise ValueError("Pilot frozen checkpoint hash changed")
            saved = torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
            net = QNetwork(saved["config"]["num_frames"])
            net.load_state_dict(saved["online_state_dict"],strict=True)
            if saved["gradient_steps"]!=14801 or not all(torch.isfinite(v).all().item() for v in saved["online_state_dict"].values()):
                raise ValueError("Pilot checkpoint invalid")
            variants.add(receipt["variant"])
        if variants != {"v0","v2"}:
            raise ValueError("Both genuine V0/V2 pilots are required")
        ev = json.loads(z.read("development_evaluation/evaluation_validation.json"))
        if not ev["valid"] or ev["reserved_heldout_evaluated"]:
            raise ValueError("Pilot evaluation validation failed")
    return {"valid":True,"protocol":cfg["protocol_id"],"pilots":["v0_seed42","v2_seed42"],
        "actions_each":75000,"updates_each":14801,
        "compatibility":"Only publish() durability changed; all other learner/agent/environment AST and bytes match pilots"}


def verify_action_policy(out, device="cuda"):
    """Compare frozen bridge with the actual training encoding and masked Q argmax."""
    import arc_agi
    import numpy as np
    from types import SimpleNamespace
    from arc_env.gym_adapter import ArcGymEnv
    from arc_env.encoding import ActionMap
    from DQN.action_codec import decode_action,encode_action,valid_action_mask
    from DQN.dqn_agent import DQNAgent
    from DQN.dqn_improvement_2 import FrameHistory
    from DQN.q_network import encode_frames
    from agents.dqn_agent import DQNPolicy
    torch.set_num_threads(1)
    amap = ActionMap(64)
    for i,spec in enumerate(amap.specs):
        act,data=decode_action(i)
        if encode_action(act,data)!=i or spec.action_id!=act.value or spec.data!=(data or {}):
            raise ValueError("Action coordinate map differs")
    results=[]
    with zipfile.ZipFile("Evidence/verified_75k_evidence.zip") as z:
        arcade=arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
        cfg=json.loads(Path("Configs/dqn_extended_v1.json").read_text())
        for variant in ("v0","v2"):
            model_path=out/(variant+"_75k.pt")
            raw=z.read(next(n for n in z.namelist() if n.startswith(variant+"_seed42_") and n.endswith("/frozen/step075000.pt")))
            model_path.write_bytes(raw)
            saved=torch.load(model_path,map_location="cpu",weights_only=True)
            agent=DQNAgent(num_frames=saved["config"]["num_frames"],device=device)
            agent.online.load_state_dict(saved["online_state_dict"])
            for game in cfg["development_games"]:
                env=ArcGymEnv(game,arcade=arcade,game_seed=0,click_grid=64,reset_mode="level")
                obs,info=env.reset()
                policy=DQNPolicy(model_path,click_grid=64,seed=0,device=device)
                history=FrameHistory(saved["config"]["num_frames"])
                history.reset(SimpleNamespace(frame=[obs]))
                expected_traj=[json.loads(s) for s in z.read(f"development_evaluation/{variant}_s42_step75000/traj/{variant}_s42_step75000_s0_{game}.jsonl").decode().splitlines()]
                actions=[]
                for step in range(20):
                    index=policy.act(obs,info)
                    expected=agent.select_action(history.state(),[__import__('arcengine').GameAction.from_id(a) for a in info["available_actions"]],0)
                    if index!=expected or not info["action_mask"][index]:
                        raise ValueError("Frozen/training policy or mask differs")
                    encoded=encode_frames(history.state()).numpy()[0]
                    from agents.checkpoint_encoding import GridHistory
                    reference=GridHistory(saved["config"]["num_frames"])
                    for frame in history.frames:
                        last=reference.push(frame)
                    np.testing.assert_array_equal(encoded,last)
                    label=amap.decode(index).label()
                    if label != expected_traj[step]["action"]:
                        raise ValueError("Pilot click behavior cannot be reproduced from its checkpoint")
                    nxt,reward,term,trunc,ninfo=env.step(index)
                    policy.observe(obs,index,reward,nxt,term,trunc,ninfo)
                    history.advance(SimpleNamespace(frame=[nxt]))
                    obs,info=nxt,ninfo
                    actions.append(label)
                    if term:
                        obs,info=env.reset();policy.reset_episode();history.reset(SimpleNamespace(frame=[obs]))
                results.append({"variant":variant,"game":game,"actions":20,"action_counts":dict(Counter(actions)),"valid":True})
    return {"valid":True,"all_4102_action_indices_verified":True,"policy_checks":results,
        "conclusion":"Repeated clicks reproduce trained masked Q argmax; no decoding/masking/preprocessing/checkpoint mismatch found"}


def forced_recovery_test(out, persistent, device="cuda"):
    """Kill a live training subprocess after a safe checkpoint, then restore it."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    def command(run,stop,resume=None,persist=None):
        cmd=[sys.executable,"-u","-m","DQN.extended_train","--variant","v2","--seed","42","--device",device,
             "--run-dir",str(run),"--stop-at",str(stop)]
        if resume:cmd += ["--resume",str(resume)]
        if persist:cmd += ["--persistent-root",str(persist)]
        return cmd
    def checked(cmd,name):
        with (out/(name+".log")).open("x") as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    checked(command(out/"reference",1200),"reference")
    checked(command(out/"prefix",1100,persist=persistent),"prefix")
    checkpoint=Path(persistent)/"prefix/resume/step001100"
    check_resume(checkpoint)
    with (out/"interrupted.log").open("x") as f:
        process=subprocess.Popen(command(out/"interrupted",450000,resume=checkpoint),stdout=f,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+180
        from Experiments.full_queue import latest_progress
        while time.monotonic()<deadline:
            p=latest_progress(out/"interrupted")
            if p and p[0]>1120:
                process.kill();process.wait(timeout=20);break
            if process.poll() is not None:
                raise RuntimeError("Interruption probe exited before the controlled kill")
            time.sleep(0.05)
        else:
            process.kill();process.wait();raise RuntimeError("Interruption probe timed out")
    checked(command(out/"restored",1200,resume=checkpoint),"restored")
    a,b=[check_resume(out/x/"resume/step001200") for x in ("reference","restored")]
    if a["journal"]!=b["journal"]:raise ValueError("Restored environment transcript differs")
    for kind in ("online_state_dict","target_state_dict"):
        for key,tensor in a[kind].items():torch.testing.assert_close(tensor,b[kind][key],rtol=0,atol=0)
    return {"valid":True,"device":device,"forced_process_kill":True,"restored_from":1100,
        "verified_at":1200,"transcript_and_parameters_bitwise_equal":True,"drive_checkpoint":str(checkpoint)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--persistent",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    proof=verify_pilot_evidence()
    print("Verified actual75K pilot evidence",flush=True)
    policy=verify_action_policy(args.out,"cuda")
    print("Verified decoding, masks, training/frozen observation and policy behavior",flush=True)
    recovery=forced_recovery_test(args.out/"recovery",args.persistent/args.out.name,"cuda")
    report={"valid":True,"source_sha256":source_hashes(),"pilot_evidence":proof,"policy":policy,"recovery":recovery,
        "torch_version":str(torch.__version__),"python_version":sys.version,
        "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip() if Path("/proc/sys/kernel/random/boot_id").exists() else "local-test",
        "cuda_device":torch.cuda.get_device_name(0),"time":time.time()}
    json_write(args.out/"PREFLIGHT_PASSED.json",report)
    publish(args.out,args.persistent)
    print("PREFLIGHT PASSED: safe to launch full queue",flush=True)


if __name__=="__main__":main()
