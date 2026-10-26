"""Common final-only DQN/PPO/random development evaluation, with learning guards.

Default: smoke first, then all24 final models (216 game/seed runs) plus9 random.
No reserved games, checkpoint selection by score, optimizer steps or training.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import uuid
import zipfile

GAMES=("ft09-0d8bbf25","vc33-5430563c","cd82-fb555c5d")
EXPECTED={"DQN":({"v0","v1","v2_history_only","v2"},{42,123,456}),
          "PPO":({"baseline","curiosity","memory_only","memory_curiosity"},{0,1,2})}
BASE=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:json.dump(value,f,indent=2);f.write("\n")


def state_digest(value):
    import torch
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            x=x.detach().cpu().contiguous()
            h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b"dict")
            for key in sorted(x,key=repr):add(key);add(x[key])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for y in x:add(y)
        else:h.update((type(x).__name__+":"+repr(x)).encode())
    add(value)
    return h.hexdigest()


def forbid_learning(*args,**kwargs):
    raise RuntimeError("Learning/optimizer steps are forbidden during frozen evaluation")


class FrozenGuard:
    def __init__(self,policy,algorithm):
        self.policy=policy
        self.net=policy.model if algorithm=="DQN" else policy.model.policy
        self.net.eval()
        for p in self.net.parameters():p.requires_grad_(False)
        self.weights_before=state_digest(self.net.state_dict())
        self.optimizer=getattr(self.net,"optimizer",None)
        self.optimizer_before=state_digest(self.optimizer.state_dict()) if self.optimizer else None
        if self.optimizer:self.optimizer.step=forbid_learning
        if hasattr(policy.model,"learn"):policy.model.learn=forbid_learning

    def check(self):
        if state_digest(self.net.state_dict())!=self.weights_before:
            raise ValueError("In-memory policy weights changed during evaluation")
        if self.optimizer and state_digest(self.optimizer.state_dict())!=self.optimizer_before:
            raise ValueError("Optimizer state changed during evaluation")
        return dict(weights_unchanged=True,optimizer_state_unchanged=True,
                    gradient_updates_allowed=False,weights_sha256=self.weights_before)


def validate_protocol(path):
    import yaml
    c=yaml.safe_load(path.read_text(encoding="utf-8"));e=c["eval"]
    if (set(c["games"])!={"dev"} or tuple(c["games"]["dev"])!=GAMES or e["seeds"]!=[0,1,2] or
        e["game_seed"]!="same_as_run" or e["max_actions_per_game"]!=1000 or e["max_seconds_per_game"] is not None or
        e["click_grid"]!=64 or e["operation_mode"]!="offline"):
        raise ValueError("The fixed shared development protocol changed")
    return c


def validate_models(rows,root):
    for algorithm,(variants,seeds) in EXPECTED.items():
        selected=[r for r in rows if r["algorithm"]==algorithm]
        if len(selected)!=12 or {(r["variant"],r["training_seed"]) for r in selected}!={(v,s) for v in variants for s in seeds}:
            raise ValueError("Require all twelve final models for "+algorithm)
        if len({r["sha256"] for r in selected})!=12:raise ValueError("Duplicate model weight files")
        for r in selected:
            path=(root/r["path"]).resolve()
            if r["checkpoint_step"]!=503808 or not path.is_relative_to(root.resolve()):
                raise ValueError("Only registered final503808 checkpoints are allowed")
            if sha(path)!=r["sha256"]:raise ValueError("Model checksum differs: "+r["id"])
    if len(rows)!=24:raise ValueError("Unexpected additional models")


def validate_records(rows,seeds,budget):
    if len(rows)!=len(GAMES)*len(seeds) or {(r["game_id"],r["seed"]) for r in rows}!={(g,s) for g in GAMES for s in seeds}:
        raise ValueError("Incomplete or duplicate game/seed matrix")
    for r in rows:
        if (r["game_seed"]!=r["seed"] or r["action_budget"]!=budget or r["scorecard_runs"]!=1 or
            r["full_resets_after_start"]!=0 or r["illegal_actions"]!=0 or
            r["score_official"] is None or r["score_recomputed"] is None or
            not math.isfinite(r["score_official"]) or not 0<=r["score_official"]<=100 or
            abs(r["score_official"]-r["score_recomputed"])>=1e-6 or
            not 0<r["official_actions"]<=budget):
            raise ValueError("Shared scoring/action/reset contract failed")
        if not r["success"] and (r["official_actions"]!=budget or r["stop_reason"]!="budget"):
            raise ValueError("Unsuccessful run stopped before its selected-action cap")
        if r["success"] and r["stop_reason"]!="win":raise ValueError("Successful run has an inconsistent stopping reason")


def aggregate(records):
    groups=defaultdict(list)
    for r in records:groups[(r["algorithm"],r["variant"],r["training_seed"])].append(r)
    seed_rows=[]
    for (algorithm,variant,seed),rs in sorted(groups.items()):
        row=dict(algorithm=algorithm,variant=variant,training_seed=seed)
        for key in ("score_official","levels_completed","success","runtime_s","action_entropy_nats"):
            per_game=defaultdict(list)
            for r in rs:per_game[r["game_id"]].append(float(r[key]))
            row[key]=statistics.fmean(statistics.fmean(v) for v in per_game.values())
        seed_rows.append(row)
    variants=defaultdict(list)
    for r in seed_rows:variants[(r["algorithm"],r["variant"])].append(r)
    combined=[]
    for (algorithm,variant),rs in sorted(variants.items()):
        row=dict(algorithm=algorithm,variant=variant,independent_training_seeds=len(rs))
        for key in ("score_official","levels_completed","success","runtime_s","action_entropy_nats"):
            values=[r[key] for r in rs]
            row[key]=dict(mean=statistics.fmean(values),sd=statistics.stdev(values) if len(values)>1 else None,
                          min=min(values),max=max(values),individual_training_seed_means=values)
        combined.append(row)
    return dict(seed_level_means=seed_rows,across_training_seed_summary=combined,
        interpretation="Training n=3 per variant; evaluation seeds are repeated measurements. Report individual points and SD/range. All-zero scores do not establish narrow uncertainty.")


def evaluate_one(row,out,device,config,seeds,budget):
    import evaluate
    guards=[]
    original_loader=evaluate.load_policy
    random=row["algorithm"]=="random"
    spec="random" if random else "agents.dqn_agent:DQNPolicy" if row["algorithm"]=="DQN" else "agents.ppo_agent:PPOPolicy"
    def loader(spec,kwargs,seed,click_grid):
        p=original_loader(spec,kwargs,seed,click_grid)
        if not random:guards.append(FrozenGuard(p,row["algorithm"]))
        return p
    before=None if random else sha(BASE/row["path"])
    args=["--config",str(config),"--agent",spec,"--agent-version",row["id"],"--split","dev",
          "--seeds",*[str(s) for s in seeds],"--budget",str(budget),"--out",str(out),"--save-traj"]
    if not random:args += ["--checkpoint-step","503808","--agent-kwargs",json.dumps(dict(checkpoint=str(BASE/row["path"]),device=device))]
    evaluate.load_policy=loader
    try:evaluate.main(args)
    finally:evaluate.load_policy=original_loader
    rows=[json.loads(s) for s in (out/"episodes.jsonl").read_text().splitlines()]
    validate_records(rows,seeds,budget)
    checks=[g.check() for g in guards]
    if not random and sha(BASE/row["path"])!=before:raise ValueError("Checkpoint file changed")
    for r in rows:
        trajectory=[json.loads(x) for x in Path(r["trajectory_file"]).read_text().splitlines()]
        counts=Counter(x["action"] for x in trajectory)
        if len(trajectory)!=r["official_actions"]:raise ValueError("Trajectory action count differs")
        r.update(algorithm=row["algorithm"],variant=row["variant"],training_seed=row.get("training_seed"),
            checkpoint_step=None if random else 503808,model_sha256=before,action_counts=dict(counts),
            action_entropy_nats=-sum((n/len(trajectory))*math.log(n/len(trajectory)) for n in counts.values()))
    write_json(out/"frozen_policy_validation.json",dict(valid=True,model=row,guard_checks=checks,
               weight_file_unchanged=True,learning_performed=False))
    del guards
    gc.collect()
    import torch
    if device=="cuda":torch.cuda.synchronize();torch.cuda.empty_cache()
    return rows


def run_matrix(models,out,device,config,smoke):
    out.mkdir(parents=True,exist_ok=False)
    selected=[r for r in models if (r["algorithm"],r["variant"],r["training_seed"]) in
        {("DQN","v0",42),("DQN","v2",42),("PPO","baseline",0),("PPO","memory_curiosity",0)}] if smoke else models
    seeds,budget=([0],10) if smoke else ([0,1,2],1000)
    learned=[]
    for i,row in enumerate(selected,1):
        print(f"{'SMOKE' if smoke else 'FULL'} {i}/{len(selected)}: {row['id']}",flush=True)
        learned.extend(evaluate_one(row,out/row["id"],device,config,seeds,budget))
    random=evaluate_one(dict(id="random_type_uniform",algorithm="random",variant="type_uniform"),out/"random",device,config,seeds,budget)
    if len(learned)!=(12 if smoke else 216) or len(random)!=(3 if smoke else 9):raise ValueError("Incomplete matrix")
    write_json(out/"records.json",learned)
    write_json(out/"random_records.json",random)
    if not smoke:
        summary=aggregate(learned)
        write_json(out/"training_seed_aggregate.json",summary)
        flat=[]
        for r in summary["across_training_seed_summary"]:
            row={k:r[k] for k in ("algorithm","variant","independent_training_seeds")}
            for key in ("score_official","levels_completed","success","runtime_s","action_entropy_nats"):
                row[key+"_mean"]=r[key]["mean"];row[key+"_sd"]=r[key]["sd"]
            flat.append(row)
        with (out/"variant_summary.csv").open("x",newline="") as f:
            writer=csv.DictWriter(f,list(flat[0]));writer.writeheader();writer.writerows(flat)
    validation=dict(valid=True,formal_evaluation=not smoke,learned_policy_game_runs=len(learned),
        random_game_runs=len(random),selected_action_cap=budget,evaluation_seeds=seeds,
        model_count=len(selected),checkpoint_step=503808,reserved_heldout_evaluated=False,learning_performed=False)
    write_json(out/"evaluation_validation.json",validation)
    return validation


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry",type=Path,default=Path("Configs/final_shared_checkpoints.json"))
    ap.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    ap.add_argument("--out",type=Path)
    ap.add_argument("--check-only",action="store_true")
    ap.add_argument("--smoke-only",action="store_true")
    a=ap.parse_args(argv)
    os.chdir(BASE)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if os.environ.get("ONLY_RESET_LEVELS")=="true":raise ValueError("Unset ONLY_RESET_LEVELS")
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    versions={n:metadata.version(n) for n in ("torch","numpy","arc-agi","arcengine","gymnasium","stable-baselines3","sb3-contrib","cloudpickle")}
    for k,v in {"numpy":"2.5.4","arc-agi":"0.9.9","arcengine":"0.9.3","gymnasium":"1.4.0","stable-baselines3":"2.9.0","sb3-contrib":"2.9.0"}.items():
        if versions[k]!=v:raise ValueError(f"Expected{k}=={v}; found{versions[k]}")
    if a.device=="cuda" and (str(torch.__version__)!="2.11.0+cu126" or torch.version.cuda!="12.6" or not torch.cuda.is_available()):
        raise RuntimeError("Use the verified PyTorch2.11.0+cu126 environment; CUDA must pass. No CPU fallback.")
    config=BASE/"Configs/final_shared_development.yaml";validate_protocol(config)
    registry=json.loads((BASE/a.registry).read_text());models=registry["models"];validate_models(models,BASE)
    from evaluate import load_policy
    preflight=[]
    for row in models:
        policy=load_policy("agents.dqn_agent:DQNPolicy" if row["algorithm"]=="DQN" else "agents.ppo_agent:PPOPolicy",
                           dict(checkpoint=str(BASE/row["path"]),device=a.device),0,64)
        if row["algorithm"]=="DQN":
            saved=torch.load(BASE/row["path"],map_location="cpu",weights_only=True)
            if saved["global_step"]!=503808 or saved["seed"]!=row["training_seed"]:raise ValueError("DQN checkpoint metadata differs")
            del saved
        elif policy.model.num_timesteps!=503808 or policy.model.seed!=row["training_seed"]:
            raise ValueError("PPO checkpoint metadata differs")
        guard=FrozenGuard(policy,row["algorithm"])
        import numpy as np
        action=policy.act(np.zeros((64,64),dtype=np.uint8),dict(action_mask=np.ones(4102,dtype=bool)))
        if not 0<=action<4102:raise ValueError("Policy action preflight failed")
        preflight.append(dict(id=row["id"],checks=guard.check()))
        del guard,policy
        gc.collect()
    runtime=dict(versions=versions,python=platform.python_version(),platform=platform.platform(),device=a.device,
                 gpu=torch.cuda.get_device_name(0) if a.device=="cuda" else None,deterministic_algorithms=True,
                 cuda_build=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
                 source_training_budgets_equal=503808,protocol_sha256=sha(config),registry_sha256=sha(BASE/a.registry),
                 source_sha256={name:sha(BASE/name) for name in ("evaluate.py","agents/dqn_agent.py","agents/ppo_agent.py",
                    "agents/checkpoint_encoding.py","arc_env/gym_adapter.py","arc_env/encoding.py","Experiments/evaluate_final_shared.py")})
    print("SHARED PREFLIGHT PASSED: all24 final models loaded; DQN12 + PPO12",flush=True)
    if a.check_only:return
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:6]
    out=a.out.resolve() if a.out else BASE/"results"/f"FINAL_SHARED_{'SMOKE' if a.smoke_only else 'EVALUATION'}_{stamp}"
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"runtime.json",runtime);write_json(out/"preflight.json",preflight)
    started=time.perf_counter()
    smoke=run_matrix(models,out/"smoke",a.device,config,True)
    print("DQN/PPO/RANDOM SMOKE PASSED",flush=True)
    full=None if a.smoke_only else run_matrix(models,out/"full",a.device,config,False)
    validate_models(models,BASE)
    write_json(out/"COMPLETE.json",dict(valid=True,formal_evaluation=not a.smoke_only,smoke=smoke,full=full,
        wall_seconds=time.perf_counter()-started,all_checkpoint_hashes_unchanged=True,
        no_learning=True,reserved_heldout_evaluated=False))
    archive=out.with_name(out.name+"_evidence.zip")
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob("*") if p.is_file()}
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(out/name,out.name+"/"+name)
        z.writestr(out.name+"/EVIDENCE_SHA256.json",json.dumps(files,indent=2))
    print("SMOKE COMPLETE" if a.smoke_only else "FINAL SHARED EVALUATION COMPLETE",flush=True)
    print("RESULTS_DIRECTORY:",out,flush=True)
    print("UPLOAD_THIS_RESULTS_ZIP:",archive,flush=True)
    return out


if __name__=="__main__":main()
