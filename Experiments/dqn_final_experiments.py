"""Reproducible new experiments only; existing artifacts are never replaced."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
AUDIT=ROOT/'Analysis/DQN/final_task1'

def run(command,console):
    path=ROOT/console
    with path.open('x',encoding='utf-8') as file:
        file.write('COMMAND: '+subprocess.list2cmdline(command)+'\n\n'); file.flush()
        subprocess.run(command,stdout=file,stderr=subprocess.STDOUT,check=True)
    print(f'Completed: {console}',flush=True)

def verify_history_run(seed,suffix='',steps=3000):
    import torch
    name=f'dqn_v2_history_only_ls20_seed{seed}{suffix}'
    config=json.loads((ROOT/f'Logs/DQN/{name}_config.json').read_text())
    with (ROOT/f'Logs/DQN/{name}.csv').open() as file:
        rows=list(csv.DictReader(file))
    assert len(rows)==steps
    assert all(float(r['novelty_reward'])==0. for r in rows)
    assert all(float(r['reward'])==float(r['task_reward']) for r in rows)
    cp=torch.load(ROOT/f'DQN/checkpoints/{name}.pt',weights_only=True)
    assert config['num_frames']==4 and config['novelty_beta']==0.
    assert cp['online_state_dict']['encoder.0.weight'].shape[1]==64
    assert cp['gradient_steps']==sum(bool(r['loss']) for r in rows)
    print(f'Verified {name}: steps={steps}, gradient_updates={cp["gradient_steps"]}, intrinsic_reward=0 exactly',flush=True)

def ablation():
    run([sys.executable,'-m','DQN.test_task1'], 'Analysis/DQN/final_task1/tests_console.txt')
    common=['-m','DQN.train','--variant','v2_history_only','--game','ls20','--episode-limit','300',
        '--batch-size','8','--warmup','64','--train-every','5','--lr','0.0001','--gamma','0.99',
        '--epsilon-start','1.0','--epsilon-end','0.1','--epsilon-decay','2000',
        '--buffer-capacity','3000','--target-update-interval','250','--num-frames','4','--novelty-beta','0.0']
    # Smoke crosses a reset and performs actual updates; full-run settings remain matched.
    smoke=[sys.executable]+common+['--seed','42','--steps','60','--run-suffix','smoke60','--warmup','10','--episode-limit','30']
    run(smoke,'Logs/DQN/dqn_v2_history_only_ls20_seed42_smoke60_console.txt')
    verify_history_run(42,'_smoke60',60)
    for seed in (42,123,456):
        name=f'dqn_v2_history_only_ls20_seed{seed}'
        run([sys.executable]+common+['--seed',str(seed),'--steps','3000'],f'Logs/DQN/{name}_console.txt')
        verify_history_run(seed)
        run([sys.executable,'-m','DQN.evaluate','--checkpoint',f'DQN/checkpoints/{name}.pt',
            '--game','ls20','--seed','1001','--max-actions','200'],f'Evaluations/DQN/{name}_console.txt')

def diagnostics():
    import inspect
    import random
    import numpy as np
    import torch
    import arc_agi
    from arcengine import GameAction
    torch.set_num_threads(1)
    traces=[]
    signatures=[]
    for seed in (1001,1002,1003):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        arc=arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
        env=arc.make('ls20-9607627b',seed=seed)
        o=env.reset()
        signatures.append(str(inspect.signature(env._game_class)))
        trace=[]
        def capture(o):
            return {'frame_sha256':hashlib.sha256(np.asarray(o.frame[-1],dtype=np.uint8).tobytes()).hexdigest(),
                'state':o.state.name,'levels':o.levels_completed}
        trace.append(capture(o))
        for i in range(80):
            o=env.step([GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4][i%4])
            trace.append(capture(o))
            if o.state.name in ('WIN','GAME_OVER'): break
        traces.append({'requested_seed':seed,'effective_game_seed':env._game._seed,'trace':trace})
    equal=all(t['trace']==traces[0]['trace'] for t in traces)
    assert equal and all(t['effective_game_seed']==0 for t in traces)
    result={'game':'ls20-9607627b','class_signatures':signatures,'identical_traces':equal,
        'protocol':'Initial frame plus 80 scripted actions [ACTION1,ACTION2,ACTION3,ACTION4] repeated; independent environments and process RNG seeds.',
        'interpretation':'Wrapper only forwards seed when constructor accepts it. Ls20 constructor has no seed parameter; source has no random/RNG use. Additional evaluation seeds are not independent environment samples.',
        'traces':traces}
    (AUDIT/'evaluation_seed_diagnostic.json').write_text(json.dumps(result,indent=2))
    # Reset semantics are tested separately; simulated progression is never a policy result.
    arc=arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    env=arc.make('ls20',seed=42)
    records=[]
    def record(label,o): records.append(dict(event=label,levels=o.levels_completed,full_reset=o.full_reset))
    record('initial_reset',env.reset())
    env.step(GameAction.ACTION1); record('reset_after_action',env.reset())
    env._game.next_level(); env.step(GameAction.ACTION1); record('reset_immediately_after_simulated_advance',env.reset())
    env._game.next_level(); env.step(GameAction.ACTION1); env.step(GameAction.ACTION1)
    record('reset_after_action_in_simulated_level2',env.reset())
    assert [(r['levels'],r['full_reset']) for r in records]==[(0,True),(0,False),(0,True),(1,False)]
    (AUDIT/'reset_diagnostic.json').write_text(json.dumps({'simulated_not_policy_progress':True,'records':records},indent=2))
    print('Seed and reset diagnostics passed',flush=True)

def cross_game():
    # Both games use 64x64, colors 0..15; ACTION6 clicks are supported by the codec.
    # This is zero-shot transfer with an unsupervised spatial action head, not training on vc33.
    out=ROOT/'Evaluations/DQN/zero_shot_final_task1'
    out.mkdir(parents=True,exist_ok=True)
    for variant in ('v0','v1','v2_history_only','v2'):
        for seed in (42,123,456):
            name=f'dqn_{variant}_ls20_seed{seed}'
            run([sys.executable,'-m','DQN.evaluate','--checkpoint',f'DQN/checkpoints/{name}.pt',
                '--game','vc33-5430563c','--seed','1001','--max-actions','200',
                '--output-dir',str(out)],f'Evaluations/DQN/zero_shot_final_task1/{name}_console.txt')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['ablation','diagnostics','cross-game'])
    stage=parser.parse_args().stage
    if stage=='ablation': ablation()
    elif stage=='diagnostics': diagnostics()
    else: cross_game()
