"""Audit saved results and build Task 1 tables/figures without retraining.

Run: python -m Analysis.DQN.dqn_final_analysis
Historical artifacts are verified against the pre-work SHA256 manifest.
"""
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Analysis/DQN/final_task1'
TABLE=ROOT/'Tables/DQN-Tables/final_task1'
FIG=OUT/'figures'
VARIANTS=('v0','v1','v2_history_only','v2')
LABELS={'v0':'V0 baseline','v1':'V1 novelty','v2_history_only':'History only','v2':'V2 novelty + history'}
COLORS=['#286A9A','#D78824','#3A9678']
SEEDS=(42,123,456)

def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def mean(values): return float(statistics.mean(values))
def sd(values): return float(statistics.stdev(values)) if len(values)>1 else 0.
def action_key(row):
    return f"ACTION6({row['x']},{row['y']})" if row['action']=='ACTION6' else row['action']

def describe_actions(rows):
    sequence=[action_key(row) for row in rows]
    counts=Counter(sequence)
    longest=current=0
    previous=None
    for action in sequence:
        current=current+1 if action==previous else 1
        longest=max(longest,current); previous=action
    entropy=-sum((n/len(sequence))*math.log2(n/len(sequence)) for n in counts.values())
    return dict(action_counts=dict(counts),unique_actions=len(counts),dominant_action=counts.most_common(1)[0][0],
        dominant_action_share=max(counts.values())/len(sequence),action_entropy_bits=entropy,
        longest_identical_action_run=longest,repeat_adjacent_fraction=sum(a==b for a,b in zip(sequence,sequence[1:]))/max(1,len(sequence)-1))

def preservation_audit():
    manifest=json.loads((OUT/'preservation_manifest.json').read_text())
    changes=[p for p,digest in manifest.items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest]
    assert not changes, changes
    return {'artifacts_checked':len(manifest),'unchanged':True}

def collect():
    records=[]; curves=[]; action_rows=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            name=f'dqn_{variant}_ls20_seed{seed}'
            rows=read_csv(ROOT/f'Logs/DQN/{name}.csv')
            cfg=json.loads((ROOT/f'Logs/DQN/{name}_config.json').read_text())
            cp=torch.load(ROOT/f'DQN/checkpoints/{name}.pt',weights_only=True,map_location='cpu')
            assert len(rows)==3000 and [int(r['global_step']) for r in rows]==list(range(1,3001))
            assert cp['global_step']==3000 and cp['seed']==seed
            expected=dict(steps=3000,episode_limit=300,batch_size=8,warmup=64,train_every=5,lr=.0001,
                gamma=.99,epsilon_start=1.,epsilon_end=.1,epsilon_decay=2000)
            assert all(cfg[k]==v for k,v in expected.items())
            frames=cfg.get('num_frames',1)
            assert frames==(4 if variant in ('v2','v2_history_only') else 1)
            assert cp['online_state_dict']['encoder.0.weight'].shape[1]==frames*16
            assert cfg.get('buffer_capacity',3000)==3000 and cfg.get('target_update_interval',250)==250
            intrinsic=[float(r['novelty_reward']) for r in rows] if 'novelty_reward' in rows[0] else [0.]*len(rows)
            # V0 reward is task-only by definition. This is a valid derived metric, not an extra historical log.
            task=[float(r['task_reward']) for r in rows] if 'task_reward' in rows[0] else [float(r['reward']) for r in rows]
            assert all(abs(float(r['reward'])-t-b)<1e-9 for r,t,b in zip(rows,task,intrinsic))
            if variant=='v2_history_only': assert all(b==0 for b in intrinsic) and cfg['novelty_beta']==0
            losses=[float(r['loss']) for r in rows if r['loss']]
            assert len(losses)==cp['gradient_steps']==588
            assert np.isfinite(losses).all()
            evaluation=json.loads((ROOT/f'Evaluations/DQN/{name}_eval_ls20_seed1001.json').read_text())
            trajectory=read_csv(ROOT/f'Evaluations/DQN/{name}_eval_ls20_seed1001.csv')
            assert evaluation['game']=='ls20-9607627b' and evaluation['evaluation_seed']==1001
            assert evaluation['max_actions']==200 and evaluation['actions_used']==len(trajectory)
            assert evaluation['gradient_updates_during_eval']==0
            assert max(int(r['levels_completed']) for r in trajectory)==evaluation['levels_completed']
            actions=describe_actions(trajectory)
            record=dict(variant=variant,label=LABELS[variant],training_seed=seed,num_frames=frames,
                novelty_beta=(cfg.get('novelty_beta',.01) if variant!='v0' else 0.),
                training_interactions=len(rows),gradient_updates=cp['gradient_steps'],
                training_max_levels=max(int(r['levels_completed']) for r in rows),
                first_level_reach_step=next((int(r['global_step']) for r in rows if int(r['levels_completed'])>0),None),
                task_reward_sum=sum(task),intrinsic_reward_sum=sum(intrinsic),total_reward_sum=sum(float(r['reward']) for r in rows),
                reward_component_source='logged separately' if 'task_reward' in rows[0] else 'V0 task=logged total; intrinsic=0 by disabled mechanism',
                unique_observation_count=int(rows[-1]['unique_states']) if 'unique_states' in rows[0] and variant!='v0' else None,
                loss_mean=mean(losses),loss_final=losses[-1],loss_last50_mean=mean(losses[-50:]),
                training_elapsed_seconds=float(rows[-1]['elapsed_seconds']),
                evaluation_seed=1001,evaluation_actions=evaluation['actions_used'],evaluation_levels=evaluation['levels_completed'],
                local_scorecard_score=evaluation.get('local_scorecard_score'),success=evaluation['success'],final_state=evaluation['final_state'],
                evaluation_gradient_updates=evaluation['gradient_updates_during_eval'],**actions,
                training_action_counts=dict(Counter(r['action'] for r in rows)),
                checkpoint=f'DQN/checkpoints/{name}.pt',training_csv=f'Logs/DQN/{name}.csv',
                evaluation_json=f'Evaluations/DQN/{name}_eval_ls20_seed1001.json')
            records.append(record)
            update=0
            for row,t,b in zip(rows,task,intrinsic):
                if row['loss']: update+=1
                curves.append(dict(variant=variant,seed=seed,global_step=int(row['global_step']),levels_completed=int(row['levels_completed']),
                    gradient_update=update if row['loss'] else None,loss=float(row['loss']) if row['loss'] else None,
                    task_reward=t,intrinsic_reward=b))
            for action,count in actions['action_counts'].items():
                action_rows.append(dict(variant=variant,seed=seed,action=action,count=count,share=count/len(trajectory)))
    return records,curves,action_rows

def aggregates(records):
    result=[]
    for variant in VARIANTS:
        group=[r for r in records if r['variant']==variant]
        a=dict(variant=variant,label=LABELS[variant],n_training_seeds=3,
            train_max_levels_per_seed=[r['training_max_levels'] for r in group],
            eval_levels_per_seed=[r['evaluation_levels'] for r in group],
            successes=sum(r['success'] for r in group),success_rate=mean([int(r['success']) for r in group]),
            final_state_counts=dict(Counter(r['final_state'] for r in group)))
        for metric in ('training_max_levels','evaluation_levels','evaluation_actions','local_scorecard_score',
            'task_reward_sum','intrinsic_reward_sum','loss_last50_mean','gradient_updates','dominant_action_share'):
            values=[r[metric] for r in group if r[metric] is not None]
            a[metric+'_mean']=mean(values) if values else None
            a[metric+'_sd']=sd(values) if values else None
        result.append(a)
    return result

def save_figure(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=320,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def figures(records,curves,aggregate):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'grid.alpha':.18,'figure.titlesize':16,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True,layout='constrained')
    for ax,variant in zip(axes.flat,VARIANTS):
        for seed,color,style in zip(SEEDS,COLORS,['-','--',':']):
            group=[r for r in curves if r['variant']==variant and r['seed']==seed]
            ax.plot([r['global_step'] for r in group],[r['levels_completed'] for r in group],
                color=color,linestyle=style,lw=1.7,label=f'Seed {seed}',alpha=.9)
        ax.set_title(LABELS[variant]); ax.set_ylim(-.06,1.2); ax.set_yticks([0,1])
        ax.set_xlabel('Selected environment actions'); ax.set_ylabel('Reported levels completed')
        ax.legend(loc='upper left',frameon=False,fontsize=8)
    reach=next((r for r in records if r['variant']=='v1' and r['training_max_levels']>0),None)
    if reach:
        axes[0,1].annotate(f"Seed 123 reaches level 1\nat action {reach['first_level_reach_step']}",
            xy=(reach['first_level_reach_step'],1),xytext=(350,.55),fontsize=9,
            arrowprops={'arrowstyle':'->','color':'#555'})
    fig.suptitle('Training progress across all four configurations')
    save_figure(fig,'01_training_progression')

    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
    values=np.array([[r['evaluation_levels'] for r in records if r['variant']==v] for v in VARIANTS])
    ax.imshow(values,cmap=ListedColormap(['#EDF1F5']),vmin=0,vmax=max(1,values.max()),aspect='auto')
    for i,variant in enumerate(VARIANTS):
        for j,r in enumerate([r for r in records if r['variant']==variant]):
            state='Game over' if r['final_state']=='GAME_OVER' else 'Budget exhausted'
            ax.text(j,i,f"{r['evaluation_levels']}/7 levels · {r['evaluation_actions']} actions\n{state}",ha='center',va='center',fontsize=11)
    ax.set_xticks(range(3),[f'Training seed {s}' for s in SEEDS])
    ax.set_yticks(range(4),[LABELS[v] for v in VARIANTS]); ax.grid(False)
    success_count=sum(r['success'] for r in records)
    ax.set_title(f'Frozen evaluation: {success_count}/12 successes; completed levels shown explicitly',pad=18)
    fig.supxlabel('ls20-9607627b · evaluation seed 1001 · maximum 200 actions · no learning',fontsize=10)
    save_figure(fig,'02_frozen_evaluation')

    fig,axes=plt.subplots(2,2,figsize=(11,7.5),sharex=True,sharey=True,layout='constrained')
    for ax,variant in zip(axes.flat,VARIANTS):
        for seed,color in zip(SEEDS,COLORS):
            losses=np.array([r['loss'] for r in curves if r['variant']==variant and r['seed']==seed and r['loss'] is not None])
            updates=np.arange(1,len(losses)+1)
            ax.plot(updates,np.maximum(losses,1e-12),color=color,lw=.4,alpha=.18)
            smooth=np.convolve(losses,np.ones(25)/25,mode='valid')
            ax.plot(updates[24:],np.maximum(smooth,1e-12),color=color,lw=1.6,label=f'Seed {seed}')
        ax.set_title(LABELS[variant]); ax.set_yscale('log'); ax.set_xlabel('Gradient updates'); ax.set_ylabel('Huber TD loss (log scale)')
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Learning curves: 25-update mean over raw losses')
    fig.supxlabel('Small TD loss indicates target fitting, not task success. Values are floored at 10⁻¹² for display.',fontsize=9)
    save_figure(fig,'03_training_loss')

    fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
    action_names=sorted({k for r in records for k in r['action_counts']})
    palette=['#286A9A','#E6A13B','#3A9678','#A45A7C','#7A74A1','#9B9C9D']
    left=np.zeros(len(records))
    for action,color in zip(action_names,palette):
        shares=np.array([r['action_counts'].get(action,0)/r['evaluation_actions'] for r in records])
        ax.barh(range(len(records)),shares,left=left,label=action,color=color,height=.72)
        for i,(share,start) in enumerate(zip(shares,left)):
            if share>.12: ax.text(start+share/2,i,f"{records[i]['action_counts'][action]}",ha='center',va='center',color='white',fontsize=9,fontweight='bold')
        left+=shares
    ax.set_yticks(range(len(records)),[f"{LABELS[r['variant']]} · seed {r['training_seed']}" for r in records])
    ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel('Share of selected actions (numbers show counts)')
    ax.set_title('Greedy policies: action variety remains limited',pad=15)
    ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False)
    save_figure(fig,'04_greedy_actions')

    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,metric,title in zip(axes,['task_reward_sum','intrinsic_reward_sum'],['Task reward: observed positive level deltas','Intrinsic reward: count-based bonus']):
        for i,variant in enumerate(VARIANTS):
            vals=[r[metric] for r in records if r['variant']==variant]
            ax.scatter([i-.12,i,i+.12],vals,c=COLORS,s=55,zorder=3,edgecolors='white')
            ax.plot([i-.23,i+.23],[mean(vals)]*2,color='#222',lw=2)
        ax.set_xticks(range(4),['V0','V1','History\nonly','V2']); ax.set_ylim(bottom=-.04 if metric=='task_reward_sum' else -1)
        ax.set_title(title); ax.set_ylabel('Sum over 3,000 selected actions')
    fig.suptitle('Reward components must be interpreted separately')
    fig.supxlabel('Colored dots: training seeds 42 / 123 / 456. Black line: seed mean. Intrinsic reward does not establish task success.',fontsize=9)
    save_figure(fig,'05_reward_components')

    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    matrix=np.zeros((2,2))
    ax.imshow(matrix,cmap=ListedColormap(['#EDF1F5']),vmin=0,vmax=1)
    placement={'v0':(0,0),'v1':(0,1),'v2_history_only':(1,0),'v2':(1,1)}
    for a in aggregate:
        row,col=placement[a['variant']]
        ax.text(col,row,f"{LABELS[a['variant']]}\nTrain maxima: {a['train_max_levels_per_seed']}\nFrozen levels: {a['eval_levels_per_seed']}\nSuccess: {a['successes']}/3",ha='center',va='center',linespacing=1.6,fontsize=12)
    ax.set_xticks([0,1],['Novelty off (β=0)','Novelty on (β=0.01)'])
    ax.set_yticks([0,1],['One frame','Four frames']); ax.grid(False)
    ax.set_title('2 × 2 ablation: no frozen-policy gain demonstrated',pad=20)
    fig.supxlabel('List order: training seeds 42, 123, 456. Three runs per cell; descriptive comparison only.',fontsize=10)
    save_figure(fig,'06_factorial_ablation')

def cross_results():
    rows=[]
    for path in sorted((ROOT/'Evaluations/DQN/zero_shot_final_task1').glob('*_seed1001.json')):
        e=json.loads(path.read_text())
        trajectory=read_csv(path.with_suffix('.csv'))
        variant=Path(e['checkpoint']).stem.removeprefix('dqn_').split('_ls20_seed')[0]
        seed=int(Path(e['checkpoint']).stem.split('_seed')[-1])
        rows.append(dict(variant=variant,training_seed=seed,training_game='ls20-9607627b',evaluation_game=e['game'],
            evaluation_seed=e['evaluation_seed'],levels=e['levels_completed'],actions=e['actions_used'],success=e['success'],
            final_state=e['final_state'],local_scorecard_score=e['local_scorecard_score'],**describe_actions(trajectory)))
    if rows:
        assert len(rows)==12
        save_csv(TABLE/'zero_shot_per_seed.csv',[{k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in rows])
        (TABLE/'zero_shot_per_seed.json').write_text(json.dumps(rows,indent=2))
    return rows

def main():
    torch.set_num_threads(1)
    TABLE.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    assert not (TABLE/'per_seed_results.json').exists(), 'Use a fresh output directory to avoid replacing completed analysis'
    preservation=preservation_audit()
    records,curves,action_rows=collect()
    aggregate=aggregates(records)
    save_csv(TABLE/'per_seed_results.csv',[{k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in records])
    (TABLE/'per_seed_results.json').write_text(json.dumps(records,indent=2))
    save_csv(TABLE/'variant_summary.csv',[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in a.items()} for a in aggregate])
    (TABLE/'variant_summary.json').write_text(json.dumps(aggregate,indent=2))
    save_csv(TABLE/'training_curves.csv',curves)
    save_csv(TABLE/'evaluation_action_counts.csv',action_rows)
    cross=cross_results()
    figures(records,curves,aggregate)
    versions={'python':platform.python_version(),'platform':platform.platform(),'packages':{}}
    for package in ('torch','numpy','matplotlib','arc-agi','arcengine','pillow'):
        try: versions['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions['packages'][package]='not registered'
    versions['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in list((ROOT/'DQN').glob('*.py'))+list((ROOT/'environment_files').rglob('*.py'))}
    (OUT/'environment_and_source_manifest.json').write_text(json.dumps(versions,indent=2))
    packages=sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata['Name']},key=str.lower)
    (OUT/'requirements_environment.txt').write_text('\n'.join(packages)+'\n')
    (OUT/'verification.json').write_text(json.dumps(dict(preservation=preservation,main_runs=len(records),
        matched_configs=True,checkpoint_metadata_verified=True,all_intrinsic_history_only_exactly_zero=True,
        all_gradient_updates=588,cross_game_runs=len(cross),figures=6),indent=2))
    print(json.dumps(aggregate,indent=2))
    print('Saved numerical tables and six PNG/PDF figures; historical hashes unchanged.')

if __name__=='__main__': main()
