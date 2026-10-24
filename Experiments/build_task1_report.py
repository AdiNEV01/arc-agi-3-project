"""Render the report and handover from audited numerical outputs."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
TABLE=ROOT/'Tables/DQN-Tables/final_task1'
OUT=ROOT/'Reports/DQN-Reports/final_task1'
ANALYSIS=ROOT/'Analysis/DQN/final_task1'
VARIANTS=('v0','v1','v2_history_only','v2')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=json.loads((TABLE/'per_seed_results.json').read_text())
    aggregates=json.loads((TABLE/'variant_summary.json').read_text())
    cross=json.loads((TABLE/'zero_shot_per_seed.json').read_text()) if (TABLE/'zero_shot_per_seed.json').exists() else []
    verification=json.loads((ANALYSIS/'verification.json').read_text())
    template=(Path(__file__).with_name('task1_report_template.md')).read_text(encoding='utf-8')
    table=['| Configuration | Training max levels | Frozen levels | Eval actions | Local score | Success |',
        '|---|---|---|---|---|---|']
    reward=['| Configuration | Task reward | Intrinsic reward | Updates per run |',
        '|---|---:|---:|---:|']
    for a in aggregates:
        group=[r for r in records if r['variant']==a['variant']]
        def listing(metric): return ', '.join(str(r[metric]) for r in group)
        table.append(f"| {a['label']} | {listing('training_max_levels')} | {listing('evaluation_levels')} | {listing('evaluation_actions')} | {listing('local_scorecard_score')} | {a['successes']}/3 |")
        reward.append(f"| {a['label']} | {a['task_reward_sum_mean']:.3f} ± {a['task_reward_sum_sd']:.3f} | {a['intrinsic_reward_sum_mean']:.3f} ± {a['intrinsic_reward_sum_sd']:.3f} | 588 |")
    collapse=sum(r['unique_actions']==1 for r in records)
    v2seed42=next(r for r in records if r['variant']=='v2' and r['training_seed']==42)
    hist=[r for r in records if r['variant']=='v2_history_only']
    behavior=f"{collapse} of twelve frozen policies selected one identical action throughout evaluation. V2 seed 42 selected ACTION2 38 times and ACTION3 91 times (129 actions), yet completed no levels. "
    behavior+='History-only action counts were '+ '; '.join(f"seed {r['training_seed']}: "+', '.join(f'{a} × {n}' for a,n in r['action_counts'].items()) for r in hist)+'.'
    unique='; '.join(f"{next(r['label'] for r in records if r['variant']==v)}: "+', '.join(str(r['unique_observation_count']) for r in records if r['variant']==v) for v in VARIANTS if v!='v0')+' (seed order 42, 123, 456)'
    if cross:
        cross_text='The locally available **vc33-5430563c** uses the same grid/color representation and ACTION6 coordinate clicks supported by the codec. All twelve checkpoints trained only on ls20 were evaluated without adaptation on vc33 using seed 1001 and a 200-action budget, in OFFLINE mode with the cached game version. This is zero-shot cross-game evaluation, not within-game learning or training on vc33.\n\n'
        cross_text+='| Configuration | Zero-shot levels (42, 123, 456) | Actions | Success |\n|---|---|---|---|\n'
        for v in VARIANTS:
            group=sorted([r for r in cross if r['variant']==v],key=lambda r:(42,123,456).index(r['training_seed']))
            label=next(r['label'] for r in records if r['variant']==v)
            cross_text+=f"| {label} | {', '.join(str(r['levels']) for r in group)} | {', '.join(str(r['actions']) for r in group)} | {sum(r['success'] for r in group)}/3 |\n"
        cross_text+='\nThe spatial output head received no supervised selected-action TD targets during ls20 training because ls20 permits only directional actions. Encoder updates can still affect spatial values, but click-head parameters are not learned from clicks. The transfer check is consequently a limited compatibility and distribution-shift stress test, not strong evidence about general reasoning or the causal benefit of either improvement. Coordinate-level action counts and entropy are saved separately; ACTION6 family counts alone would hide variation in click position. No additional models were trained. No existing team evaluation adapter was present to coordinate with; the DQN frozen evaluator was reused.'
    else:
        cross_text='Additional games were inspected but not evaluated. No cross-game performance claim is made.'
    zero=all(r['evaluation_levels']==0 and not r['success'] for r in records)
    conclusion='none of the four configurations completed a level in frozen evaluation, so neither improvement demonstrated a frozen-policy performance gain.' if zero else 'the measured results above describe performance at this budget; three seeds do not support a broad performance claim.'
    values={'V1_REACH':next(str(r['first_level_reach_step']) for r in records if r['variant']=='v1' and r['training_seed']==123),
        'PARAM1':'164,423','PARAM4':'178,247','MAIN_TABLE':'\n'.join(table),'AGGREGATE_TABLE':'\n'.join(reward),
        'BEHAVIOUR_TEXT':behavior,'UNIQUE_TEXT':unique,'CROSS_TEXT':cross_text,
        'PRESERVED':str(verification['preservation']['artifacts_checked']),'CONCLUSION':conclusion}
    for key,value in values.items(): template=template.replace('{{'+key+'}}',value)
    assert '{{' not in template
    (OUT/'DQN_Task1_report.md').write_text(template,encoding='utf-8')
    header='''# Task 1 DQN handover

Prepared 20 October 2026 for the group assignment due 27 October 2026.

## Start here

- `DQN_Task1_report.md`: report contribution, equations, results, limitations and six figures.
- `../../../Tables/DQN-Tables/final_task1/per_seed_results.csv` and `.json`: all 12 within-game runs.
- `../../../Tables/DQN-Tables/final_task1/variant_summary.csv` and `.json`: descriptive aggregates.
- `../../../Tables/DQN-Tables/final_task1/training_curves.csv`: underlying numerical curve/reward data.
- `../../../Tables/DQN-Tables/final_task1/evaluation_action_counts.csv`: greedy action counts/shares.
- `../../../Analysis/DQN/final_task1/figures/`: six 320-dpi PNGs and vector PDFs, numbered 01–06.
- `../../../Analysis/DQN/final_task1/verification.json`: checks of configurations, rows, updates and preservation.
- `../../../Analysis/DQN/final_task1/evaluation_seed_diagnostic.json` and `reset_diagnostic.json`: diagnostic evidence.
- `../../../Analysis/DQN/final_task1/environment_and_source_manifest.json`: package/platform versions and source hashes.
- `../../../Analysis/DQN/final_task1/requirements_environment.txt`: installed package versions for this Windows environment.
- `../../../Analysis/DQN/final_task1/artifact_manifest.csv`: all deliverables and SHA256 checksums.

Paths below are relative to the repository root. Run commands there with `.venv312`.

## Implementation and reproduction

`DQN/train.py` accepts `v0`, `v1`, `v2`, `v2_history_only`. The history-only variant defaults to four frames and β=0 and rejects a conflicting frame count or beta. All other variants retain their prior defaults. Replay capacity and target-copy interval are now explicit optional settings; defaults remain 3000 and 250. `DQN/evaluate.py` reconstructs frame count from checkpoint config (legacy fallback 1) and refuses existing CSV/JSON outputs. Use a new `--output-dir` for an intentional separate evaluation.

Tests and the new runner:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\\.venv312\\Scripts\\python.exe -m DQN.test_task1
.\\.venv312\\Scripts\\python.exe -m Experiments.dqn_final_experiments ablation
```

The ablation command was already completed. It intentionally refuses existing console/config/checkpoint outputs; do not run it again over completed results. To reproduce training separately, use a fresh checkout/output location or a new training `--run-suffix`. For each seed in 42,123,456, the exact full-run settings are:

```powershell
.\\.venv312\\Scripts\\python.exe -m DQN.train --variant v2_history_only --game ls20 --seed 42 --steps 3000 --episode-limit 300 --batch-size 8 --warmup 64 --train-every 5 --lr 0.0001 --gamma 0.99 --epsilon-start 1 --epsilon-end 0.1 --epsilon-decay 2000 --buffer-capacity 3000 --target-update-interval 250 --num-frames 4 --novelty-beta 0
.\\.venv312\\Scripts\\python.exe -m DQN.evaluate --checkpoint DQN/checkpoints/dqn_v2_history_only_ls20_seed42.pt --game ls20 --seed 1001 --max-actions 200
```

Frozen evaluation uses epsilon zero, checks unchanged parameters and records actual evaluation gradient updates. A 60-step smoke test (warmup10, episode-limit30, seed42) is saved with `_smoke60` and is excluded from full-run summaries. Six regression tests check variant defaults, β=0, frame/channel order, replay copies/terminal learning, all nine legacy checkpoint representations, and greedy legal-action masking.

Diagnostics and optional zero-shot transfer:

```powershell
.\\.venv312\\Scripts\\python.exe -m Experiments.dqn_final_experiments diagnostics
$env:OPERATION_MODE='offline'
.\\.venv312\\Scripts\\python.exe -m Experiments.dqn_final_experiments cross-game
Remove-Item Env:OPERATION_MODE
```

These were completed. The cross-game runner refuses existing console files. Its 12 vc33 evaluations are under `Evaluations/DQN/zero_shot_final_task1/` and summarised in `Tables/DQN-Tables/final_task1/zero_shot_per_seed.csv` / `.json`. No training took place on vc33.

Analysis and report generation (completed; do not replace the completed numerical snapshot):

```powershell
.\\.venv312\\Scripts\\python.exe -m Analysis.DQN.dqn_final_analysis
.\\.venv312\\Scripts\\python.exe -m Experiments.build_task1_report
```

The analysis script refuses an existing final numerical result. Use fresh output directories when producing a new analysis version. The report builder fills the adjacent Markdown template from those audited data. No spreadsheet dependency is required; analysis uses NumPy and Matplotlib.

## New experiment files

For each seed `s` in 42,123,456:

- `DQN/checkpoints/dqn_v2_history_only_ls20_seed{s}.pt`
- `Logs/DQN/dqn_v2_history_only_ls20_seed{s}.csv`
- `Logs/DQN/dqn_v2_history_only_ls20_seed{s}_config.json`
- `Logs/DQN/dqn_v2_history_only_ls20_seed{s}_console.txt`
- `Evaluations/DQN/dqn_v2_history_only_ls20_seed{s}_eval_ls20_seed1001.csv`
- `Evaluations/DQN/dqn_v2_history_only_ls20_seed{s}_eval_ls20_seed1001.json`
- `Evaluations/DQN/dqn_v2_history_only_ls20_seed{s}_console.txt`

The original nine V0/V1/V2 checkpoints, configs, logs and evaluation files remain unchanged. SHA256 hashes for 60 original artifacts were recorded before implementation and checked after execution. Pre-change training/evaluation source is retained in `Analysis/DQN/final_task1/*.before.txt`; source changes are separately reviewable in `implementation_changes.diff`.

## Results and limitations

'''
    tail='''

Every full run has 3,000 selected actions and 588 gradient updates. Every history-only intrinsic reward is exactly zero. Training maxima are not repeated completions; V1 seed123's exploratory level event is not a frozen success. Local scorecards are not official Kaggle scores. All statistics here are descriptive over three training seeds; there is only one deterministic within-game evaluation instance.

Evaluation seeds 1001–1003 do not randomize this ls20 class: the wrapper cannot pass a seed to its zero-argument constructor, and the diagnostic traces match. No duplicate model evaluations at those seeds were added as independent evidence. Training RNG seeds remain meaningful. Reset often restarts only the current level; an immediate reset after entering a new level can restart the whole game. Controller resets are outside the selected-action budget, and nonterminal time-limit transitions retain the original bootstrap behavior. These choices were matched, not repaired silently. Runtime comparisons with old runs are unsupported. Old checkpoints do not contain replay/RNG state for exact continuation. The current source/dependency manifest does not reconstruct the unrecorded historical software state.

For vc33, only zero-shot transfer was tested. The spatial output head was not trained on selected click actions during ls20 training; read its supplementary results with that limitation.

## Git and submission handoff

The Task 1 implementation and results are prepared for a local commit on `feature/dqn-task1`. Source, trained checkpoints, CSV/JSON results, cached games, report and figures are selected for team review. The original empty `DQN/dqn_baseline.py` placeholder is replaced by the working baseline in `dqn_agent.py`, `q_network.py` and `train.py`. The local `.venv312` and raw console files are excluded; all local files remain on disk. No push or merge was performed. The original audit snapshots describe the checkout before packaging. Use `team_artifact_manifest.csv` for shared-file checksums; the original full local artifact manifest also references excluded consoles.

Before submission: integrate this Markdown contribution/figures into the group report; align heading/figure numbering and citation style; check the assignment rubric and required team evaluation/submission interface (none was present locally); complete the group's required submission packaging. No further DQN run is required to substantiate the limited claims made here. If an official Kaggle submission is required by the rubric, it remains a separate team action; local scorecards cannot substitute for it.
'''
    (OUT/'README.md').write_text(header+'\n'.join(table)+'\n\n'+cross_text+tail,encoding='utf-8')
    manifest=[]
    folders=[ROOT/'Analysis/DQN/final_task1',TABLE,OUT,ROOT/'Evaluations/DQN/zero_shot_final_task1']
    paths=[p for folder in folders for p in folder.rglob('*') if p.is_file()]
    paths += [p for folder in ('DQN/checkpoints','Logs/DQN','Evaluations/DQN') for p in (ROOT/folder).glob('*v2_history_only*') if p.is_file()]
    paths += [ROOT/p for p in ('DQN/train.py','DQN/evaluate.py','DQN/test_task1.py','Experiments/dqn_final_experiments.py',
        'Experiments/build_task1_report.py','Experiments/task1_report_template.md','Experiments/final_task1_verify.py','Analysis/DQN/dqn_final_analysis.py')]
    for p in sorted(set(paths)):
        # The current assembly console is still open; it is not a research artifact.
        if p.name in ('artifact_manifest.csv','manifest_build_console.txt'): continue
        manifest.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with (ANALYSIS/'artifact_manifest.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=['path','bytes','sha256']); writer.writeheader(); writer.writerows(manifest)
    print(f'Report, handover and {len(manifest)}-file artifact manifest written')

if __name__=='__main__': main()
