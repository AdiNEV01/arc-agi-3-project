# Task 1 DQN handover

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
.\.venv312\Scripts\python.exe -m DQN.test_task1
.\.venv312\Scripts\python.exe -m Experiments.dqn_final_experiments ablation
```

The ablation command was already completed. It intentionally refuses existing console/config/checkpoint outputs; do not run it again over completed results. To reproduce training separately, use a fresh checkout/output location or a new training `--run-suffix`. For each seed in 42,123,456, the exact full-run settings are:

```powershell
.\.venv312\Scripts\python.exe -m DQN.train --variant v2_history_only --game ls20 --seed 42 --steps 3000 --episode-limit 300 --batch-size 8 --warmup 64 --train-every 5 --lr 0.0001 --gamma 0.99 --epsilon-start 1 --epsilon-end 0.1 --epsilon-decay 2000 --buffer-capacity 3000 --target-update-interval 250 --num-frames 4 --novelty-beta 0
.\.venv312\Scripts\python.exe -m DQN.evaluate --checkpoint DQN/checkpoints/dqn_v2_history_only_ls20_seed42.pt --game ls20 --seed 1001 --max-actions 200
```

Frozen evaluation uses epsilon zero, checks unchanged parameters and records actual evaluation gradient updates. A 60-step smoke test (warmup10, episode-limit30, seed42) is saved with `_smoke60` and is excluded from full-run summaries. Six regression tests check variant defaults, β=0, frame/channel order, replay copies/terminal learning, all nine legacy checkpoint representations, and greedy legal-action masking.

Diagnostics and optional zero-shot transfer:

```powershell
.\.venv312\Scripts\python.exe -m Experiments.dqn_final_experiments diagnostics
$env:OPERATION_MODE='offline'
.\.venv312\Scripts\python.exe -m Experiments.dqn_final_experiments cross-game
Remove-Item Env:OPERATION_MODE
```

These were completed. The cross-game runner refuses existing console files. Its 12 vc33 evaluations are under `Evaluations/DQN/zero_shot_final_task1/` and summarised in `Tables/DQN-Tables/final_task1/zero_shot_per_seed.csv` / `.json`. No training took place on vc33.

Analysis and report generation (completed; do not replace the completed numerical snapshot):

```powershell
.\.venv312\Scripts\python.exe -m Analysis.DQN.dqn_final_analysis
.\.venv312\Scripts\python.exe -m Experiments.build_task1_report
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

| Configuration | Training max levels | Frozen levels | Eval actions | Local score | Success |
|---|---|---|---|---|---|
| V0 baseline | 0, 0, 0 | 0, 0, 0 | 200, 129, 200 | 0.0, 0.0, 0.0 | 0/3 |
| V1 novelty | 0, 1, 0 | 0, 0, 0 | 129, 200, 129 | 0.0, 0.0, 0.0 | 0/3 |
| History only | 0, 0, 0 | 0, 0, 0 | 129, 129, 200 | 0.0, 0.0, 0.0 | 0/3 |
| V2 novelty + history | 0, 0, 0 | 0, 0, 0 | 129, 129, 129 | 0.0, 0.0, 0.0 | 0/3 |

The locally available **vc33-5430563c** uses the same grid/color representation and ACTION6 coordinate clicks supported by the codec. All twelve checkpoints trained only on ls20 were evaluated without adaptation on vc33 using seed 1001 and a 200-action budget, in OFFLINE mode with the cached game version. This is zero-shot cross-game evaluation, not within-game learning or training on vc33.

| Configuration | Zero-shot levels (42, 123, 456) | Actions | Success |
|---|---|---|---|
| V0 baseline | 0, 0, 0 | 50, 50, 50 | 0/3 |
| V1 novelty | 0, 0, 0 | 50, 50, 50 | 0/3 |
| History only | 0, 0, 0 | 50, 50, 50 | 0/3 |
| V2 novelty + history | 0, 0, 0 | 50, 50, 50 | 0/3 |

The spatial output head received no supervised selected-action TD targets during ls20 training because ls20 permits only directional actions. Encoder updates can still affect spatial values, but click-head parameters are not learned from clicks. The transfer check is consequently a limited compatibility and distribution-shift stress test, not strong evidence about general reasoning or the causal benefit of either improvement. Coordinate-level action counts and entropy are saved separately; ACTION6 family counts alone would hide variation in click position. No additional models were trained. No existing team evaluation adapter was present to coordinate with; the DQN frozen evaluator was reused.

Every full run has 3,000 selected actions and 588 gradient updates. Every history-only intrinsic reward is exactly zero. Training maxima are not repeated completions; V1 seed123's exploratory level event is not a frozen success. Local scorecards are not official Kaggle scores. All statistics here are descriptive over three training seeds; there is only one deterministic within-game evaluation instance.

Evaluation seeds 1001–1003 do not randomize this ls20 class: the wrapper cannot pass a seed to its zero-argument constructor, and the diagnostic traces match. No duplicate model evaluations at those seeds were added as independent evidence. Training RNG seeds remain meaningful. Reset often restarts only the current level; an immediate reset after entering a new level can restart the whole game. Controller resets are outside the selected-action budget, and nonterminal time-limit transitions retain the original bootstrap behavior. These choices were matched, not repaired silently. Runtime comparisons with old runs are unsupported. Old checkpoints do not contain replay/RNG state for exact continuation. The current source/dependency manifest does not reconstruct the unrecorded historical software state.

For vc33, only zero-shot transfer was tested. The spatial output head was not trained on selected click actions during ls20 training; read its supplementary results with that limitation.

## Git and submission handoff

The Task 1 implementation and results are prepared for a local commit on `feature/dqn-task1`. Source, trained checkpoints, CSV/JSON results, cached games, report and figures are selected for team review. The original empty `DQN/dqn_baseline.py` placeholder is replaced by the working baseline in `dqn_agent.py`, `q_network.py` and `train.py`. The local `.venv312` and raw console files are excluded; all local files remain on disk. No push or merge was performed. The original audit snapshots describe the checkout before packaging. Use `team_artifact_manifest.csv` for shared-file checksums; the original full local artifact manifest also references excluded consoles.

Before submission: integrate this Markdown contribution/figures into the group report; align heading/figure numbering and citation style; check the assignment rubric and required team evaluation/submission interface (none was present locally); complete the group's required submission packaging. No further DQN run is required to substantiate the limited claims made here. If an official Kaggle submission is required by the rubric, it remains a separate team action; local scorecards cannot substitute for it.
