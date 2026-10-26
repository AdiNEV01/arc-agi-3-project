# DQN: current and historical experiments

The current controlled matrix has **12 completed models at 503,808 actions each**. Start with [the extended report](../Reports/DQN-Reports/extended_503808/TRAINING_RESULTS.md), [training tables and audits](../results/dqn_extended_503808/README.md), [the checkpoint registry](../Configs/dqn_503808_checkpoints.json) and [artifact access](../docs/ARTIFACT_ACCESS.md). Formal frozen evaluation of these final models remains outstanding.

Extended training uses `DQN/extended_train.py` and the shared environment. The original training path and results below remain historical evidence.

## Historical 3K Task 1 experiments

This branch contains the implementation, trained checkpoints, training/evaluation data, consolidated tables, figures and report contribution. The empty original `dqn_baseline.py` placeholder has been replaced by the working baseline in `dqn_agent.py`, `q_network.py` and `train.py`.

## Results

Seeds are listed in order 42, 123, 456. Every full run used 3,000 selected actions and 588 gradient updates.

| Configuration | Training max levels | Frozen levels | Frozen actions |
|---|---|---|---|
| V0 baseline | 0, 0, 0 | 0, 0, 0 | 200, 129, 200 |
| V1 novelty | 0, 1, 0 | 0, 0, 0 | 129, 200, 129 |
| History only (β=0) | 0, 0, 0 | 0, 0, 0 | 129, 129, 200 |
| V2 novelty + history | 0, 0, 0 | 0, 0, 0 | 129, 129, 129 |

All frozen evaluations used seed1001, epsilon zero, and a 200-action limit. All local scores were zero. The history-only intrinsic reward was exactly zero. Twelve supplementary vc33 zero-shot tests also completed zero levels; the spatial action head was not trained on clicks during ls20 training. No official Kaggle score is claimed.

## Files to review

- [Report section, with equations and figures](../Reports/DQN-Reports/final_task1/DQN_Task1_report.md)
- [Handover and exact commands](../Reports/DQN-Reports/final_task1/README.md)
- [All per-seed metrics](../Tables/DQN-Tables/final_task1/per_seed_results.csv) / [JSON](../Tables/DQN-Tables/final_task1/per_seed_results.json)
- [Aggregate summaries](../Tables/DQN-Tables/final_task1/variant_summary.csv)
- [Six report figures](../Analysis/DQN/final_task1/figures/)
- [Checkpoints](checkpoints/)
- [Training CSVs and configurations](../Logs/DQN/)
- [Frozen trajectories and summaries](../Evaluations/DQN/)
- [Cached, versioned games](../environment_files/)
- [Tests](test_task1.py), [experiment runner](../Experiments/dqn_final_experiments.py), [analysis](../Analysis/DQN/dqn_final_analysis.py)

## Run locally

Use Python 3.12 with the package versions recorded in [requirements_environment.txt](../Analysis/DQN/final_task1/requirements_environment.txt). That file records the original Windows CPU environment; adjust platform-specific package availability when installing elsewhere. The local `.venv312` itself is not committed.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m DQN.test_task1
```

Completed experiment names are protected against accidental overwrite. For new training use `--run-suffix`; for a separate evaluation use `--output-dir`. Do not rerun completed experiments merely to view their results.

Training reset calls may restart only the current level and are not counted in the selected-action budget. Evaluation seeds do not randomize the cached ls20 class. See the report before interpreting the results.

## Sharing and provenance

Task 1 source, checkpoints, CSV/JSON results and figures are included for team review. Raw execution consoles remain local because they contain transient API identifiers and machine-specific output. CSVs, configs, summaries, diagnostics and reproducible scripts are included. `Analysis/DQN/final_task1/artifact_manifest.csv` is the original full local audit and includes those local-only console paths; `team_artifact_manifest.csv` is the checksum inventory of committed Git blob contents; checkout line endings may differ across systems.

Before the final group submission, integrate the report section, align citations/figure numbers, check the assignment rubric and package the required team deliverables. No shared branch was merged.
