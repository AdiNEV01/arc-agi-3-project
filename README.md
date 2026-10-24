# ARC-AGI-3 integration

Local integration of all three task branches and current remote main, on
`integration/2026-10-24`. Nothing has been pushed or merged to main.

**Start with the [integration audit and today's plan](docs/INTEGRATION_AUDIT_2026-10-24.md).**

## Shared evaluation

- `arc_env.gym_adapter.ArcGymEnv`: shared grid/action/reset contract.
- `agents.dqn_agent:DQNPolicy`: frozen original DQN checkpoints.
- `agents.ppo_agent:PPOPolicy`: frozen seven-action MaskablePPO checkpoints.
- Root `evaluate.py`: common official local scorecard and trajectories.
- `agents.kaggle_bridge.PolicyBridge`: helper for the official starter. Actual
  Kaggle execution and submission evidence are still needed.

Legacy DQN/PPO training scripts retain their original contracts. They have not
been silently migrated to shared-wrapper training. Original PPO scripts run from
the `PPO` directory, where `arc_env.py` differs from the root `arc_env` package.
Use the root evaluator for new evaluation evidence.

## Setup and checks

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest tests DQN/test_task1.py -q
.\.venv\Scripts\python.exe -m Experiments.integration_smoke --out results/my_new_smoke
```

DQN artifact tests and the complete smoke require the separately distributed
`DQN/checkpoints/*.pt`. These exist in this checkout but are Git-ignored. Smoke
also requires the cached ls20 game. Do not inspect hidden game rules to design agents.

This chat used the original `.venv312` interpreter plus libraries in the isolated
`.integration-deps` folder, leaving the original environment unchanged. Tested
versions are in `results/integration_audit/runtime_versions.json`. The original
PPO machine freeze is preserved in `requirements-ppo-original.txt`; the root
requirements avoid its platform-specific CUDA wheel pin.

24 tests passed. Final smoke evidence is in
`results/integration_smoke_verified_20261024/verification.json`: 14 compatibility
runs, not comparative performance evidence. Its full-pixel click protocol differs
from the original Task 3 8x8-click random baseline; do not pool the results.

The historical split in `Configs/eval.yaml` conflicts with actual agent usage.
Read the audit before claiming untouched-heldout performance. The historical
PPO CSV uses proxy scores and unavailable 75K-450K checkpoints; only three
12,288-step PPO pilots are available. See `PPO/Report/ppo_report_draft.tex`.

## Task 1 — DQN implementation and final results

The complete DQN contribution is on branch `feature/dqn-task1`.

- [Start here: DQN results and team guide](DQN/README.md)
- [Report contribution](Reports/DQN-Reports/final_task1/DQN_Task1_report.md)
- [Reproduction commands and full output paths](Reports/DQN-Reports/final_task1/README.md)
- [Per-seed comparison CSV](Tables/DQN-Tables/final_task1/per_seed_results.csv)
- [Report figures (PNG and PDF)](Analysis/DQN/final_task1/figures/)

All four configurations (baseline, novelty, history only, and novelty + history) were trained for 3,000 interactions per seed on ls20. All twelve frozen evaluations completed zero levels. V1 seed123 reached one level during exploratory training only. The report documents the reset, seeding and zero-shot transfer limitations; local scores are not official Kaggle scores.
