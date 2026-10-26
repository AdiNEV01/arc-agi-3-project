# ARC-AGI-3: DQN and PPO

This team repository contains DQN and MaskablePPO baselines, novelty and memory improvements, ablations, shared frozen-policy evaluation, and audited experiment evidence.

**Current state:** training is complete for both 12-model final matrices at **503,808 training interactions per model**. The common final evaluation, joint report and competition evidence remain outstanding. See [submission status](SUBMISSION_STATUS.md) for the complete checklist.

## Start here

- [Extended DQN report](Reports/DQN-Reports/extended_503808/TRAINING_RESULTS.md), [training summaries and audit](results/dqn_extended_503808/README.md), [figures](Analysis/DQN/extended_503808/figures/).
- [DQN checkpoint registry](Configs/dqn_503808_checkpoints.json) and [actual training configuration](Configs/dqn_503808_training.json).
- [PPO final checkpoint registry](Configs/ppo_final_checkpoints.json) and [verified metadata](results/ppo_finals_verified_20261024_complete/verification.json).
- [Model access and reproduction](docs/ARTIFACT_ACCESS.md).
- [Historical PPO failure analysis](PPO/Failures/README.md), [historical DQN 3K experiments](DQN/README.md).
- [Common interface and scoring contract](docs/interface.md), [notebook guide](Notebooks/README.md).

| Final matrix | Variants | Independent training seeds | Interactions per model |
| --- | --- | --- | ---: |
| DQN | baseline, novelty, history only, history + novelty | 42, 123, 456 | 503,808 |
| PPO | baseline, curiosity, memory only, memory + curiosity | 0, 1, 2 | 503,808 |

Equal training interaction counts do not equalise rewards, action spaces, data reuse or compute. Historical proxy scores and exploratory training rewards are distinct from official local scorecard measurements and Kaggle scores. All changes, including unsuccessful improvements, need evidence-backed interpretation.

## Setup and code checks

Use a separate Python 3.12 environment for portable CPU checks. Keep the verified Linux recovery environment unchanged when continuing historical training.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Tests cover action masks, frame history, resets, scoring, device handling, replay/optimizer/RNG restoration and interrupted queues. CUDA checks skip without a GPU; historical checkpoint/delivery checks skip explicitly when their external fixtures are absent. Portable tests use public test fixtures rather than reserved games. On Windows, a short temporary path such as `--basetemp .pt` avoids long-path failures.

Runtime/package evidence is recorded with the experiments. Root requirements support portable setup; they are not a replacement for the exact cu126 recovery environment described in [README_CU126.md](README_CU126.md).

## Completed-model evaluation

Download the verified DQN artifact from the location in the registry, then materialize weights without training:

```bash
python -m Experiments.evaluate_dqn_503808 --materialize-from /path/to/DQN_503808_CONTINUED_20261026T040955Z_0cff534c.zip --steps 450000 503808 --validate-only
```

Game caches are external. Follow [artifact access](docs/ARTIFACT_ACCESS.md) to obtain only the declared exposed games through the official toolkit. Then evaluate all twelve final models on ft09/vc33/cd82, evaluation seeds 0/1/2 and 1,000 selected actions per game, plus a matched random reference:

```bash
python -m Experiments.evaluate_dqn_503808 --device cpu --out results/dqn_extended_503808/development_final_unique
```

Use CUDA in the verified Linux environment, or add `--steps 450000 503808` for the paired-budget comparison. Always use a new output folder. This path freezes weights and performs no learning. See the report for the exact evaluation contract and output files.

Root `evaluate.py` is the common evaluator for all policies:

- `agents.dqn_agent:DQNPolicy` loads frozen DQN weights.
- `agents.ppo_agent:PPOPolicy` loads frozen seven-action MaskablePPO weights.
- `arc_env.gym_adapter.ArcGymEnv` defines the shared action/reset/scoring contract.
- `agents.kaggle_bridge.PolicyBridge` is a competition integration helper; actual Kaggle execution and submission evidence are separate requirements.

Extended DQN training uses `DQN/extended_train.py` and the shared environment. Original native DQN and PPO scripts retain their historical contracts. PPO legacy training runs from `PPO/`, where its `arc_env.py` differs from the root package. Read the relevant protocol before reproducing training.

## Evidence and artifact policy

Keep source, configs, compact tables, figures and reports in Git. Full DQN recovery/replay archives, raw histories, temporary delivery packages, virtual environments, credentials and downloaded game rules remain external or Git-ignored. The checkpoint registries carry exact hashes and source references.

Original V0/V1/V2 files and historical results remain preserved. The DQN migration from T4/cu130 to RTX3050/cu126 was authorised and documented; historical cross-hardware bitwise equivalence failed. Current training audit does not claim another environment-restoration run.

Reserved sb26/sk48/su15 games remain unconfirmed and untested. Old split labels in `Configs/eval.yaml` are historical and do not establish unseen evaluation. Current development evaluation uses `Configs/dqn_extended_development.yaml`; final-evaluation proposals remain unapproved proposals.

Earlier planning/audit documents under `docs/` retain their original dates and can contain superseded recommendations. Use this README, [submission status](SUBMISSION_STATUS.md) and the current registries as entry points. No new training, reserved-game evaluation, Kaggle submission or main-branch merge is performed by setup or testing.
