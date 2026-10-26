# Extended DQN: training completion and evidence

All twelve models completed 503,808 selected training interactions: V0, novelty-only V1, history-only and history-plus-novelty V2, each with training seeds 42, 123 and 456. They continued from 450,000, adding 53,808 interactions per model and 645,696 across the matrix. Each model used 167,936 actions on each of ls20, ar25 and bp35, in deterministic round robin.

## What the training logs show

The table sums nonnegative level-delta task reward during exploratory training across the three training games. These are cumulative events across the history, not official RHAE scores, distinct solved games or frozen-policy evaluation returns. Seed order is 42, 123, 456.

| Variant | 450K task reward by seed | 503,808 task reward by seed | Extra task reward | Full-game wins at 503,808 |
| --- | --- | --- | ---: | ---: |
| V0 | 3, 3, 1 | 3, 3, 1 | 0 | 0 |
| V1 | 1, 2, 3 | 1, 2, 3 | 0 | 0 |
| History only | 3, 2, 3 | 3, 3, 3 | 1 | 0 |
| V2 | 3, 1, 3 | 3, 1, 3 | 0 | 0 |

Matrix-wide task reward rose from 28 to 29. The only added level-delta event was history-only seed 123 on ar25. No training task reward was recorded on bp35. This shows limited additional task progress during the extension; it does not establish the ranking of the final frozen policies. Novelty accumulates substantially in V1/V2 and must remain separate from task achievement. Loss reduction and intrinsic return are not evidence of puzzle success.

Figures under `Analysis/DQN/extended_503808/figures/` show individual training seeds and the 450K boundary. Curves use 5,000-action bin endpoints; exact event timing remains in raw logs. Keep the three trained seeds as the independent replicate unit. Repeated evaluation seeds do not increase training n.

## Verification and experiment history

The complete archive passed 360 member CRC checks. All twelve final models loaded safely on CPU with finite online/target parameters and finite outputs. Every one of the 6,045,696 logged actions passed sequence, round-robin, per-game counter, epsilon, reward eligibility and no-truncation checks. There are 100,562 finite gradient updates per model. All 84 milestone hashes, 72 recovery payload hashes and 18 recorded training/queue source hashes matched.

All 72 frozen milestones through 450K retain their source hashes. Every continued history's 450K prefix matches its original summary. Only the protocol identifier, action cap and milestone list changed; epsilon remained 0.1 during the extension and learning hyperparameters were retained. Source/derived checkpoint receipts and migration/restoration records accompany the archive. Original 450K full recovery payloads remain in the separate backup, so a fresh bitwise comparison against those full states was not repeated here.

The experiment moved from Colab T4/PyTorch 2.11.0+cu130 to RTX 3050/PyTorch 2.11.0+cu126 after quota and driver constraints. Historical cross-environment bitwise equality failed, and the user authorised a documented numerical migration. The supplied record reports a successful same-GPU split-restoration check. Seed 42 V0, V1 and history-only retained cu130-labelled 450K endpoints and migrated for the extension. V2 seed 42 migrated from its 400K checkpoint; the remaining seeds trained under cu126. The school driver was not changed.

This Windows audit used Python 3.12/PyTorch 2.14.1+cpu to inspect frozen models. It checksum-verified complete 500K and 503,808 recovery files, but did not reconstruct environments or execute another training continuation. It is not a new CUDA recovery test or proof of cross-hardware deterministic equivalence.

Recorded extension throughput was approximately 313–325 actions/s for one-frame models and 213–215 actions/s for four-frame models, including persistence within the measured loop. These 53,808-action measurements exclude prior training, setup and initial restoration. They are not whole-experiment runtimes or a controlled CPU/GPU comparison against the earlier 10K/410s run.

Matching PPO's 503,808 training interactions aligns the interaction budget. It does not equalise rewards, observation history, action representation, data reuse or hardware history. PPO's separate development validation actions remain outside its training count. Comparative conclusions require common frozen evaluation.

## Repository artifacts and reproduction

`Configs/dqn_503808_checkpoints.json` registers twelve final models and twelve paired 450K snapshots. `Configs/dqn_503808_training.json` contains the extended configuration. Small tables and audits live under `results/dqn_extended_503808/`; figures and plot source live under `Analysis/DQN/extended_503808/`. Historical 3K/10K reports remain separate.

The full ZIP has SHA256 `52756cb244b7b0ca1fb62b06d14d52d3ad9481a0063c69ce06a60fdc611e0bcc`. Preserve full model/replay backups in Drive or other artifact storage. Git ignores local materialized binaries and raw logs under `artifacts/dqn_extended/52756cb244b7/`. Add a verified, accessible archive URL to the registry before publication; no new upload or sharing change was made here. Commit reviewed source, configs, small tables, audits, figures and this report deliberately. Keep raw result folders and cached game assets outside Git.

The team integration branch is `integration/2026-10-24`. Pull the published update before following these commands. Model/recovery binaries are distributed separately through the verified artifact archive; Git does not transfer ignored local output.

In an environment containing this source and project dependencies, restore endpoint weights from the archive and verify all 24 endpoints without evaluating:

```bash
python -m Experiments.evaluate_dqn_503808 --materialize-from /path/to/DQN_503808_CONTINUED_20261026T040955Z_0cff534c.zip --steps 450000 503808 --validate-only
```

Existing matching files are reused; differing files are never overwritten. This materializes 24 endpoint weights and 12 metadata sidecars in ignored storage. It does not restore replay or optimizer state. In this Windows checkout the files are already materialized; use the existing dependency environment.

## Next experiment: frozen development evaluation

Formal development evaluation has not run. A six-game integration smoke using final V0/V2 seed 42 models at ten actions/game passed scorecard, reset, legality, score-recomputation and unchanged-weight checks. Its scores are not substitutes for the full protocol.

The prepared launcher evaluates frozen models greedily without learning on already-exposed ft09, vc33 and cd82, evaluation seeds 0, 1, 2 and 1,000 selected actions/game, plus one matched random matrix. It preserves click_grid 64, the common scorecard/reset contract and offline operation. The final-only matrix is 108 model/game/seed runs plus nine random runs:

```bash
python -m Experiments.evaluate_dqn_503808 --device cpu --out results/dqn_extended_503808/development_final_new_run
```

Use `--device cuda` in the existing verified Linux cu126 environment after transferring this source and materialized weights or the archive. CUDA unavailability produces an error without silent fallback. For the planned paired-budget evaluation, add `--steps 450000 503808`; this runs 216 learned-policy game/seed combinations plus nine random runs. Always choose a new output directory.

Outputs include per-run official/recomputed scores, trajectories, `records.json`, `training_seed_aggregate.json` and evaluation validation. These will support assessment of final policy performance and the extension's effect. Reserved sb26, sk48 and su15 remain unconfirmed and untouched. Held-out and PPO performance claims require their own completed common-protocol evaluation.
