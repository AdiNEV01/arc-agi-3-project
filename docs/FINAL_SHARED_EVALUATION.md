# Final shared DQN/PPO development evaluation

`Experiments/evaluate_final_shared.py` uses the existing root Task 3 evaluator for both algorithms and the matched random reference. `Configs/final_shared_checkpoints.json` fixes all twelve DQN and twelve PPO models at their final 503,808 training interactions. Earlier 450K snapshots are excluded from this comparison.

## Fixed protocol

`Configs/final_shared_development.yaml` contains only ft09-0d8bbf25, vc33-5430563c and cd82-fb555c5d. Each model is tested with evaluation seeds 0, 1, 2, the same per-game seed mapping, a 1,000-selected-action cap, one scorecard/session, legal-action masks, level resets after GAME_OVER and early stopping on WIN. Scoring is the official local toolkit RHAE, cross-checked by the shared evaluator. Reset actions enter official scored-action accounting separately from the selected-action cap.

DQN variants are V0, novelty V1, history-only and V2, training seeds 42, 123, 456. PPO variants are baseline, curiosity, memory-only and memory+curiosity, training seeds 0, 1, 2. This is 216 learned-policy game/seed runs and nine random-reference runs. Evaluation seeds are repeated measurements, not extra independently trained models.

The 64x64 click representation retains both bridges' established semantics: DQN has a learned pixel-action head, while PPO predicts one of seven action types and samples a legal pixel with its per-game seeded RNG. This difference remains part of the comparison; equal interaction counts do not erase action/reward/data-reuse differences.

## Execution

Obtain the registered models using `docs/ARTIFACT_ACCESS.md`, and the exact exposed offline game assets through the toolkit. A private standalone Linux bundle additionally includes the weights, three offline games and licensed portable PPO dependencies; it needs no clone or pip installation and remains outside regular Git history.

```bash
python -m Experiments.evaluate_final_shared --device cpu --check-only
python -m Experiments.evaluate_final_shared --device cpu
```

For the verified Linux environment use `--device cuda` with PyTorch 2.11.0+cu126. There is no silent CPU fallback. Explicit deterministic inference settings and package/source/protocol hashes are recorded. On Windows, choose a short output directory using `--out` if the default path exceeds the platform's path-length limit.

The default launch checks all 24 model files and metadata, performs real inference, then runs a separate smoke on V0/V2 DQN and baseline/memory+curiosity PPO plus random. That smoke uses seed 0, ten actions per game and the same three games: twelve learned-policy runs plus three random runs. Only a passing smoke permits the full evaluation to start. `--smoke-only` stops after this diagnostic and cannot create a formal-completion claim.

Weights have gradients disabled. Learning and optimizer-step calls raise an error. In-memory weights, PPO optimizer state and checkpoint-file hashes are checked unchanged. This does not reconstruct or continue training; frame history and seeded click RNG remain the documented evaluation memory.

## Outputs and status

Each launch uses a new results folder. `smoke/` and `full/` are separate. The full folder contains 216 labelled learned-policy records, nine random records, `variant_summary.csv`, training-seed aggregates and per-model trajectories/scorecard/runtime checks. The parent contains runtime/source hashes, loading checks and a completion receipt. Only a completed full run produces `formal_evaluation: true`. An evidence ZIP is generated and printed automatically.

Do not pool partial or smoke records with formal results. Interrupted output remains preserved; there is no automatic evaluation resume in this launcher. Reserved sb26, sk48 and su15 are absent and must remain untouched. A successful local scorecard evaluation is not Kaggle submission evidence.
