# Task 1 DQN V2 results

V2 combines four decision-time observations with V1 count-based observation novelty. Each grid has 16 color channels; frame-major concatenation gives 64 input channels. Reset repeats the first grid four times. Each environment step advances history exactly once, including exploratory actions. Replay stores independent uint8 histories before reset. Terminal targets do not bootstrap. Evaluation reconstructs history using checkpoint config; older checkpoints default to one frame.

## Validation

Passed network shapes [1,64,64,64] -> [1,4102] and legacy [1,16,64,64] -> [1,4102]; distinct color channel ordering; replay copy isolation; terminal-mask learning; history reset; loading all three existing V1 checkpoints. The 60-step smoke test used seed42, episode-limit30, batch8, warmup10, train-every5 and had 11 gradient updates in 4.39 seconds. Its files use the suffix `_smoke60`.

## Matched experiment

All runs: ls20-9607627b, 3000 selected actions, episode-limit300, batch8, warmup64, train-every5, lr0.0001, gamma0.99, epsilon1.0->0.1 over2000 steps, novelty beta0.01; default replay capacity3000 and target update interval250. num_frames=4. Saved V1 configs were checked field by field. CPU runs use OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1; runtimes are not directly comparable with earlier runs. Each frozen evaluation uses seed1001, epsilon0, max-actions200, and zero gradient updates.

| Seed | Train max levels | Gradient updates | Eval levels | Eval actions | Final state | Eval action counts |
|---|---:|---:|---:|---:|---|---|
| 42 | 0 | 588 | 0 | 129 | GAME_OVER | {'ACTION2': 38, 'ACTION3': 91} |
| 123 | 0 | 588 | 0 | 129 | GAME_OVER | {'ACTION2': 129} |
| 456 | 0 | 588 | 0 | 129 | GAME_OVER | {'ACTION4': 129} |

## Reset semantics and comparability

The installed LocalEnvironmentWrapper.reset() dispatches GameAction.RESET. BaseGame.handle_reset() uses a level reset after an action, including GAME_OVER; a full reset occurs when the current-level action count is zero or the game is WIN. set_level() clears that counter. ONLY_RESET_LEVELS=true changes this behavior (it was not enabled for these runs). ls20 has no reset override.

Verified in a separate, discarded offline diagnostic: initial reset full_reset=True, levels0; reset after one action full_reset=False, levels0. After a simulated level advance, an immediate reset gave levels1->0/full_reset=True; after one further action in the new level, reset gave levels1->1/full_reset=False. Simulated progress is only a reset diagnostic, never a model result.

Consequently, logged training episodes are level attempts rather than guaranteed independent games. Training maximum levels can accumulate across resets; the interaction budget counts selected actions and excludes controller RESET calls. The existing episode-limit reset/truncation behavior, novelty counting across resets and transition bootstrapping at nonterminal time limits were retained for V0/V1 comparability. These limits also apply to interpreting V1 seed123 reaching one level during exploratory training. Training novelty/level events do not establish frozen-policy improvement or generalization. Only one game and one frozen evaluation seed are tested.

Existing V0/V1 evaluation summaries all report zero levels. Local scorecard values are local diagnostics, not official Kaggle scores.

## Exact commands and outputs

PowerShell environment for runs: `$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"`.

```
.\.venv312\Scripts\python.exe -u -m DQN.train --variant v2 --game ls20 --seed 42 --steps 3000 --episode-limit 300 --batch-size 8 --warmup 64 --train-every 5 --lr 0.0001 --gamma 0.99 --epsilon-start 1 --epsilon-end 0.1 --epsilon-decay 2000 --novelty-beta 0.01
.\.venv312\Scripts\python.exe -u -m DQN.evaluate --checkpoint DQN/checkpoints/dqn_v2_ls20_seed42.pt --game ls20 --seed 1001 --max-actions 200
```

Checkpoint: `DQN/checkpoints/dqn_v2_ls20_seed42.pt`
Training CSV/config/console: `Logs/DQN/dqn_v2_ls20_seed42.csv`, `dqn_v2_ls20_seed42_config.json`, `dqn_v2_ls20_seed42_console.txt`
Evaluation trajectory/summary/console: `Evaluations/DQN/dqn_v2_ls20_seed42_eval_ls20_seed1001.csv`, `.json`, `Evaluations/DQN/dqn_v2_ls20_seed42_console.txt`

```
=== TRAINING SUMMARY ===
Interactions: 3000
Gradient updates: 588
Max levels completed: 0
Runtime: 236.75 seconds
Log: Logs\DQN\dqn_v2_ls20_seed42.csv
Checkpoint: DQN\checkpoints\dqn_v2_ls20_seed42.pt
```

```
{
  "checkpoint": "DQN\\checkpoints\\dqn_v2_ls20_seed42.pt",
  "game": "ls20-9607627b",
  "evaluation_seed": 1001,
  "actions_used": 129,
  "max_actions": 200,
  "levels_completed": 0,
  "total_levels": 7,
  "success": false,
  "final_state": "GAME_OVER",
  "runtime_seconds": 2.816,
  "local_scorecard_score": 0.0,
  "gradient_updates_during_eval": 0
}
```

```
.\.venv312\Scripts\python.exe -u -m DQN.train --variant v2 --game ls20 --seed 123 --steps 3000 --episode-limit 300 --batch-size 8 --warmup 64 --train-every 5 --lr 0.0001 --gamma 0.99 --epsilon-start 1 --epsilon-end 0.1 --epsilon-decay 2000 --novelty-beta 0.01
.\.venv312\Scripts\python.exe -u -m DQN.evaluate --checkpoint DQN/checkpoints/dqn_v2_ls20_seed123.pt --game ls20 --seed 1001 --max-actions 200
```

Checkpoint: `DQN/checkpoints/dqn_v2_ls20_seed123.pt`
Training CSV/config/console: `Logs/DQN/dqn_v2_ls20_seed123.csv`, `dqn_v2_ls20_seed123_config.json`, `dqn_v2_ls20_seed123_console.txt`
Evaluation trajectory/summary/console: `Evaluations/DQN/dqn_v2_ls20_seed123_eval_ls20_seed1001.csv`, `.json`, `Evaluations/DQN/dqn_v2_ls20_seed123_console.txt`

```
=== TRAINING SUMMARY ===
Interactions: 3000
Gradient updates: 588
Max levels completed: 0
Runtime: 254.15 seconds
Log: Logs\DQN\dqn_v2_ls20_seed123.csv
Checkpoint: DQN\checkpoints\dqn_v2_ls20_seed123.pt
```

```
{
  "checkpoint": "DQN\\checkpoints\\dqn_v2_ls20_seed123.pt",
  "game": "ls20-9607627b",
  "evaluation_seed": 1001,
  "actions_used": 129,
  "max_actions": 200,
  "levels_completed": 0,
  "total_levels": 7,
  "success": false,
  "final_state": "GAME_OVER",
  "runtime_seconds": 1.693,
  "local_scorecard_score": 0.0,
  "gradient_updates_during_eval": 0
}
```

```
.\.venv312\Scripts\python.exe -u -m DQN.train --variant v2 --game ls20 --seed 456 --steps 3000 --episode-limit 300 --batch-size 8 --warmup 64 --train-every 5 --lr 0.0001 --gamma 0.99 --epsilon-start 1 --epsilon-end 0.1 --epsilon-decay 2000 --novelty-beta 0.01
.\.venv312\Scripts\python.exe -u -m DQN.evaluate --checkpoint DQN/checkpoints/dqn_v2_ls20_seed456.pt --game ls20 --seed 1001 --max-actions 200
```

Checkpoint: `DQN/checkpoints/dqn_v2_ls20_seed456.pt`
Training CSV/config/console: `Logs/DQN/dqn_v2_ls20_seed456.csv`, `dqn_v2_ls20_seed456_config.json`, `dqn_v2_ls20_seed456_console.txt`
Evaluation trajectory/summary/console: `Evaluations/DQN/dqn_v2_ls20_seed456_eval_ls20_seed1001.csv`, `.json`, `Evaluations/DQN/dqn_v2_ls20_seed456_console.txt`

```
=== TRAINING SUMMARY ===
Interactions: 3000
Gradient updates: 588
Max levels completed: 0
Runtime: 178.81 seconds
Log: Logs\DQN\dqn_v2_ls20_seed456.csv
Checkpoint: DQN\checkpoints\dqn_v2_ls20_seed456.pt
```

```
{
  "checkpoint": "DQN\\checkpoints\\dqn_v2_ls20_seed456.pt",
  "game": "ls20-9607627b",
  "evaluation_seed": 1001,
  "actions_used": 129,
  "max_actions": 200,
  "levels_completed": 0,
  "total_levels": 7,
  "success": false,
  "final_state": "GAME_OVER",
  "runtime_seconds": 1.333,
  "local_scorecard_score": 0.0,
  "gradient_updates_during_eval": 0
}
```
