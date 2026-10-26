# Integration audit and same-day execution plan

Audited 24 October 2026, Africa/Johannesburg. Assignment deadline: **27 October
2026, 23:59 SAST**, verified in the local RL_Assignment.pdf. The target is readiness
today. Branch integration and compatibility are complete; the joint report and
external submission evidence are not.

## Actual branch state

`git fetch origin` succeeded before inspection.

| Remote ref | Commit | Role and evidence |
|---|---|---|
| main | af5811f | Earlier PPO implementation. Original local main is 4506980, four commits behind. |
| feature/dqn-task1 | 6e79be6 | Working DQN V0, novelty V1, four-frame novelty V2 and history-only ablation; three training seeds, 3,000 selected actions each; logs, tables, figures and report contribution. |
| PPO | cf7f39f | Newer MaskablePPO, novelty, memory, memory-only ablation CSV, draft article, GIFs and three pilot model ZIPs. Replaces older PPO code on main. |
| task3-interface | ead3f20 | Shared Gymnasium adapter, encoding, policy interface, official local scoring evaluator, split, random baseline and 10 tests. Original checkout is on this branch. |

The original repo has no tracked edits. Its untracked virtual environment, DQN
checkpoints, logs, evaluations and local analysis remain in place. Integration is
a separate clone in this chat's workspace on `integration/2026-10-24`. It contains
all four remote heads in its ancestry. No push or merge into main occurred.

## Contract differences

| Contract | DQN legacy | PPO legacy | Shared path |
|---|---|---|---|
| State | Last 64x64 grid; 16 one-hot channels; history variants use four decision frames | 16-channel one-hot; memory has 64 channels | uint8 64x64 grid, padding 16; bridges rebuild one/four-frame tensors |
| Actions | 6 simple + 4,096 learned pixel clicks | 7 types; ACTION6 coordinates random | 6 simple + click_grid squared positions; default 8x8 |
| Training reward | Level-count delta plus optional count novelty | WIN-only plus optional count novelty | Level-count delta by default; custom reward_fn supported |
| Reset / evaluation stop | Level attempts with episode caps; frozen evaluation stops on first GAME_OVER or 200 actions | Level reset; frozen evaluator stops at first terminal or 1,000 actions | One session, level reset after GAME_OVER, continue to fixed action budget |
| Scoring | Custom native evaluator | Proxy score; predict omits action_masks | Root evaluator, explicit masks, toolkit official local scorecard and recomputed cross-check |

`Configs/integration_smoke.yaml` uses **click_grid 64**. DQN's shared action indices
then exactly equal its native 4,102 indices. PPO keeps its seven-way decision and
uniform pixel clicks, now with a seeded per-game RNG. At click_grid 8, the DQN
click head would be restricted and PPO's click distribution would change.

Bridges clear frame history at game start and GAME_OVER. The competition helper
also clears history on resets without reseeding per-game randomness. Padding maps
to all-zero colour channels. Frozen weights persist; bridges perform no training.

Legacy training scripts remain unchanged. A frozen bridge does not establish
shared-wrapper training. Old DQN results remain evidence for the original
interface and its documented reset/seed limitations. New common evaluations can
compare frozen behavior, with unequal training budgets explicitly disclosed.
Changing training action/reward/reset contracts requires retraining for matched
training claims. Historical DQN and PPO curves are not a controlled comparison.

## Available artifacts and missing evidence

**DQN:** 15 local .pt files, including pilots/smokes, were copied and retained at
the original location. Twelve full runs cover V0/V1/history-only/V2 and training
seeds 42/123/456. All frozen ls20 results completed zero levels; V1 seed123's one
level during exploratory training is not frozen-policy improvement. Checkpoints
are Git-ignored and need an explicit reproducibility bundle.

**PPO:** The CSV contains 288 rows: 72 each for baseline, curiosity, memory-only,
memory+curiosity; seeds 0/1/2; checkpoints 75K/150K/225K/300K/375K/450K;
dc22/ft09/g50t/ka59. Two rows complete one ft09 level but end in GAME_OVER.
The rhae column is a heuristic, not the official score; aggregate episode length
cannot recover correct per-level scoring. Mask omission affects its decisions.

Only three PPO ZIPs are available: baseline seed1, curiosity seed0 and
memory+curiosity seed0. Saved metadata confirms **12,288 timesteps and 20 updates
each**. No memory-only model or complete training-seed set exists in the repo.
The user confirmed the repo is the complete available set; no additional Drive
archive exists. The larger-run CSV is historical supplied evidence whose models
cannot be reproduced here. Never relabel the pilots as 500K-trained models.

**Task 3:** The existing random baseline uses an 8x8 lattice. Reevaluate random
for any new 64x64 comparison rather than pooling incompatible runs.

**Submission:** No finished joint eight-page RLC-template report, submission
notebook or competition/Paper Track evidence was found in the repo. PPO's draft
uses article with custom margins. Joining existing sections alone will not meet
the required format and main-content page limit.

## Split conflicts

| Game | Conflict |
|---|---|
| ls20 | Listed as dev; DQN and PPO trained on it. |
| bp35 | Listed as heldout; PPO training list includes it. |
| vc33 | Listed as heldout; DQN supplementary tests already used it. |
| g50t, ka59 | Listed as heldout; PPO checkpoint evaluation used them. |

dc22/ft09 and Task 3 probe games are development-exposed too. Disclose usage
rather than renaming a split to imply no exposure. sb26/sk48/su15 are candidate
untouched games based on available records, pending every teammate's confirmation.
No candidate heldout game was run in this audit. Preserve the historical config;
freeze a separately named final protocol before any final heldout evaluation.

## Integration and verification completed

- Merged all task branches locally, preserving current remote main ancestry.
- Resolved requirements/ignore/probe conflicts; kept the richer Task 3 probe.
- Excluded cached game sources and LaTeX build intermediates from the integrated
  tracked tree, without deleting original files or rewriting historical commits.
  Hidden game source was not inspected to design agents.
- Added frozen DQN/PPO bridges, masking, history/padding and competition-reset support.
- Rejected negative shared indices and made illegal policy actions abort evaluation,
  preventing a loop whose official-action counter could never advance.
- **24 tests pass**, including original DQN/Task 3 tests, DQN full-map decision
  equivalence, PPO native/shared mask ordering, seeded clicks, history/padding,
  terminal resets and real-toolkit scorecard checks.
- **14 final smoke runs pass**: random; DQN V0/V1/V2; PPO baseline/curiosity/
  memory+curiosity, each on cached ls20 and the real-toolkit test fixture. One
  evaluation seed, 140 ls20 actions, 30 fixture actions. All runs had zero illegal
  actions, one scorecard play, no unintended full resets, and official/recomputed
  score agreement. All ls20 runs scored zero. Fixture wins are software checks,
  not benchmark achievements. Model-file hashes were unchanged before/after smoke.

Authoritative evidence: `results/integration_smoke_verified_20261024/verification.json`,
its per-policy trajectories/score files, `results/integration_audit/tests.txt`,
runtime versions, checkpoint metadata and the preservation inventory. The earlier
smoke directory is superseded by the verified run.

## Prioritised plan to finish today (SAST)

Time boxes assume work starts around 15:30; experiment throughput is not guaranteed.

| Priority / window | Must-have outcome | Execution |
|---|---|---|
| 1. 15:30-16:00 | Freeze checkpoint selection, games, full-pixel actions, evaluation budget/resets/seeds. | Group agrees once; confirm exposure; designate report and submission owners. |
| 2. 16:00-17:30 | Official local scores for available three versions of each algorithm plus random. | Root evaluator, same games and action budget. Reuse DQN models. Label unequal training budgets and incomplete PPO training seeds. Three evaluation seeds do not create three training seeds. Include history-only ablation if feasible. |
| 3. Decision by 16:00 | Choose the PPO evidence path. | Fast route: report 12,288-step pilots with explicit limitations. For matched/replicated claims, run only a small predeclared baseline/curiosity/memory/memory-only matrix under one training contract. Cap compute; log actual rollout-rounded interactions. Do not repeat the whole 500K suite. |
| 4. 17:30-20:00 | Joint eight-page report in required template. | Use existing evidence, official new scores, failures, ablations, budgets and honest limitations. Separate pilot reevaluation from unavailable larger-run results. |
| 5. 20:00-21:30 | Starter/notebook packaging check, model bundle and actual submission evidence. | Submission owner verifies Kaggle team/track requirements. Local helper smoke is not a competition submission. |
| 6. 21:30-22:30 | Fresh-run/package review, citations/AI declaration, contributions and page limit. | Group reviews the concrete integration result. Approval is required before pushing or merging into main. |

Stop training at the agreed cap. If unavailable checkpoints cannot be replaced
within it, disclose the limitation; do not substitute historical proxy scores for
official numbers. Submission readiness still depends on the report and actual
external submission evidence. The hard Moodle deadline remains 27 October,
23:59 SAST.

**Optional:** longer training, new novelty representations, learned click policies,
recurrent networks, additional feature engineering, cosmetic figures and a broad
heldout sweep. None is required for an honest negative-result investigation.

## Next evaluation entry point

```powershell
# From this checkout using an installed Python 3.12 environment:
python -m pytest tests DQN/test_task1.py -q
python -m Experiments.integration_smoke --out results/new_unique_smoke_name

# Example dev evaluation: 3,000-step TRAINING checkpoint; three EVALUATION seeds.
# Always use a NEW output folder. Freeze one config for all policies first.
python evaluate.py --config Configs/integration_smoke.yaml `
  --agent agents.dqn_agent:DQNPolicy `
  --agent-kwargs '{"checkpoint":"DQN/checkpoints/dqn_v0_ls20_seed42.pt"}' `
  --agent-version dqn-v0 --checkpoint-step 3000 `
  --games ls20-9607627b --seeds 0 1 2 --budget 1000 --out results/new_dqn_eval

# PPO: --agent agents.ppo_agent:PPOPolicy
# --agent-kwargs '{"checkpoint":"PPO/logs/baseline/ppo_baseline_seed1.zip"}'
# --checkpoint-step 12288. This remains one training seed.
```

The smoke config has a 60-second safety cap. For final action-budget-only runs,
freeze a separate config with `max_seconds_per_game: null`; use the same config
and cap for every agent. Do not evaluate candidate untouched games until all
settings and checkpoint selection are frozen.


1	Agree on the final evaluation settings and available checkpoints (Entire group)
2	Run shared DQN vs PPO vs random evaluations	(Work, with team verification)
3	Generate the definitive results table	(Work)
4	Assemble the eight-page report	(Group, with Work assisting)
5	Check submission files and evidence	(Submission owner)
6	Review and merge integration branch into main	(Group)