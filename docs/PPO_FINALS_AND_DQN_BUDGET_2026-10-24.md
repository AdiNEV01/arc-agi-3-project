# PPO finals verified and DQN training decision

24 October 2026, Africa/Johannesburg. Work stayed on `integration/2026-10-24`.
No push, main merge, full 24-model evaluation or untouched-heldout run was performed.

## Decision

All twelve larger PPO finals are valid and usable through the existing adapter.
DQN can now train through Task 3's shared environment. A controlled fresh 10K V0
pilot completed, but no task-performance improvement appeared in development tests.

**Finish-today recommendation:** retain the existing 3,000-action, four-configuration,
three-training-seed DQN ablation as the primary DQN panel. Report the new 10K V0
pilot separately as budget sensitivity. Keep every primary DQN configuration at
one budget; do not replace only its baseline with the 10K pilot. Use all twelve
PPO final models, with unequal training budgets explicitly stated. This is a
comparison of complete implemented systems under common frozen evaluation, not
proof that one learning algorithm is intrinsically superior.

If the group chooses an additional complete DQN training matrix, 10K/model is
the only budget I would consider tonight: about three CPU hours, before evaluation
and report work. 30K/75K/450K matrices exceed today's practical compute window.
The zero-reward V0 pilot does not prove that more training, novelty or memory
cannot help. It provides little evidence for prioritising a three-hour rerun now.

## 1. PPO verification

Fetched `origin/PPO` with `--filter=blob:none --no-tags`, pinning source commit
`7d15ec69dd9e619344fa2f37092dbb4bca2f2957`. Materialized only the twelve finals;
no intermediate or best-model checkpoint files were checked out. All final blobs
were already present in the local Git object database from the earlier fetch,
so this verification required no additional large model-blob downloads.

Paths follow `PPO/logs/{configuration}_seed{0,1,2}/final_model.zip`.
They remain ignored by Git in this integration checkout. A pinned manifest with
all paths, source Git blobs, SHA-256 values, sizes and seeds is at
`Configs/ppo_final_checkpoints.json`. Rehydration/verification is reproducible with
`python -m Experiments.verify_ppo_finals --out results/new_unique_verification`.
No wholesale PPO merge or `.gitignore` relaxation was needed.

| Configuration | Seeds | Actual transitions each | Channels x height x width | Policy parameters | ZIP bytes |
|---|---|---:|---|---:|---:|
| baseline | 0,1,2 | 503,808 | 16 x 64 x 64 | 631,464 | 9,540,660 |
| curiosity | 0,1,2 | 503,808 | 16 x 64 x 64 | 631,464 | 9,540,661 |
| memory_only | 0,1,2 | 503,808 | 64 x 64 x 64 | 729,768 | 16,487,475-16,487,479 |
| memory_curiosity | 0,1,2 | 503,808 | 64 x 64 x 64 | 729,768 | 16,487,484 |

Every ZIP passed CRC checks and loaded on CPU through `PPOPolicy`; all twelve
policy weight hashes are distinct; every policy tensor is finite; weights were
unchanged after compatibility checks. These are ordinary Git ZIP blobs, not Git
LFS pointers; the source tree has no `.gitattributes` or `.lfsconfig`.

All use MaskableActorCriticCnnPolicy with NatureCNN and a seven-action output.
Memory models use four frame-major one-hot grids. Shared API smoke used six
selected actions on the synthetic toolkit fixture and six on development ls20
for each final: 24 tiny compatibility runs, not a final performance matrix.
All masks/reset/one-scorecard invariants passed. Valid/loading does not establish
that the policies perform well; final common-protocol scores are still pending.

Shared PPO settings verified from saved state: learning rate 0.0003; gamma 0.99;
GAE lambda 0.95; clip 0.2; entropy coefficient 0.01; batch64; 2,048 steps per
environment; 3 environments; 10 epochs; normalize_images=False. All have 820
recorded training epochs and **78,720 actual Adam steps**, checked in saved
optimizer state. Full architectures/configuration/resource records are in
`results/ppo_finals_verified_20261024_complete/verification.json`.

## 2. What the action budgets mean

A DQN selected action and one PPO environment transition are comparable at the
level of one dispatched game action. They are not equivalent amounts of learning
or per-game exposure. Controller RESETs and animation frames are separate.

- PPO's 503,808 is aggregated across ls20/ar25/bp35, not per game. Each received
  167,936 training transitions. 82 complete rollouts x 2,048 x 3 = 503,808;
  the requested 500,000 was rounded up. The 450K snapshots were intermediate.
- Each PPO model also used **3,000 cd82 validation actions**, from six callbacks
  with five episodes each; these are excluded from `num_timesteps`. RESET calls
  are excluded too. Training-time validation is development exposure and a cost.
- PPO's reported resource FPS uses requested 500K rather than actual 503,808.
  Its recorded aggregate training runtime is about 12.2 GPU hours across twelve
  runs on an RTX 4050 laptop. This is not comparable to local CPU latency.
- Historical DQN used only ls20 and 3,000 selected actions, with 588 replay
  updates. The new 10K pilot used 1,988 updates, batch8, one update every five
  actions after warmup64. PPO makes ten data passes per rollout; the saved 820
  counter is epochs, not the number of optimizer minibatches.
- PPO final / old DQN total training actions = 167.936; PPO final / pilot = 50.3808.
  Per-ls20 exposure ratios are approximately 56 and 16.8 respectively. DQN also
  lacks PPO's additional two training games and validation interactions.
- Training contracts still differ: DQN level-delta reward, PPO WIN-only reward;
  novelty beta 0.01 versus 0.1, with different no-op/history counting; historical
  DQN episode cap300 versus uncapped PPO. Common frozen scoring does not erase
  those training differences. Raw action matching alone cannot isolate algorithm effects.

## 3. Shared DQN implementation and pilot

Original DQN trained directly on the native interface. Added an explicit
`--backend shared` path through `ArcGymEnv`; native remains the default.
`DQN/shared_environment.py` exposes only permitted grid/info to the old learning
loop. It verifies all 4,102 native/shared action indices are identical at a
64x64 click lattice. RESET is never learned as an action.

One/four-frame replay history and novelty are retained. Shared padding16 is
encoded as all-zero colour channels, matching the frozen adapter; network shapes
and checkpoint parameter names stay compatible. No new algorithm was added.
The shared task reward is checked against the level-count delta. WIN/GAME_OVER
are terminals; optional time-limit truncation remains separate and bootstraps.
The pilot disabled the arbitrary episode cap and used level resets only after
terminals, retaining progress exactly as the shared competition-style controller
permits. Training can start new games after WIN; evaluation stops at WIN.

Pilot: fresh seed42, ls20, 10K actions; CPU with OMP/MKL threads1; lr0.0001;
gamma0.99; batch8; replay3000; target sync250 updates; warmup64; train every5;
epsilon1->0.1 over2K, then0.1. Existing 3K model files were not resumed/overwritten.
A new shared 3K prefix snapshot was saved in the same run. Checkpoints are frozen
snapshots; exact resumption is not claimed because replay/novelty/RNG state is not saved.

Command:

```powershell
python -m DQN.train --backend shared --operation-mode offline --reset-mode level `
 --variant v0 --game ls20 --seed 42 --steps 10000 --episode-limit 0 `
 --batch-size 8 --warmup 64 --train-every 5 --lr 0.0001 --gamma 0.99 `
 --epsilon-start 1 --epsilon-end 0.1 --epsilon-decay 2000 `
 --checkpoint-steps 3000 --run-suffix shared10k_20261024
```

Measured: **410.18 seconds (6.84 min)**, 10,000 official selected actions,
72 controller resets, 1,988 gradient updates, zero artificial truncations,
finite losses, zero task reward and zero levels. Prefix snapshot: 3K/588 updates.
Both new files use `DQN/checkpoints/dqn_v0_ls20_seed42_shared10k_20261024`, with
`_step3000.pt` for the prefix and `.pt` for the final. Training CSV and configuration
have the corresponding stem in `Logs/DQN`.

## 4. Development result and runtime estimate

Evaluated old native3K, new shared3K and shared10K V0, all seed42, on already-exposed
ls20 and ft09; three evaluation seeds0/1/2; 1,000 actions/game; identical full-pixel
mask, level-reset controller and official local scorer. 18 runs/18,000 actions
completed in about 150 seconds of evaluation runtime. All scored zero and completed
zero levels; all one-scorecard/no-full-reset/no-illegal-action checks passed.

The old and new 3K policies chose identical action sequences on both games.
Their parameters differ slightly numerically, so bitwise training identity is
not claimed. The 10K model changed behavior: ls20 seed0 used ACTION1 538 times,
ACTION3 455 and ACTION2 seven times (action entropy 0.727 nats), versus repeated
ACTION1 at3K. ft09 changed its repeated click from (17,47) to (15,4), with zero
reward in both cases. A policy change is not evidence of improved goal-seeking.
The cached games and deterministic argmax can yield identical results across
evaluation seeds; these are not three independent training runs.

| Actions/model | V0 | V1 estimate | History-only estimate | V2 estimate | All four x three seeds, serial |
|---|---:|---:|---:|---:|---:|
| 10,000 | 6.8 min | 9.3 min | 24.0 min | 20.7 min | 3.04 h |
| 30,000 | 20.5 min | 28.0 min | 72.0 min | 62.0 min | 9.13 h |
| 75,000 | 51.3 min | 69.9 min | 180.1 min | 155.1 min | 22.82 h |
| 450,000 | 307.6 min | 419.7 min | 1080.6 min | 930.7 min | 136.93 h |

Only V0 at10K is measured under the new shared path. Higher budgets are linear
extrapolations; other variants scale using historical three-seed median runtime
ratios. These are approximate planning figures, not new measurements or a promise
of throughput. Reserve overhead; hardware load and the shared padding path can
change them. Do not parallelize CPU runs indiscriminately: prior thread oversubscription
made runtime substantially worse.

## 5. Exact proposed final selection and protocol

`Configs/final_checkpoint_selection_proposal.json` enumerates **all 24 exact paths
and hashes**: twelve historical DQN 3K models (V0/V1/history-only/V2 x seeds42/123/456)
and twelve PPO 503,808-step finals (baseline/curiosity/memory_only/memory_curiosity
x seeds0/1/2). The V0 10K pilot is supplementary, not a replacement within the
primary equal-budget DQN ablation. No dev-score cherry-picking or best-model ZIP
selection is proposed. If a new complete DQN budget is approved and all twelve
runs finish, replace the complete DQN panel consistently.

`Configs/final_evaluation_proposal.yaml` remains **unapproved**:

- Development: ls20-9607627b and ft09-0d8bbf25, both already exposed.
- Candidate untouched final set: sb26-7fbdac44, sk48-d8078629, su15-1944f8ab,
  only after every teammate confirms no previous use and the user approves.
  None was made, downloaded or played here.
- Previously exposed bp35/vc33/g50t/ka59 are excluded from untouched claims;
  ls20 is also training-exposed, not evidence of generalization.
- Common environment actions: 6 simple types + all4,096 pixel clicks. DQN chooses
  pixels by Q values; PPO chooses among7 types with seeded uniform click pixels.
  Both use the same external legal actions; coordinate policy capacity still differs.
- Frozen argmax/epsilon0; no gradient updates; memory cleared between games and
  on controller resets; per-game click RNG reset from evaluation seed.
- Evaluation seeds0/1/2, same game seed mapping; independent training seeds kept
  separate. 1,000 selected game actions, RESET excluded from that budget but
  included by official scorecard accounting. Continue after GAME_OVER via level
  reset; stop at WIN or budget. No wall-clock truncation for the final comparison.
- Toolkit official **local** score, game/level completion, all failures, actions,
  resets, runtime and uncertainty. Validate scorecard_runs=1, no full resets after
  start and zero illegal actions; investigate failures before aggregating.
- Rerun random with the same full-pixel representation. The old8x8 random results,
  native200-action DQN evaluations and PPO proxy scores cannot be pooled here.

With two dev plus three confirmed heldout games, the proposed matrix is
24 x 5 x 3 = 360 model-game runs, plus matched random references. It has not run.
Actual final runtime will depend on game speed; reserve at least an hour plus
packaging time, and measure completion rather than promising a fixed finish.

## 6. Required next work and reruns

1. Approve the exact checkpoint/protocol proposal and confirm the three genuinely
   untouched candidates. If matching training budgets is required instead, first
   predeclare common training games, reward/reset rules, interaction accounting,
   validation and compute limits. Increasing DQN actions alone does not achieve this.
2. Run the approved common frozen matrix and random reference once, saving
   trajectories. Preserve the 10K V0 diagnostic as additional budget evidence.
3. Replace historical proxy/custom-evaluator numbers in the main comparison with
   these official local scores. Keep existing stage/ablation training curves with
   honest metric labels: intrinsic training return is not the benchmark score.
   Download intermediate models only if a specific additional official checkpoint
   curve is required; choose its times before viewing heldout outcomes.
4. Finalize the joint report with individual seeds/uncertainty, zero-score failures,
   different training budgets/games/rewards/architectures and hardware. Limit claims
   to implemented systems and per-algorithm ablations. An algorithm-only matched
   training claim requires matched reruns of both systems under an agreed design.
5. Package code/notebook and the appropriate checkpoint bundle; verify official
   starter execution, required report format, competition/Paper Track evidence,
   author contributions and submission contents. Seek approval for any main push/merge.

Tests/evidence: 27 pytest checks pass, all12 finals load, 24 tiny shared checks pass,
18 development V0 evaluations pass scorer invariants. All130 historical DQN artifacts,
all old smoke checkpoint hashes and all12 new final hashes are preserved. Raw model
ZIP/PT files, downloaded game cache, local environments and console logs remain ignored.
