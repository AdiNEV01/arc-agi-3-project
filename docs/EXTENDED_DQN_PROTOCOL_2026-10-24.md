# Extended DQN protocol and launch decision

Protocol ID: dqn_extended_v1_20261024. Status: code/pilot preparation; 75K GPU pilots and 12-run matrix have NOT run in this session. Source remains on integration/2026-10-24 with prior uncommitted work preserved. No push or main merge.

## 1. Recommended shared training protocol

Train four DQN variants, each at seeds 42/123/456, through SharedDQNEnvironment -> Task 3 ArcGymEnv -> local arc_agi engine. Use the exact games ls20-9607627b, ar25-0c556536, bp35-0a0ad940. Select one action per game in deterministic round-robin order: 450,000 total =150,000/game, not 450,000/game. Every variant has the same ordering and exposure counts; per-game trajectories naturally differ. Each game has separate observation/history/novelty/controller state. Replay is shared across the three games. A transition always bootstraps within its own game, never into the next scheduled game's observation.

| Setting | Fixed value |
|---|---|
| Total selected game actions | 450,000 per run; RESET counted separately |
| Pilot horizon | 75,000 each for V0 and V2 seed 42; same 450K configuration |
| Observation | last settled grid, one-hot16, padding16 -> zero channels |
| History | one frame V0/V1; four decision-time frames history-only/V2 |
| Actions | six simple actions plus4096 learned click coordinates; legal masking |
| Task reward | nonnegative newly completed level count |
| Novelty | beta0.01/sqrt(count) on latest grid, separate counters per game, no unchanged-grid bonus; only V1/V2 |
| Epsilon | 1.0 ->0.1 linearly over150,000 selected actions, then0.1 |
| Replay | 20,000 raw uint8 transitions, FIFO ring; uniform sampling |
| Learner | batch8; warmup1000; one update every5 actions; Adam1e-4; gamma0.99 |
| Target | copy every250 gradient updates; Huber loss; gradient clip10 |
| Reset | level reset on GAME_OVER; new game after WIN; no artificial episode cap |
| Saved evaluation points | 75K,150K,225K,300K,375K,450K; no best-checkpoint selection |
| Continuation state | every25K and final stop; latest two complete generations retained |
| Reproducibility | seed Python/NumPy/PyTorch/agent RNG; explicit game seed; deterministic torch algorithms; TF32 off; fixed source/assets/versions |

PPO audit: verified pinned source7d15ec69dd9e619344fa2f37092dbb4bca2f2957 and metadata from all twelve saved final ZIPs. Training games were ls20/ar25/bp35; three DummyVecEnv lanes. Each final contains503,808 actions =82 x2048 x3, or167,936/game. Requested500K was rollout-rounded;450K files are intermediate checkpoints. An additional3,000 cd82 validation actions/model are excluded from num_timesteps. The original environment did not explicitly pass a game seed to Arcade.make, and exact historical game file hashes were not recorded. Modern pinned versions cannot prove identical historical game revisions.

The existing PPO models use WIN-only task reward, seven action types with random click coordinates, NatureCNN, and curiosity beta0.1 that can count history/no-op states. DQN uses level-delta reward, learned coordinates, a different CNN and latest-grid novelty beta0.01. PPO has503,808 actions plus validation; DQN's proposed450K is10.68% fewer training actions. The primary scientific claim is a controlled2x2 DQN ablation plus common-evaluator comparison of complete systems. Equal budgets alone do not isolate algorithm effects. If an algorithm-only comparison is essential, retrain PPO on the same shared reward/reset/action/observation protocol; increasing DQN actions cannot repair historical differences. Optional later budget-matched comparison can use verified PPO450K intermediate snapshots, still disclosing reward/action differences.

## 2. Necessary implementation changes

New DQN/extended_train.py leaves the established short benchmark and native trainer intact. It supplies multi-game round-robin training, separate history/novelty per game, compact replay, long-horizon epsilon, deterministic settings, CUDA placement inherited from the tested agent, finite-loss checks, level/action-distribution logging and complete continuation state. Configs/dqn_extended_v1.json declares every training setting. Unique run directories refuse overwrite. Model/target/optimizer state-dict keys and num_frames metadata remain compatible with the frozen DQNPolicy loader.

Experiments/run_dqn_extended.py defaults to TWO pilots only. Matrix mode requires both full 75K CUDA pilot logs, matching source/protocol/runtime hashes, correct counters, finite losses and loadable checkpoints. It never launches the full matrix from the pilot notebook. Validation is mechanical evidence, not a finding that a policy learns.

## 3. Resources and provisional cost

Measured short-run T4 rate:218.05 actions/s on ls20 V0; CPU29.95. At unchanged throughput450K V0 is34.4min and twelve V0-equivalent runs6.88h. The new three-game contract, longer greedy portion, deterministic CUDA kernels, replay compression and Drive copying change that rate. History throughput is not yet measured. Reserve roughly 8-14 T4 training hours as a planning range, based on history runs taking approximately1.5-2.5x the V0 cost and additional overhead; this is not a result or promise. Two pilots may take roughly 15-30min, but initialization/game speed/Drive can increase this. Add development evaluation separately; all six DQN checkpoint curves involve648K evaluation actions plus random references. Budget roughly 2-4 extra evaluation hours provisionally, then replace with actual measurements.

After both pilots, estimate serial training seconds as3*450000*(2/r_V0 +2/r_V2), where rates include checkpoint/persistence time. This approximates V1 by V0 and history-only by V2. The notebook records this assumption and adds25% reserve. Do not infer V2 throughput from V0 alone.

Use one T4 and serial runs. Plan for at least 8GiB available host RAM (12GiB-class VM comfortable); verify actual memory in the notebook. Replay payload is about246MB for one frame or738MB for four, before Python overhead. Encoding/save/load can temporarily duplicate these arrays. T4 memory is checked by nvidia-smi, and peak allocated CUDA memory is captured per run. Two retained uncompressed-equivalent replay generations per run can approach1.5GB for history variants; reserve approximately 25GB persistent space for all 12 runs, logs and snapshots unless pilot compression measurements justify less. Actual compressed sizes are reported by the pilots. No storage quota is assumed from the mounted Drive filesystem.

Colab resources/lifetimes are not guaranteed: https://research.google.com/colaboratory/faq.html . Split work across resumable sessions rather than assuming uninterrupted availability. No anti-idle or quota workarounds are used.

## 4. Checkpoint and genuine continuation strategy

Frozen step*.pt files contain model, target, optimizer, step/update counters and configuration, for Task 3 evaluation. They are NOT complete training checkpoints.

Each resume/stepNNNNNN generation contains state.pt, compressed replay.npz and a COMPLETE.json checksum marker. Each generation also stores a compressed canonical action-log prefix so restored runs retain all committed training rows without duplicating an unfinished tail. State includes ring position, learner/optimizer, Python/NumPy/PyTorch/CUDA/agent RNG states, per-game histories, novelty visit counts, counters and the interleaved action/reset transcript with expected observation digests. The toolkit has no public environment-state restore API; restoration recreates the three seeded games and replays all recorded actions/resets, checking every response and final history before restoring learner RNG. It uses permitted observations; no hidden game object is inspected. Reconstruction time grows with elapsed training and is additional to future training time.

Restore requires identical protocol, source, game assets, torch version and device type. CPU fake-game tests establish bitwise equivalence for all four variants, and real three-game V2 reconstruction is tested locally. Colab runs CUDA continuation tests AND a real-game CUDA reconstruction preflight before pilots. This supports verified same-environment continuation, not cross-hardware bitwise guarantees. A mismatch fails loudly. Interrupted unsaved work since the last25K checkpoint must be rerun.

The VM writes first, copies and hashes completed payloads to Drive, then publishes COMPLETE last. Keep the latest two resume generations; retain all six frozen checkpoints. Cleanup is restricted to the newly created run's resume directory. Old3K/10K experiments and the original benchmark ZIP are never overwritten. On reconnect, upload the exact same source bundle, install the recorded package versions, mount Drive, copy one complete generation to VM disk, then run extended_train with --resume and a NEW output directory. A changed dependency runtime requires recreating the recorded environment or a new run; do not silently label it exact continuation.

## 5. Exact split and leakage limitations

- Training: ls20-9607627b, ar25-0c556536, bp35-0a0ad940.
- Development decisions/curves: ft09-0d8bbf25, vc33-5430563c, cd82-fb555c5d. These are already exposed; cd82 was PPO validation.
- Training diagnostics: the same three training games, reported separately from transfer results.
- Reserved, NOT confirmed untouched: sb26-7fbdac44, sk48-d8078629, su15-1944f8ab. User confirmed on this turn to keep them reserved pending teammate confirmation. No downloads, makes, probes, evaluation or training on them.
- Never label bp35, ls20, ar25, cd82, ft09, vc33, g50t, ka59 or previously probed dc22 as unseen. The original Configs/eval.yaml split is historical and cannot establish independence after actual exposure.

Final heldout testing requires teammate confirmation and a frozen protocol before any outcome is viewed. If confirmation never arrives, report these results as exposed-game development evidence and omit unseen-generalization claims.

## 6. Launch sequence

1. Upload Notebooks/DQN_extended_75K_pilots.ipynb to Colab; select T4 GPU.
2. Upload the NEW dqn_extended_source_*.zip, not the earlier benchmark bundle. Exact SHA256 and every member hash are verified; no remote branch is required.
3. Mount Drive. Record package/hardware/RAM/disk information; verify exposed game caches offline.
4. Run CPU/CUDA regression tests and real three-game CUDA continuation check. A failure stops progression.
5. Run V0 seed 42 to75K, then V2 seed 42 to75K, with the full 450K epsilon schedule. Both save at25K/50K/75K and publish to Drive.
6. Validate all 75K CSV rows,14,801 updates,25K actions/game, finite losses, tensors, replay shapes, masks, progress/reset logs and GPU samples. Zero completions remain a valid scientific outcome.
7. Evaluate the two frozen75K policies and matched random reference on the declared DEVELOPMENT set with Task 3; save official scores and trajectories. Failures/invariant mismatches must be repaired before a matrix.
8. Download the evidence ZIP and inspect the updated cost estimate. Freeze the protocol after implementation review. Only then use the separately gated matrix command (not present as an executable notebook cell). All12 runs have450K actions. Fresh matrix runs are the default; exact validated pilot continuation may be chosen explicitly using the resume command without changing settings.
9. Evaluate all six fixed checkpoints, aggregate by independent TRAINING seed, then perform the separately approved heldout evaluation only after exposure confirmation.

Shared evaluation: click_grid64, legal masks, greedy frozen policies, no learning, level resets after GAME_OVER, stop atWIN or1000 selected actions, no time cap; evaluation seeds0/1/2 with same game-seed mapping; memory cleared on game/controller reset. Official local toolkit score is primary; cross-check recomputed score, exactly one scorecard run and zero illegal/full resets after start. RESET is excluded from selected-action budget but included in official efficiency accounting. Preserve failures rather than dropping them. Intrinsic rewards are never task success.

## Report-ready methodology and planned analysis

We test a2x2 ablation: novelty absent/present x decision history1/4. V0=(0,1), V1=(0.01,1), history-only=(0,4), V2=(0.01,4). Prespecified contrasts are V1-V0 and V2-history-only for novelty; history-only-V0 and V2-V1 for history; their difference estimates interaction descriptively. No hypothesis is changed because pilots score zero. Pilots validate implementation/compute; they are not used to select a winning seed or cherry-pick a checkpoint.

For each checkpoint/game, average repeated evaluation seeds within each trained model, then compare the three independent training-seed means. Report individual points, mean, sample SD and range; n=3 gives weak uncertainty resolution. Do not count nine evaluation episodes as nine independent trained models, or interpret zero observed variance as proof of no stochastic uncertainty. Keep games equally weighted and report per-game outcomes too.

Learning curves plot official score, completion rate and levels versus selected TRAINING actions at75K intervals, with individual-seed traces and between-training-seed variability. Report training task-reward and intrinsic-reward curves separately; loss magnitude is not comparable evidence of task competence across shaped/unshaped rewards. Use trajectory action histograms, click behavior, entropy and repeated-action patterns to diagnose policy changes. Log throughput and compute separately from sample efficiency.

For the eight-page RLC report, use one compact protocol/ablation table, one fixed-checkpoint learning-curve figure, one official-score/per-seed result table and a short behavior/failure panel. Place detailed source hashes, resume evidence and configuration listings in the reproducibility package if the submission rules permit. The original native3K panel and shared 10K diagnostics remain historical pilot evidence; do not pool them with the new three-game 450K experiment or present them as repeats of the same contract. No new training result is written into the report until measured.

## Validation recorded before Colab handoff

Local focused suite:14 passed,4 CUDA-only tests skipped. Follow-up persistence/gating/aggregation checks:3 passed. Real three-game V2 at1200 actions, interrupted at1100, restored an identical transcript and identical online/target parameters. A separate Task3 smoke evaluated that1200-action snapshot for20 actions each on ft09/vc33/cd82; all official-score/reset/legal-action invariants passed. These are compatibility checks, NOT the requested75K GPU pilots and not learning results. See results/extended_design_20261024. The notebook runs the skipped CUDA checks before any pilot.
