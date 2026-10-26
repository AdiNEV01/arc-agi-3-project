# Continue all twelve DQN models from 450,000 to 503,808

Install this additive patch into the existing Linux `~/Downloads/arc_dqn_cu126` folder. It reuses the working `.venv-arc-cu126`, original learner and existing numerical-migration tools. It does not install packages or drivers, change the old protected source/config files, overwrite the 450K results, or upload anything.

Copy `dqn_extend_503808_patch.zip` into Linux Downloads, then:

```bash
cd ~/Downloads
unzip -o dqn_extend_503808_patch.zip -d arc_dqn_cu126
cd arc_dqn_cu126
bash extend_503808.sh
```

The first launch finds the completed 450K result folder from `NUMERICAL_ACTIVE_RESULTS.json`. All twelve jobs must have validated 450K full continuation states, complete committed histories, correct update counts, intact original frozen milestones, and a valid numerical-restoration provenance receipt. A missing or incomplete job stops initialization. It never starts a replacement learner from zero.

The patch creates a separate `DQN_503808_CONTINUED_<unique id>` result tree. It changes exactly three protocol fields in derived checkpoint copies: total action cap, protocol identifier and checkpoint schedule (adding the 503808 final). Original weights, target network, Adam state, replay buffer/position, RNG states, history, novelty visits, game transcript and action counters are preserved. Every converted checkpoint's dynamic fields and copied replay/history contents are checked exactly. Original state files remain unchanged.

Three source seed42 models may still carry cu130/T4 software tags because they finished before migration. Those are explicitly converted into new cu126 copies under the already-approved numerical migration policy. That build change and the source hashes are recorded for each model. The remaining source states already use cu126. Cross-hardware bitwise equivalence is not claimed. The same learner and game-reconstruction checks used in the successful 450K continuation still apply.

After initialization, expect:

```text
READY TO EXTEND v0_seed42: 450000 -> 503808
TRAIN v0_seed42: 450000/503808 -> ...
```

The queue extends each of V0, V1, history-only and V2 for seeds42/123/456, sequentially. All twelve finished 450K learners now have a larger cap; none is skipped merely for having finished the old cap. Each receives **53,808 additional selected training actions**, totaling 645,696 new matrix actions. Final counts are 503,808 per model, 167,936 per training game, and 6,045,696 across twelve models.

Replay is not cleared; no warmup or epsilon schedule restarts. Global step450000 continues with epsilon0.1, the original train-every5 schedule, batch8, learning rate, target-update phase, frame history and novelty counts. Expected gradient steps are 89,801 at450K and 100,562 at503,808: 10,761 additional updates per model. The final three actions follow the normal schedule, without an extra gradient update to force round numbers. Controller resets and evaluation actions are excluded from training-interaction counts.

Full recovery states continue at475K,500K and the exact503,808 final. All original six frozen milestones, including450K, are copied byte-for-byte. A seventh frozen503,808 model is added. Latest two full recovery generations are retained within the new tree. Historical450K results are not pruned. Initialization copies the necessary states and milestones, not every old failed attempt.

Each process reconstructs its saved game's history before selecting new actions. This can take several minutes; reconstruction does not add training actions. Keep the terminal open and the PC awake. If interrupted, rerun `bash extend_503808.sh`; the new pointer and checkpoints are verified, completed503,808 jobs are skipped, and incomplete jobs resume from their latest complete state. Do not relaunch the old450K entry point to perform this extension.

Check progress or completion:

```bash
cd ~/Downloads/arc_dqn_cu126
bash extend_503808.sh --status
```

Completion requires `ALL 12 COMPLETE: True`, with all twelve entries showing `complete 503808/503808`. The result folder is printed as `503808 RESULTS FOLDER:` and recorded in `EXTENSION_503808_ACTIVE.json`. `NUMERICAL_ACTIVE_RESULTS.json` continues pointing to the preserved450K experiment. To verify/import without starting new training, use `bash extend_503808.sh --verify-only`.

Results include per-job `frozen/step503808.pt`, validated `RUN_COMPLETE.json`, canonical full `actions.csv.gz`, and full recovery states under `attempts/`. A job's new `summary.json` measures its latest extension attempt; its cumulative counters include the earlier training. Original summaries/receipts are retained in the original450K folder and copied under each new job's `source_450k/`. `EXTENSION_PROVENANCE.json` records the exact budget changes, per-model sources and software changes. Original numerical-migration evidence is copied under `extension_provenance/`. Back up the entire new result tree to Drive after stopping or finishing and retain the original450K tree. Sync/upload remains manual.

PPO's saved finals record82 rollouts x2048 steps x3 environments =503,808 actual training interactions, rather than the requested500,000. Its additional3,000 cd82 validation actions per model are outside that training count. This extension matches training-interaction and per-training-game counts. Reward definitions, action spaces, optimizer/data-reuse schedules and the recorded hardware migration still differ; matching counts alone does not establish algorithm-only fairness. Evaluation and the reserved sb26/sk48/su15 games are not launched by this patch.

Prepared and CPU-tested on Windows. Actual source validation and GPU continuation execute when launched on the Linux PC; no remote training is claimed to have started during packaging.
