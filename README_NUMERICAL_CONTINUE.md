# Approved numerical continuation on the existing RTX 3050

This small patch uses the existing `arc_dqn_cu126` directory and its working `.venv-arc-cu126`. It adds files only. It installs no packages or drivers and leaves the strict launcher, original checkpoints, prior audit and original delivery ZIPs intact. The user has authorised continuing with the changed numerical environment after the CUDA13/T4 equivalence check failed.

Copy `dqn_continue_numerical_patch.zip` into Linux Downloads, then run:

```bash
cd ~/Downloads
unzip -o dqn_continue_numerical_patch.zip -d arc_dqn_cu126
cd arc_dqn_cu126
bash continue_numerical.sh --accept-numerical-migration
```

The new entry point skips the failed comparison against the historical T4 trajectory. It still verifies the original source, protocol, packages, actual CUDA operations, checkpoint hashes and every reconstructed game response. The acceptance flag records the already-approved numerical migration; it is not an interactive confirmation prompt.

For the current source, V0/V1/history-only seed42 are complete at 450K, V2 seed42 has an original 400K checkpoint, and seeds123/456 are pending. The original source comes from the failed audit's `original_checkpoint_root`, never from its `from_earlier` diagnostic output. `--checkpoints /absolute/path/to/the/complete/original/folder-or.zip` can select the original source explicitly.

Before launching the queue, the unchanged learner restores the actual latest checkpoint. A ten-action uninterrupted continuation is compared with a five-action continuation, controlled stop, and five-action resume in a new process. All model/target/optimizer tensors, replay arrays/masks, counters, histories, novelty visits and RNG states must match bit for bit on this GPU. Game reconstruction checks every original saved response. Input conversion is also compared exactly with the original full state. Any corrupt state, reconstruction mismatch, nonfinite value or same-GPU restoration failure still stops execution.

This needs three reconstructions of the saved game history and may take several minutes. It does not replay the historical 25K training interval. `VERIFYING` messages point to per-process logs. No game action occurs until a process has reconstructed its saved state; reconstruction itself does not increment the selected-action budget.

After validation, a new `CU126_450K_RESULTS_NUMERICAL_<unique id>` tree receives a complete copy of the original experiment. The verified ten-action continuation is adopted there. For the current interrupted model, expect `TRAIN v2_seed42: 400010/450000`; the adopted ten actions count once in its canonical training history. Duplicate diagnostic actions are excluded from matrix counts. The three finished runs are validated and skipped, V2 finishes its remaining actions, then the eight genuinely unstarted seed123/456 entries run in the original order. A previously active job is never restarted at zero because its state is missing.

The original 450K budget, seeds, game set, reward, batch size, update frequency and milestone schedule stay fixed. No heldout evaluation is launched. New checkpoints continue every 25K actions. Keep the terminal open and the PC awake. To stop, press Ctrl+C. To resume the verified local copy later, run `bash continue_numerical.sh`; the saved pointer and proof are checked, so the verification is not repeated unless the environment or patch changes.

The actual results location is printed as `Local persistent results:` and recorded in `NUMERICAL_ACTIVE_RESULTS.json`. Print it with `cat ~/Downloads/arc_dqn_cu126/NUMERICAL_ACTIVE_RESULTS.json`. Back up that whole result tree to Drive after stopping or finishing. Uploads are manual; the patch does not sync or upload anything.

`numerical_migration/RESTORATION_PASSED.json` in the result tree records the approval, GPU/driver/software, original paths and hashes, same-GPU checks and adopted checkpoint. Package freeze and verification logs are alongside it. `numerical_audits/NUMERICAL_*/CONTINUATION_REPORT.json` records status or failures. These results must be described as a documented T4/cu130 to RTX3050/cu126 numerical migration. Historical cross-hardware bitwise equivalence failed and is not claimed. The same-GPU check verifies restoration in the new environment, not identity with future T4 training.

This patch was prepared and CPU-tested on Windows. The launcher performs actual GPU restoration checks on the Linux PC; no remote training is claimed to have started during preparation.
