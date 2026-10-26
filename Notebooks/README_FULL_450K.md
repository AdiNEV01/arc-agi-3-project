# Full450K DQN launch

1. Open https://colab.research.google.com and use File > Upload notebook > DQN_FULL_450K_COLAB.ipynb.
2. Runtime > Change runtime type > T4 GPU.
3. Run all. In Cell2, upload only dqn_extended_full_source.zip.
4. In Cell3, approve Drive mounting and Google authentication for account storage checks. At least4GiB free Drive storage is required initially.
5. Cell4 checks genuine75K evidence, action encoding and short CUDA process-kill/resume recovery. It never retrains75K pilots.
6. Cell5 starts ALL12 full450K runs sequentially. Look for TRAIN v0_seed42 and increasing actions. DRIVE COMPLETE ...25000 confirms durable checkpointing.

Progress/checkpoints: MyDrive/ARC_DQN_extended/FULL_450K_dqn_extended_v1_20261024.

After a disconnect: reconnect a T4, rerun Cells1-4 with the SAME notebook/ZIP, then click RESUME TRAINING. No paths need editing. Completed jobs are verified and skipped; incomplete jobs use their latest complete checkpoint. Do not delete the Drive experiment folder. If the runtime/package versions change, the learner refuses incompatible continuation rather than silently changing the experiment.

Complete state is saved every25K; all six75K frozen milestone models are retained. Latest two complete resume generations survive across attempts. At most two recoverable subprocess retries occur; protocol corruption, incompatible versions, unavailable GPU and exhausted quota stop safely. The whole VM cannot restart itself. No session keep-alives, CPU fallback, purchases, pushes, merges or heldout evaluations are performed.

Scientific protocol remains dqn_extended_v1_20261024. The sole learner-file change since successful pilots hardens publish(): all transferred files are checksummed before COMPLETE. AST comparison verifies the rest of the learner is unchanged. The archive includes genuine pilot evidence so matrix gates do not depend on an absent remote branch or missing pilot replays.

Development evaluation is a separate command supplied at the end of the notebook; training is the immediate priority.
