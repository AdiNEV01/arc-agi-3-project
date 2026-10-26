# Continue the Colab DQN experiment on Linux

For the RTX 3050 with driver 565, use the separate `dqn_linux_cu126_recovery.zip` and `README_CU126.md`: `bash start_linux.sh --cuda-build cu126`. The original cu130 path below requires a CUDA 13-compatible driver. The cu126 path tests actual saved-interval equivalence before any matrix continuation.

Download the complete Drive folder `MyDrive/ARC_DQN_extended/FULL_450K_dqn_extended_v1_20261024` to this PC's `Downloads` folder. In Drive, right-click the folder and choose **Download**. Keep its ZIP name beginning `FULL_450K_dqn_extended_v1_20261024`. The download must contain `QUEUE.json`, `progress.json`, all job folders, frozen models, and complete `attempts/.../resume/step.../` folders. A model `.pt` alone cannot restore replay, optimiser state, counters, RNG or game history.

Download `dqn_linux_resume.zip` to `Downloads` too. Use a new directory for this bundle, with the existing venv active:

```bash
cd ~/Downloads
unzip dqn_linux_resume.zip -d arc_dqn_linux
cd arc_dqn_linux
bash start_linux.sh
```

If Drive created several checkpoint ZIPs, extract **all** parts into one folder first. Then use `bash start_linux.sh --checkpoints /absolute/path/to/the/extracted/folder`. This option also accepts one complete ZIP. Do not use a partial archive or download individual checkpoint files.

The launcher reuses a compatible environment. Otherwise it prepares an isolated `.venv-arc-colab` beside this launcher using uv. The original training versions are Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.5.4, arc-agi 0.9.9, arcengine 0.9.3 and Gymnasium 1.4.0. CUDA must actually run on this PC. Installation needs internet access. These version guards come from the saved experiment; the launcher does not bypass them. See the [official PyTorch CUDA wheel index](https://download.pytorch.org/whl/cu130/torch/) and [uv Python installation documentation](https://docs.astral.sh/uv/guides/install-python/).

Expect `VERIFIED v0 seed42: .../450000` and equivalent lines for all twelve jobs. The first launch then performs a short, separate GPU interruption/recovery check. It does not repeat the 75K pilots. Next it prints `TRAIN ...: <saved actions>/450000`, increasing progress, and `SKIP COMPLETE ...` for verified finished jobs. Old queue messages saying `DRIVE COMPLETE` mean **local disk** in this adapter.

The printed **Local persistent results** path is your active experiment copy. The launcher remembers it on later runs. New attempts use unique directories and complete recovery states every 25K actions; the latest two valid states and all six frozen milestones are retained. Re-running `bash start_linux.sh` rescans the saved files and resumes. Up to 25K actions after the last committed checkpoint may need repeating. `bash start_linux.sh --inspect-only` checks the import and environment without launching training.

Leave the terminal running and prevent the PC from sleeping during training. You may close the browser after downloads finish. To stop, press Ctrl+C and later rerun the launcher. Back up the active experiment folder to Drive after stopping or completing a run: **this launcher writes to local disk and does not automatically sync Drive**. Keep the original Drive copy until the migration is verified. Avoid running the Colab and Linux queues at the same time.

This bundle includes the exact unpushed integration code used in Colab. Original learner, queue, protocol, runtime assets and checkpoint formats are byte-for-byte preserved from the original source ZIP. Only a Linux launcher and delivery documentation are added. The adapter replaces Colab T4/Drive-quota guards with real CUDA and local disk checks. Resume still checks source, software, protocol, assets and complete-file hashes. It also reconstructs every saved environment response before continuing; any divergence stops the attempt.

Migration records under `linux_migrations/` capture the GPU, driver, software and imported progress. The same-GPU interruption test checks exact restoration on the new machine. Different GPU hardware may change later floating point results; cross-hardware bitwise equality is not promised. The established budget remains 450K per job. The three reserved games are not launched by this adapter. No Git push, merge, evaluation or fresh 450K run has been performed while preparing this bundle.
