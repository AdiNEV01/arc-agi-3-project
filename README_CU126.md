# RTX 3050 / driver 565: explicit CUDA 12.6 recovery

This delivery supports Python 3.13.16 and **PyTorch 2.11.0+cu126**, using a separate `.venv-arc-cu126`. It does not install a driver, use sudo, change your existing venv, overwrite the old delivery ZIP, or write into historical checkpoint folders. PyTorch's wheel includes CUDA user-space libraries; their versions necessarily differ from the cu130 build. NumPy, ARC and Gymnasium versions stay pinned. Every attempt records the complete new pip freeze and actual GPU/driver details.

Download `dqn_linux_cu126_recovery.zip` into Linux `Downloads`. Keep the complete Drive experiment download there, or keep its extracted folder. Leave the old Linux delivery in place. Stop any other copy of the training queue before verifying.

```bash
cd ~/Downloads
unzip dqn_linux_cu126_recovery.zip -d arc_dqn_cu126
cd arc_dqn_cu126
bash start_linux.sh --cuda-build cu126
```

The launcher first looks for the original launcher's saved checkpoint location, then a complete `FULL_450K_dqn_extended_v1_20261024*.zip` in Downloads. To specify it explicitly:

```bash
bash start_linux.sh --cuda-build cu126 --checkpoints '/absolute/path/to/complete/folder-or.zip'
```

If Drive split the download into several ZIPs, extract **all** parts into one directory and give that directory. A model `.pt` alone cannot resume this experiment. The folder must include `QUEUE.json`, `progress.json`, full `attempts/.../resume/step.../` states and frozen milestones. Previously recorded actions without a complete checkpoint cause a stop. No old training run is restarted at zero.

For verification without an automatic matrix launch, use the alternative direct entry point:

```bash
bash start_linux_cu126.sh --verify-only
```

What happens:

1. A new bootstrap venv installs uv, which creates `.venv-arc-cu126` with the original Python version. Only this new recovery venv receives the cu126 wheel and pinned project dependencies. Existing venvs and the school driver are untouched.
2. The program requires `torch==2.11.0+cu126`, CUDA runtime 12.6, the RTX 3050 and one CUDA device. It performs actual project one-frame and four-frame batch-8 forward/backward/Adam operations, synchronizes CUDA, and checks finite output. A positive `is_available()` alone is insufficient.
3. Source, protocol, runtime asset and checkpoint checksums are verified. For each interrupted job, **two distinct original complete cu130 checkpoints** are required. These are normally 25K actions apart. Missing, corrupt, incompatible or changing originals stop verification.
4. Verification-only checkpoint copies receive the target build metadata and a `CUDA_BUILD_CONVERSION.json` receipt. Tensor, optimizer, replay, RNG, novelty and game state are preserved. Original file hashes and original software metadata are retained in the receipt. No torch version is spoofed, no original trainer check is disabled, and no original file is edited.
5. On the RTX, the unchanged trainer reconstructs the earlier checkpoint's game history, restores the entire learner, and repeats the known interval plus ten diagnostic actions. At the historical end, the new result must match the actual cu130 reference **bit for bit** for online/target weights, optimizer, counters, masks/replay arrays, every saved RNG, frame histories, novelty visits and complete action/reset transcript. Training CSV fields including loss must match; elapsed time is excluded.
6. The unchanged trainer separately restores the **actual latest** original checkpoint copy and runs those same ten diagnostic actions. Its full saved learner, replay and history must match the uninterrupted cu126 path at the same step. This tests full latest-state restoration rather than merely loading model weights. Diagnostic actions stay outside matrix results.
7. Any mismatch or missing reference stops the program. There is no tolerance-based pass, CPU fallback, zero-step replacement, or automatic acceptance of a new numerical trajectory.
8. Only after every interrupted job passes, the entire experiment is copied into a new `CU126_450K_RESULTS_<unique id>` folder. Explicit converted checkpoints are added in new attempt directories; old attempts and frozen milestones remain in the historical source. The original queue validates and skips completed jobs and continues interrupted jobs to their original 450K target. Only genuinely unstarted entries already listed in the original queue can begin at zero. A previously active entry without its state is rejected.

This can take several minutes: verification repeats one saved interval per interrupted job and reconstructs each saved game's full transcript. It does not repeat 450K runs or the 75K pilots. Keep the terminal open and the PC awake. A failed verification is a valid stop, not permission to restart training.

Results and diagnostics:

- `cu126_audits/<unique id>/RECOVERY_REPORT.json` records success or the first failure. Probe logs are alongside it. A failed audit does not create or launch a matrix result copy.
- `CU126_ACTIVE_RESULTS.json` records the active new result folder after a successful migration. `cat CU126_ACTIVE_RESULTS.json` prints its path. Rerun the same launch command to continue from its verified latest states.
- The new result folder contains the usual job results, plus `cuda126_migration/EQUIVALENCE_PASSED.json`, package freeze and probe logs. New model checkpoints every 25K and all six frozen milestones follow the original retention policy in the new copy. The historical source remains intact.
- Results are local. Back up the whole new result folder to Drive after stopping or finishing. Keep the historical Drive and Linux originals. Do not copy new files over the original experiment folder.

Reproducibility limit: matching a saved historical interval and a short same-GPU continuation is strong empirical evidence for those tested states. It cannot establish that every future cu126/RTX trajectory will stay bitwise identical to a cu130/T4 counterfactual. The report states this limit and records the hardware/build migration. A successful gate permits the requested continuation on that documented basis; it never claims universal cross-hardware determinism. If the tested equivalence fails, this delivery stops and needs a separate scientific decision before accepting any numerically different continuation.

Official sources: [PyTorch cu126 wheels, including 2.11.0 cp313 Linux x86_64](https://download.pytorch.org/whl/cu126/torch/), [NVIDIA CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [PyTorch 2.11 reproducibility limitations](https://docs.pytorch.org/docs/2.11/notes/randomness.html). The `CUDA Version` in nvidia-smi is driver runtime support, not the GPU's compute capability; this program records the latter separately from the actual device.

This delivery was prepared and CPU-tested on Windows. CUDA operation and original Drive-checkpoint equivalence have **not** been verified on your remote RTX until this program reports its actual results there.
