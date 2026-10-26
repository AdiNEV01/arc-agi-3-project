# DQN Colab benchmark handoff

## Start

1. Save Notebooks/DQN_T4_benchmark.ipynb and the accompanying Notebooks/dqn_colab_source_*.zip locally.
2. Open https://colab.research.google.com and choose File > Upload notebook.
3. Runtime > Change runtime type > T4 GPU > Save. Run all cells.
4. Upload the source ZIP when prompted. Mount Drive when asked, or set USE_DRIVE=False before that cell.
5. Expect two runs of 2,000 selected actions and 388 updates each. Results include per-device actions/s, loop time, environment/selection/inference/update breakdowns, transfer timings, versions, source hashes, raw CSV, console logs and model snapshots. Each completed run is copied to MyDrive/ARC_DQN_benchmarks/<unique ID>. A results ZIP downloads at the end.

Availability varies: https://research.google.com/colaboratory/faq.html . T4/CUDA checks fail explicitly if the requested GPU is unavailable. Requires Python3.12+. The notebook pins ARC dependencies and keeps Colab's CUDA-enabled PyTorch>=2.8,<3, recording exact versions. There is no prepared T4 measurement yet.

## Source delivery and contract

The ZIP contains the current integration/2026-10-24 working files, including uncommitted adapter changes. No remote integration branch, GitHub login, push or merge is needed. It includes only DQN/shared-wrapper code, focused tests and the already exposed ls20 offline cache, copied opaquely without inspecting game rules. No historical checkpoints, credentials, results or heldout games are included. SHA256 checks validate every bundled file.

After source edits, rebuild from repo root with: python -m Experiments.package_colab_benchmark. Use its newly printed ZIP. This is deliberately a working-source snapshot rather than git archive HEAD, which would omit uncommitted integration work.

TASK3_README requires ArcGymEnv training. Verified path: DQN.train --backend shared -> SharedDQNEnvironment -> ArcGymEnv -> local arc_agi engine. The separate agents.dqn_agent:DQNPolicy remains a frozen evaluation bridge. Default native training is retained for compatibility; benchmark commands explicitly select shared. This measures training throughput, not official score or learning quality.

Protocol: V0; ls20-9607627b only; seed42; batch8; replay3000; warmup64; updates every5 actions; Adam1e-4; gamma0.99; target sync every250 updates; epsilon1 to0.1 over2000; one frame; 64x64 pixel clicks; level resets; episode-limit0. Reward is level delta. RESET is controller-owned and counted separately from selected actions. No heldout games are used.

## Device handling and timing

--device defaults to cpu; CUDA requests never silently fall back. Online/target models, masks, labels and Bellman math share the device. Replay stays in host uint8 memory. Compact uint8 batches transfer before int64 conversion and PyTorch one-hot on the device; padding16 encodes to all-zero channels. Model architecture and state-dict keys remain compatible. Load CUDA checkpoints on CPU with map_location='cpu'. Snapshots lack replay/RNG and are not exact-resume checkpoints.

--profile uses CUDA synchronization around timed stages. action_selection includes preprocessing, masks, inference and scalar GPU-to-CPU return; nested policy_forward isolates greedy network inference. Random selections have no forward call. gradient_update includes CPU sampling, transfers, online/target passes, backward, optimizer, target sync and loss.item download. Nested encoding and label/mask transfer times must not be added to the parent again. Environment/reset remain CPU work. Loop time includes CSV flush/logging/history; final checkpoint, whole-process setup/import time, install and Drive copy are distinguished. Synchronization adds overhead; confirm important results with profiling disabled. Batch8 and scalar transfers can offset GPU gains.

## Local validation

Six focused regression checks passed; two CUDA cases were skipped on this CPU-only host. Coverage: actual updates, masks, padding, history1/4, terminal targets, target synchronization, checkpoint reload on CPU, shared reward/reset/action mapping, timing output and overwrite refusal.

Actual matching CPU benchmark: 2,000 actions in 73.611658s = 27.169609 actions/s, 388 finite-loss updates, 15 resets, zero truncations, zero level completions. Gradient updates took 63.760s, selection 5.171s (forward 4.484s nested), environment steps 3.948s and resets 0.079s. Evidence: results/colab_benchmark_preparation_20261024/local_cpu_timing.json. Used PyTorch 2.14.1+cpu, one thread. The earlier 10K reference was 410.140s = 24.382 actions/s. Different run length, exploration mix, trajectories and software versions limit cross-run claims.

Matching local command (from repo root with dependencies available; choose NEW suffix/output folder):

```text
python -m DQN.train --backend shared --operation-mode offline --game ls20-9607627b --seed 42 --steps 2000 --variant v0 --batch-size 8 --warmup 64 --train-every 5 --epsilon-decay 2000 --episode-limit 0 --reset-mode level --torch-threads 1 --device cpu --profile --run-suffix YOUR_UNIQUE_ID --output-root outputs/YOUR_UNIQUE_ID
```

## Decision before further training

Primary comparison is the paired Colab CPU run. At least 1.5x T4 throughput, correct counts, finite losses and zero failures is a provisional GO. Confirm with reversed order, PROFILE=False and a fresh RUN_ID/output directory. At most 1.2x is no-go on speed alone; 1.2-1.5x needs another measurement. Confirm useful absolute throughput relative to local 24.382 actions/s too. If environment cost dominates, another GPU alone offers limited gains.

No 450K run is authorized or started. Full training budgets need a separate decision. V0 does not establish V1/history/V2 throughput or learning gains. Identical CPU/CUDA seeds can still yield different trajectories. Ranking another NVIDIA PC requires its exact GPU and this same benchmark. Existing checkpoints/logs/results were preserved; no push or main merge occurred.
