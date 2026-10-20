# PPO implementation checklist: stages 1 to 7

This checklist is the practical file-by-file workflow for the PPO workstream.
It does not claim that the entire group assignment is complete: the brief also
requires a second learning algorithm, two substantive improvements per
algorithm, a report, and official competition submission evidence.

## 1. Environment adapter and benchmark semantics

Files: `arc_gym.py`, `check_arc_env.py`, `diagnose_env.py`

- [x] Keep ARC Toolkit creation separate from PPO.
- [x] Give the policy only a rendered observation and four mapped movement actions.
- [x] Return Gymnasium observations, rewards, termination flags, and diagnostic info.
- [x] Log environment seeds explicitly and explain that `reset(seed=...)` does not
      recreate/reseed the underlying ARC game.
- [ ] Run `python check_arc_env.py --game-id ls20 --seed 0` in the project's `.venv`.
- [ ] Save and inspect sample frames to verify the rendered information.

## 2. Model definition and experiment variants

Files: `ppo_baseline.py`, `policy_networks.py`, `reward_wrappers.py`

- [x] Baseline PPO with Stable-Baselines3's `CnnPolicy` and level-completion reward.
- [x] `grid_cnn`: custom grid-preserving CNN; reward is unchanged.
- [x] `novelty`: observation-count exploration bonus; default CNN is unchanged.
- [x] `grid_cnn_novelty`: combined ablation.
- [ ] Train and evaluate these variants rather than assuming an improvement helped.

## 3. Full training run, checkpointing, and diagnostics

Files: `ppo_baseline.py`, `training_callbacks.py`

- [x] Seed Python, NumPy, and PyTorch.
- [x] Record configuration, Python/package/hardware details, and run summary.
- [x] Save periodic checkpoints, last/recoverable model, and final model.
- [x] Save episode and rollout logs as CSV and Stable-Baselines3 progress metrics.
- [ ] Complete the intended 2,000,000-interaction run and retain all outputs.

## 4. Matched evaluation and random reference policy

File: `evaluate_simple.py`

- [x] Evaluate random and PPO policies with the same configured game seeds.
- [x] Save every episode, including failures, terminal states, lengths, and rewards.
- [x] Keep local levels completed separate from official ARC score.
- [ ] Run random and each PPO version with comparable seed/action budgets.

## 5. Analysis tables and diagrams

Files: `analyze_results.py`, `plot_training.py`

- [x] Generate aggregate evaluation tables and per-seed results.
- [x] Generate success-rate, level-completion, action-count, and ending-state plots.
- [x] Generate a training dashboard with rewards, episode lengths, levels,
      optimisation losses, approximate KL, explained variance, and throughput.
- [ ] Interpret the curves only after real experiment CSV files exist.

## 6. Trajectories and GIF replays

Files: `record_episode.py`, `make_gif.py`

- [x] Record observation frames and per-action trajectory data.
- [x] Produce an animated GIF for random or trained PPO policies.
- [ ] Save representative failure cases and any successful cases for the report.

## 7. Reproducible evidence and reporting

Files: `README.md`, `requirements.txt`, `ASSIGNMENT_MAPPING.md`

- [x] Document exact commands and output paths.
- [x] Document the reward, observation/action abstraction, limitations, and variants.
- [x] Provide mapping to assignment expectations and an output checklist.
- [ ] Cite any external code/research and disclose AI assistance as required by Wits policy.
- [ ] Add the official ARC submission ID, official score, timestamp evidence, Paper Track
      link/write-up copy, and member contributions before final submission.
