# Submission status — 26 October 2026

This repository records completed experiments and their limitations. A clean code snapshot is ready for team review. The full coursework submission is **not yet complete**.

| Deliverable | Current evidence | Remaining work |
| --- | --- | --- |
| DQN implementation and two improvements | V0, novelty V1, history-plus-novelty V2 and history-only ablation; shared training and frozen evaluator | Complete the common frozen evaluation of the extended models |
| DQN training | All 12 models verified at 503,808 actions; full histories, continuation audit and plots | Provide the shared model archive link; retain original recovery backups |
| PPO implementation and improvements | Baseline, curiosity, memory-only and memory+curiosity; twelve verified final models at 503,808 interactions | Run the matching common evaluation; distinguish legacy proxy scores and historical failure experiments |
| Baseline/reference and final comparison | Shared scorecard evaluator and matched random-policy support; smoke evidence | Complete and analyse comparable evaluations for baseline, improvements and ablations |
| Held-out/adaptation discussion | Exposure conflicts documented | Obtain teammate exposure confirmation before using reserved games; otherwise state the justified limitation |
| Final report PDF | Historical DQN section, extended DQN training section, PPO draft and analysis | Produce a joint RLC 2026 report with cover page disabled; at most eight main-content pages excluding references; add all authors/student emails and proper attribution |
| Submitted source notebook | Historical Colab training/benchmark notebooks and competition bridge helper | Export the actual successfully evaluated Kaggle notebook matching the nominated submission |
| Competition and Paper Track evidence | No verified submission identifier/score/timestamp or Paper Track link in this snapshot | Record both tracks, nominated leaderboard entry, saved write-up and member contributions |
| Moodle package | Source, configs, curated supporting evidence | Assemble the required team-named ZIP containing final PDF, source notebook and competition evidence |

The local assignment brief lists **27 October 2026, 23:59 SAST** as the Moodle deadline. Requirements above come from sections 3–6 of the team's supplied `RL_Assignment.pdf`; the brief is reference material, not part of this repository. Current competition mechanics/deadlines must be checked directly before submitting.

## Claims to preserve

- Training interaction parity does not equalise rewards, action spaces, compute, observation history or replay reuse.
- Sparse exploratory level-delta reward and novelty bonuses are not official ARC-AGI-3 scores. No Kaggle score is claimed by local smoke tests.
- The extension added one level-delta task reward across the DQN matrix and recorded no full-game training wins. Describe this honestly; a well-supported unsuccessful improvement is still analysable evidence.
- Keep independent training seeds separate from repeated evaluation seeds.
- Document the authorised T4/cu130 to RTX3050/cu126 numerical migration. Historical cross-hardware bitwise equivalence failed.
- Reserved sb26, sk48 and su15 remain unconfirmed and untested. Historical dev/held-out labels do not undo prior exposure.
- Final report citations and any required generative-AI declaration remain the team's responsibility.

## Where the team starts

Use [README.md](README.md), [model access](docs/ARTIFACT_ACCESS.md), [the extended DQN report](Reports/DQN-Reports/extended_503808/TRAINING_RESULTS.md) and [the final checkpoint registry](Configs/dqn_503808_checkpoints.json). The current source tree includes the relevant latest teammate summaries; pinned source refs and intentionally external raw artifacts are recorded in `results/submission_readiness_20261026/teammate_source_inventory.json`.
