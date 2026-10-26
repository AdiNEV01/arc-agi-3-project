# Extended DQN: all twelve training runs verified

The complete ZIP supplied on 2026-10-26 resolves the gaps in the earlier Drive download. All 12 runs independently verify at 503,808 selected training interactions, 100,562 gradient updates and 167,936 actions per training game. Total matrix training interactions: 6,045,696.

- **Full audit:** `complete_archive_audit_20261026_verified/audit.json`. All 360 member CRC checks, 84 milestone hashes, 72 recovery payload hashes and 18 training/queue source hashes passed. All twelve final models loaded safely on CPU with finite online/target weights and outputs. All twelve full histories passed action sequence, round-robin, counters, epsilon, reward eligibility, update schedule and summary checks. No required files are missing.
- **Continuation validation:** `curated_20261026/continuation_validation.json`. All 72 original milestone hashes remain unchanged; all twelve histories match their original 450K summaries at the continuation boundary. Only the protocol ID, action cap and milestone list changed.
- **Tables:** `curated_20261026/training_by_run.csv` (12 rows), `training_by_game.csv` (36 rows) and `training_binned_5000.csv` (1,212 rows). Task and intrinsic rewards are separate. Runtime columns describe the 53,808-action continuation.
- **Registry and config:** `Configs/dqn_503808_checkpoints.json` declares twelve final models and twelve paired 450K models; `Configs/dqn_503808_training.json` preserves the extended configuration. These paths are relative to the repository root.
- **Figures and interpretation:** `Analysis/DQN/extended_503808/figures/` and `Reports/DQN-Reports/extended_503808/TRAINING_RESULTS.md`.
- **Evaluator smoke:** `evaluator_smoke_20261026/validation.json`. Real final V0/V2 models passed six development game runs at ten actions per game, with scorecard, reset, legality, score-recomputation and unchanged-weight checks. This does not substitute for formal evaluation.

The full frozen development evaluation has **not** run. Cumulative level-delta task reward was 28 across the matrix at 450K and 29 at 503,808; no full-game training wins were recorded. These exploratory training observations do not establish frozen-policy rankings or generalisation.

The full original ZIP remains unchanged at its supplied Downloads location: `DQN_503808_CONTINUED_20261026T040955Z_0cff534c.zip`, 1,182,683,484 bytes. SHA256: `52756cb244b7b0ca1fb62b06d14d52d3ad9481a0063c69ce06a60fdc611e0bcc`. It contains complete 500K and 503,808 recovery generations for all twelve runs. Recovery bytes were checksum-verified; no new environment reconstruction or optimizer continuation was performed here.

Local materialized endpoints, metadata and compressed histories are under ignored `artifacts/dqn_extended/52756cb244b7/`. Full state/replay files stay in the original archive and external backup. Preserve both complete 450K and 503,808 backups. The registry contains no invented Drive URL; add a verified shared artifact URL before publication.

The documented T4/cu130 to RTX 3050/cu126 numerical migration remains part of the experiment history. Windows CPU loading on PyTorch 2.14.1 does not establish bitwise training equivalence to Linux/PyTorch 2.11.0. Reserved sb26/sk48/su15 remain untouched.

The earlier `archive_part001_audit/` is retained as historical intake evidence and is superseded by the complete audit. No Git push or merge has been performed.
