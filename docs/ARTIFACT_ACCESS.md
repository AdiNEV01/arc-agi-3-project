# Models, raw evidence and safe reproduction

Source, configurations, compact results, figures and audit receipts belong in Git. Full recovery states, replay buffers, raw histories, delivery ZIPs, environments and cached game rules stay in backed-up artifact storage. Existing historical PPO pilot ZIPs already tracked in the repository remain preserved.

## Final DQN models

Use `Configs/dqn_503808_checkpoints.json`: all twelve final 503,808 checkpoints and their twelve paired 450K snapshots are fixed before frozen evaluation. The full archive is `DQN_503808_CONTINUED_20261026T040955Z_0cff534c.zip`, size 1,182,683,484 bytes, SHA256 `52756cb244b7b0ca1fb62b06d14d52d3ad9481a0063c69ce06a60fdc611e0bcc`.

**Shared download location:** read `archive.artifact_url` in the registry. A null value means the owner still needs to supply a verified team-accessible Drive link. Do not substitute a similarly named incomplete Drive part. Keep the complete original 450K and 503,808 folders backed up.

From the repository root, restore only endpoint weights and metadata, verify their hashes, and stop before evaluation:

```bash
python -m Experiments.evaluate_dqn_503808 --materialize-from /path/to/DQN_503808_CONTINUED_20261026T040955Z_0cff534c.zip --steps 450000 503808 --validate-only
```

This writes into ignored `artifacts/dqn_extended/52756cb244b7/`, reuses matching files and refuses conflicting files. Replay and optimizer state are not restored for frozen evaluation.

Game assets are downloaded separately through the official toolkit. Only declared training/development games are permitted by this preparation command:

```bash
python -m Experiments.prepare_extended_assets --config Configs/dqn_503808_training.json --include-development --download
```

The toolkit may require its normal account/API setup. Keep credentials outside Git. Do not inspect downloaded game-rule source when designing agents. Evaluation itself is offline and uses the exact versioned game IDs.

Run final-model development evaluation in a new folder:

```bash
python -m Experiments.evaluate_dqn_503808 --device cpu --out results/dqn_extended_503808/development_final_unique
```

Use `--device cuda` in the verified Linux environment, or add `--steps 450000 503808` for paired endpoints. This is frozen evaluation, with no learning. Baseline and improved variants all participate. Keep reserved games untouched.

## Final PPO models

`Configs/ppo_final_checkpoints.json` contains all twelve final paths, SHA256 hashes and Git blob IDs, pinned to commit `7d15ec69dd9e619344fa2f37092dbb4bca2f2957`. The files already exist in Git history on the team's PPO branch; they do not need to be copied into another commit. The new failure-analysis summaries are a separate historical experiment family and are not replacements for these final models.

Fetch the PPO branch and use the existing selective verifier to materialize only the twelve declared finals into ignored `PPO/logs/`:

```bash
git fetch origin PPO
python -m Experiments.verify_ppo_finals --ref 7d15ec69dd9e619344fa2f37092dbb4bca2f2957 --out results/ppo_finals_new_verification
```

The verifier checks model metadata and performs its own short shared-interface smoke. It is not the final comparative evaluation. Its output folder must be new.

## Historical artifacts

Original 3K DQN checkpoint blobs remain on the pinned Task 3 source ref recorded in `results/submission_readiness_20261026/teammate_source_inventory.json`; their summaries/reports remain in the current tree. Optional checkpoint tests skip clearly if those old binaries are absent.

The original Colab/Linux delivery ZIPs and 75K evidence ZIP are retained locally/external to Git. Source-delivery ZIPs contain cached game assets and must never enter the submission tree. Historical launcher source and its manifests remain part of the experiment provenance; run those launchers from their exact historical delivery bundle if recovery is required. A fresh Git clone is not that versioned recovery bundle. Package-identity tests skip when the optional delivery ZIPs are absent; synthetic learner/replay/RNG restoration tests still run.

Do not rerun completed training to reconstruct missing download files. Do not update the school driver or modify original checkpoints. The current recovery archive has complete 500K and 503,808 generations for every run; checksum verification is separate from a new environment-reconstruction test.

The checkpoint guards compare exact Python source bytes, including historical line endings. `.gitattributes` preserves those bytes across clones and gives shell launchers LF endings. Do not run an automatic formatter or newline conversion over the recorded training/queue source before attempting historical restoration.
