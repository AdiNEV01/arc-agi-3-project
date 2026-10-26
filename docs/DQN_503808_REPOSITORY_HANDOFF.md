# Completed extended DQN: artifact handoff and repository integration

All twelve trained models are now independently verified at 503,808 selected interactions. The complete ZIP supplied on 2026-10-26 resolved the gaps in the earlier partial download. See `results/dqn_extended_503808/README.md` for the full audit and curated tables. Both endpoint registries/configuration files, training plots and a training-results report are populated locally. Formal frozen-policy evaluation remains outstanding.

Keep both entire result trees in Drive: original `CU126_450K_RESULTS_...` and final `DQN_503808_CONTINUED_...`. The latter includes full canonical histories from step1 through503,808, all earlier frozen milestones, original450K summaries/metadata and migration/budget provenance. Full state/replay payloads remain necessary for future training recovery but need not enter regular Git history.

Run `make_dqn_handoff.py` with the existing Linux folder as `--workspace` (or run from that folder). It creates a unique ZIP under Downloads containing the450K and503,808 frozen snapshots for each variant/seed, the twelve full compressed canonical training logs, per-job metadata/summaries/completion receipts, final recovery checksum receipts, source manifests and migration evidence. It excludes replay arrays, full resume state binaries, Python environments, game caches and unrelated original attempts. It verifies included model/history checksums and leaves original files intact.

The single handoff is sufficient for the initial audit and paired endpoint analysis. Request additional original450K files only if source/recovery validation exposes a gap. Training artifacts alone do not establish frozen-policy evaluation performance.

Repository destinations populated after validating the actual artifacts:

- `Configs/dqn_503808_checkpoints.json`:12 final paths, seeds, frame counts, training budget and SHA256 hashes; also declares paired 450K endpoints for a budget comparison planned before frozen evaluation.
- `Configs/dqn_503808_training.json`:exact approved extended configuration; preserve historical450K config separately.
- `Experiments/`, root continuation scripts and `Notebooks/`:reviewed training, restoration, extension, packaging and evaluation source. Do not commit source-delivery ZIPs containing cached game assets.
- `results/dqn_extended_503808/`:compact completion/provenance audit, per-seed training summaries, binned curve CSVs, official frozen evaluation records and aggregates. Full raw histories remain in the external evidence bundle.
- `Analysis/DQN/extended_503808/`:verified analysis source and figure outputs.
- `Reports/DQN-Reports/extended_503808/`:measured report contribution and reproduction/limitations notes. Keep historical3K/10K reports distinct.
- `artifacts/dqn_extended/` or `DQN/checkpoints/extended_503808/`:local materialized weight/recovery bundles, Git-ignored. Repo documentation should link the external model bundle, specify expected checksums, access requirements and evaluation/reproduction commands.

Before staging, inspect actual file sizes and Git status. New ignore entries cover the proposed binary locations, full generated training trees and handoff ZIPs; the prior ignore rule covered only the old direct DQN checkpoint location. Add curated outputs deliberately. Do not blindly add complete imported folders. No push or main merge is authorised by artifact handoff alone.

The next scientific step is common frozen evaluation of the declared models, beginning on already-exposed development games and a matched random reference. Reserved sb26/sk48/su15 remain unconfirmed and untouched. PPO comparison must use the same frozen evaluation contract; equal training counts do not erase different rewards, action spaces, data reuse or hardware history. A final report must label the documented cu130/T4 to cu126/RTX numerical migration and use training seeds as independent replicates.
