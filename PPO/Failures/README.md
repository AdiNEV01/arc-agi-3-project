# Historical PPO failure experiments

Selected analysis CSVs, plots, per-run configurations and summaries were copied byte-for-byte from PPO commit `31c9a88de404ea31ae9201d41bbd4ce9a28a78ef`. The complete source-file inventory and Git blob IDs are in `results/submission_readiness_20261026/teammate_source_inventory.json`. Full raw logs and model ZIPs remain preserved on that source branch.

These runs are a historical experiment family, including 500K/2M requested budgets and a grid-action variant. They are distinct from the four final PPO variants at 503,808 interactions registered in `Configs/ppo_final_checkpoints.json`. Do not pool their seeds, budgets or scores with the final comparison.

`analysis_final/analysis_summary.md` explicitly labels its measurements as local level-completion results, not official ARC-AGI-3 scores. No tested tag completed a level in that analysis. Preserve its caveats about reset/seed semantics and include unsuccessful runs when discussing these experiments.
