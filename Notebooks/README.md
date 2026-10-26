# Notebook guide

The completed DQN matrix is now at 503,808 selected training actions per model. Start with the repository README and `Reports/DQN-Reports/extended_503808/TRAINING_RESULTS.md` to inspect results. Do not launch historical full training merely to view completed models.

| Notebook | Purpose | Status |
| --- | --- | --- |
| `DQN_T4_benchmark.ipynb` | Short CPU/T4 throughput comparison | Historical benchmark; not a learning-performance result |
| `DQN_extended_75K_pilots.ipynb` | V0/V2 pilot protocol and CUDA feasibility | Historical pilot stage |
| `DQN_FULL_450K_COLAB.ipynb` | Original 12-run 450K training/recovery queue | Historical training delivery; completed and later continued on Linux |

These notebooks originally used source-delivery ZIPs to carry unpushed code. The ZIPs remain external because some contain cached game-rule assets. A fresh Git clone does not include them; their source/manifest checks must not be bypassed. Use the current frozen evaluator to analyse the registered completed models. The actual final Kaggle notebook matching the nominated competition submission remains a separate required deliverable.
