# Evaluation analysis

These are local level-completion results, not the official ARC-AGI-3 score.
All failed episodes are included.

| Tag | Episodes | Seeds | Successful | Success rate | Mean levels | Mean steps | Mean extrinsic return |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_s0 | 3 | 3 | 0 | 0.0% | 0.000 | 129.00 | 0.000 |
| baseline_s1 | 3 | 3 | 0 | 0.0% | 0.000 | 129.00 | 0.000 |
| baseline_s2 | 3 | 3 | 0 | 0.0% | 0.000 | 129.00 | 0.000 |
| grid_s0 | 3 | 3 | 0 | 0.0% | 0.000 | 129.00 | 0.000 |
| grid_s1 | 3 | 3 | 0 | 0.0% | 0.000 | 500.00 | 0.000 |
| grid_s2 | 3 | 3 | 0 | 0.0% | 0.000 | 129.00 | 0.000 |
| novelty_s0 | 3 | 3 | 0 | 0.0% | 0.000 | 129.00 | 0.000 |
| novelty_s1 | 3 | 3 | 0 | 0.0% | 0.000 | 129.00 | 0.000 |
| novelty_s2 | 3 | 3 | 0 | 0.0% | 0.000 | 129.00 | 0.000 |
| random_reference | 3 | 3 | 0 | 0.0% | 0.000 | 129.00 | 0.000 |

## Interpretation warning
No evaluated tag completed a level. This is evidence that the tested policies did not solve these episodes; it is not evidence that PPO can never learn the game. Reward curves should be interpreted together with the observation representation, exploration strategy, and failure replays.

## Limitations

- The number of distinct environment seeds is shown separately from episode count.
- Repeated resets of the same ArcEnv do not create a new ARC layout; use distinct game seeds for distinct seeded environments.
- Local level completion is not a substitute for the official scoring interface.
