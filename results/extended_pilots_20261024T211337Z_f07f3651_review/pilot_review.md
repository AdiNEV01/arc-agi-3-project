# 75K DQN pilot review

Evidence: pilots_20261024T211337Z_f07f3651_evidence.zip. Review of actual results; no further training launched.

## Decision

Both pilots pass the technical launch criteria. Recommend keeping the declared protocol/hypotheses fixed and proceeding to the complete four-configuration, three-seed450K ablation if approximately10 T4 training hours are acceptable. The pilots do not establish useful development performance or a V2 advantage. Reserved games remain untouched and unconfirmed.

## Measured training

| Metric | V0 seed42 | V2 seed42 |
|---|---:|---:|
| Selected actions | 75,000 | 75,000 |
| Actions per training game | 25,000 | 25,000 |
| Gradient updates | 14,801 | 14,801 |
| Wall seconds including checkpoints/Drive | 339.16 | 411.26 |
| Actions/second including persistence | 221.13 | 182.37 |
| Checkpoint and persistence seconds | 9.27 | 24.24 |
| Last25K actions/second | 204.23 | 173.86 |
| Training level completions | 1 | 0 |
| Full training-game wins | 0 | 0 |

Each game received exactly25K actions in declared round-robin order. Losses/rewards were finite, with no artificial truncations. Both online/target checkpoints load on CPU, match their SHA256 receipts and have actual Adam step14,801. All bundled source hashes match the repaired integrated working source. Strict deterministic training remained enabled. ZIP CRC validation passed. Raw150K training rows and27 evaluation trajectories were inspected.

V0 completed the first ls20 level at global action33,982 (ls20 action11,328); no other training level was completed. Epsilon was still high, so one completion cannot be attributed to learned competence. V2 task reward was zero on every game despite approximately302.94 total intrinsic reward. Intrinsic reward is an exploration signal, not task success. Loss magnitude differences do not demonstrate which variant learned better.

Periodic GPU utilization samples averaged approximately36.95% for V0 and46.35% for V2; these are instantaneous samples, not continuous time-weighted utilization. Peak allocated CUDA memory was approximately133MiB and430MiB. These runs do not justify moving to a larger GPU on memory grounds.

## Actual development evaluation

Each model: ft09/vc33/cd82 x evaluation seeds0/1/2 x1000 selected actions, frozen greedy evaluation through Task3. All18 learned-policy runs scored0, completed0 levels, and won0 games. One-scorecard, zero-illegal-action, zero-unexpected-full-reset and official/recomputed-score invariants passed. All three evaluation seeds produced the same actions for each model/game, so they add no independent training replication.

Both policies mainly repeated fixed click coordinates. V2 repeated one coordinate for all1000 actions on each development game. V0 did likewise on ft09/cd82 and used two coordinates on vc33. The behavior is not evidence of solving these tasks.

The matched random reference was not entirely zero: seed2 completed one ft09 level (official score0.0237401) and one cd82 level (score0.0382093); no full game wins. All other random evaluations scored0. Mean official random score across the nine runs was0.00688327 on the toolkit's0-to100 scale. These small outcomes do not establish a statistically reliable ranking, especially with one trained seed per learned variant.

## Refined cost

Linear extrapolation of pilot averages gives V0 about33.9min per450K run and V2 about41.1min. Approximating V1 by V0 and history-only by V2 gives7.50 T4 hours across all12 runs; plus25% reserve gives9.38h. Using the last25K rates instead gives7.99h, or9.98h with25% reserve. Budget approximately10 training hours plus evaluation rather than promising exactly7.5h. The pilots ended at epsilon0.55; increased greedy inference, deeper game states and session/Drive variability can change cost.

The18 learned-policy evaluation loops took51.31s. Scaling those loops to all648 DQN checkpoint/game/seed evaluations gives30.8min under similar CUDA evaluation conditions, excluding setup/loading/persistence and other agents or heldout work. This supersedes the earlier broad CPU-based evaluation allowance only for similar CUDA execution; it is not a guaranteed runtime.

## Before full matrix

Keep the existing protocol/source archive fixed. The primary ablation needs all four variants and all three independent training seeds; do not promote seed42 pilot results to a complete panel or tune hypotheses because scores are zero. Use the gated runner against the ORIGINAL complete batch on Drive/Colab, which includes resume state. This compact evidence ZIP intentionally omits replay and complete continuation generations, so it cannot itself resume training or satisfy the full checkpoint gate. Colab receipts say those generations validated there; their payloads have not been independently rechecked from this ZIP. The separate CUDA reconstruction report is also excluded, although the unchanged runner requires its successful preflight before starting pilots.

Keep sb26/sk48/su15 reserved until every teammate confirms no prior exposure and the final evaluation protocol is frozen. Preserve the old3K/10K experiments separately. PPO remains a differently trained system with503,808 final training actions, WIN-only reward and random click coordinates; the new450K DQN matrix will not by itself isolate algorithm effects.
