# Task 1: Deep Q-learning for ARC-AGI-3

## Scope and experimental question

This contribution implements a Deep Q-Network (DQN) baseline and evaluates two changes: count-based visual novelty and short observation history. A matched 2 × 2 study separates their effects. Each configuration was trained on **ls20-9607627b** for 3,000 selected actions with training seeds 42, 123 and 456. This is a small-budget, single-game study, not evidence of general ARC-AGI-3 competence.

## Baseline design

The baseline observes the most recent 64 × 64 grid. Each integer color, from 0 to 15, is represented by a one-hot channel, producing a tensor of shape `[B,16,64,64]`. Color indices are categorical rather than scalar intensities.

The global action representation contains six ordinary actions and 4,096 spatial actions, for **4,102 Q-values**. A click at `(x,y)` has index `6 + 64y + x`. RESET belongs to the environment controller and is not selected by the policy. Legal-action masking restricts greedy selection and random exploration to the actions exposed by the game. In ls20, only ACTION1–ACTION4 are available; the click outputs are unused.

The CNN has three stride-one, padding-one 3 × 3 convolutions with channel widths 32, 32 and 64, each followed by ReLU. A discrete head pools the feature map to 4 × 4, flattens it and applies a 1,024 → 128 → 6 multilayer perceptron. A separate 1 × 1 convolution produces 64 × 64 spatial Q-values. This is a shared visual encoder with discrete and spatial output heads, not a dueling value/advantage architecture.

Learning follows the replay and target-network approach of [Mnih et al. (2015)](https://doi.org/10.1038/nature14236). For transition `(s_t,a_t,r_t,s_{t+1},d_t)`, the implemented target is

$$
y_t=r_t+\gamma(1-d_t)\max_{a'\in\mathcal A(s_{t+1})}Q_{\theta^-}(s_{t+1},a'),
\qquad
\mathcal L(\theta)=\frac{1}{B}\sum_i\operatorname{Huber}\!\left(Q_\theta(s_i,a_i)-y_i\right).
$$

Terminal or empty-mask next states have zero bootstrap value. Replay stores copied uint8 grids, rewards, terminal flags and next-action masks; one-hot conversion occurs when sampling. Uniform replay capacity is 3,000, batch size is eight, Adam learning rate is 0.0001, discount is 0.99, and gradient norm is clipped to 10. The target network is copied every 250 gradient updates. Learning occurs every five selected actions after warmup 64, giving 588 updates in each full run. This is ordinary DQN, not Double DQN.

The epsilon-greedy policy samples uniformly from legal action indices with probability

$$\epsilon_t=1.0+(0.1-1.0)\min((t-1)/2000,1).$$

Sparse task reward is the positive change in the environment's completed-level counter:

$$r_t^{\mathrm{task}}=\max(0,L_{t+1}-L_t).$$

Training maximum progress is reported as `max_t L_t`, not the sum of the same counter over observations. Summed positive task-reward deltas are reported separately and are not interpreted as a count of distinct levels across independent games.

## Improvement 1: Count-based observation novelty

With little task reward, most baseline transitions do not distinguish useful actions. The first hypothesis was that rewarding relatively unfamiliar observations would improve data collection and increase the chance of finding level progress. Count-based intrinsic rewards are motivated by exploration research such as [Bellemare et al. (2016)](https://arxiv.org/abs/1606.01868). This implementation uses exact visual hashes, not that paper's learned density-model pseudo-counts.

Let `h(o)` be a 128-bit BLAKE2b hash of the latest grid, and let `N_t(h)` be the visit count after registering the new observation. Then

$$
r_t^{\mathrm{int}}=
\begin{cases}
0,&h(o_{t+1})=h(o_t),\\
\beta/\sqrt{N_t(h(o_{t+1}))},&\text{otherwise},
\end{cases}
\qquad r_t=r_t^{\mathrm{task}}+r_t^{\mathrm{int}},\quad\beta=0.01.
$$

Counts persist across training resets; reset observations are counted. The hash depends on the latest grid, not hidden game state or the four-frame history. Visually different observations can result from movement, animation or interface changes and need not represent useful exploration. V0 has no saved observation stream from which to reconstruct a comparable unique-frame count; that metric is unavailable for V0 rather than zero.

V1 seed 123 reached one level during exploratory training, at selected action **{{V1_REACH}}**; the other two V1 runs reached none. Its frozen policy subsequently completed zero levels. This isolated event supports only the statement that progress was encountered in one training trajectory; it does not establish a reliable novelty benefit.

## Improvement 2: Four-frame observation history

A single grid may omit recent movement or temporal changes. The second hypothesis was that the most recent four **decision-time** observations would provide useful short-term context. The state is

$$s_t=[o_{t-3},o_{t-2},o_{t-1},o_t].$$

At reset, the first grid is repeated four times. Each environment action advances history once, including exploratory actions; intermediate rendering frames returned within a single action are not four separate decisions. One-hot encoding produces `[B,4,16,64,64]`, which is reshaped in frame-major order to `[B,64,64,64]`. Only the first convolution's input width changes. Replay stores independent copies of both histories before any controller reset. Terminal transitions are stored before clearing history. A nonterminal level advance remains part of the observation sequence until an explicit reset.

V2 combines this representation with V1 novelty. Frozen evaluation reads `num_frames` from checkpoint configuration and uses the same history mechanism; older V0/V1 checkpoints default to one frame. Regression checks loaded all nine original checkpoints. Four-frame models have {{PARAM4}} parameters compared with {{PARAM1}} for one-frame models: the history factor therefore also increases first-layer capacity and is not a parameter-matched representation experiment.

## Ablation study and matched protocol

| Configuration | Novelty bonus | Frames | CNN input channels |
|---|---:|---:|---:|
| V0 baseline | Off | 1 | 16 |
| V1 novelty | β = 0.01 | 1 | 16 |
| History-only ablation | β = 0 | 4 | 64 |
| V2 novelty + history | β = 0.01 | 4 | 64 |

The history-only variant is explicitly named `v2_history_only`. It uses the same network, history, replay and learning algorithm as V2. Count tracking remains active solely as a diagnostic, while every recorded intrinsic reward is **exactly zero**. Distinct checkpoint and log names prevent overwriting completed experiments.

All configurations share the preceding learning settings, a controller episode limit of 300, and 3,000 selected-action budgets. New CPU runs use `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. Historical runtime comparisons are not justified because earlier thread settings and execution conditions were not uniformly recorded. Existing V0/V1/V2 models were not retrained.

{{MAIN_TABLE}}

*Table 1. Main results, with seed order 42, 123, 456. Training maxima and frozen outcomes are different measurements. Each frozen evaluation uses seed 1001, at most 200 actions, epsilon zero and no updates.*

{{AGGREGATE_TABLE}}

*Table 2. Reward-component means ± sample standard deviation across three training seeds. V0 components are derived from its task-only total-reward log and disabled intrinsic mechanism; the other components are logged separately. These are descriptive summaries, not significance tests.*

![Training level progression](../../../Analysis/DQN/final_task1/figures/01_training_progression.png)

*Figure 1. Reported training progress per seed. The same completed-level counter is not repeatedly counted as new progress.*

![Frozen evaluation outcomes](../../../Analysis/DQN/final_task1/figures/02_frozen_evaluation.png)

*Figure 2. An annotated outcome matrix makes zero-completion results explicit while retaining action budgets and terminal outcomes.*

![Factorial comparison](../../../Analysis/DQN/final_task1/figures/06_factorial_ablation.png)

*Figure 3. The four novelty/history combinations. Neither factor produced demonstrated frozen-policy improvement at this budget.*

## Frozen evaluation, seeding and reset semantics

Every checkpoint was evaluated from an initial reset of a separately constructed ls20 environment. No replay collection or learning takes place during evaluation. Evaluation stops on WIN, GAME_OVER, or the 200-action budget. New evaluations additionally verify that online parameters remain identical and that the gradient-update counter does not change. Historical summaries report zero evaluation updates and were preserved rather than regenerated.

The argument `seed=1001` should not be mistaken for a randomized test instance. In the installed local wrapper, a seed is forwarded only when the game constructor accepts a `seed` parameter. Ls20's constructor accepts none, its effective base-game seed stays zero, and inspection found no random/RNG calls in its game source. Independent environments requested with seeds 1001, 1002 and 1003 produced identical initial frames and observation/state traces under 80 scripted actions. The scripted check cannot prove every possible trajectory, but the source and trace evidence show no meaningful evaluation-seed variation here. Additional model runs at these seeds were therefore not presented as independent evidence. Training seeds still affect network initialization, action exploration and replay sampling.

`env.reset()` dispatches RESET rather than necessarily creating a fresh game. After actions on the current level, the installed engine normally resets that level and retains prior progress, including after GAME_OVER. It fully resets when the current-level action counter is zero or the game is won; entering a new level clears that counter. A discarded white-box diagnostic verified both cases using simulated level advancement: an immediate reset changed progress 1 → 0, whereas a reset after an action on the new level retained 1 → 1. Simulated progress is not an agent result. `ONLY_RESET_LEVELS` was not enabled.

Accordingly, training “episodes” can be repeated level attempts rather than independent whole games. Controller RESET calls are excluded from the 3,000 selected-action budget. The 300-action limit causes a reset but its preceding transition is still bootstrapped if the observation is nonterminal. This existing truncation/reset protocol was retained for the ablation, not silently corrected. Future work should define whether evaluation units are games or level attempts, create verified fresh games for independent episodes where intended, log reset events and full-reset flags, and distinguish termination from truncation. One should not reinterpret current results under that proposed protocol.

The recorded `local_scorecard_score` is a local diagnostic. It is not an official Kaggle score. Its limited discrimination when no levels are completed is why this study also reports progress, terminal states and action counts.

## Policy behaviour and learning signals

{{BEHAVIOUR_TEXT}}

![Greedy action distributions](../../../Analysis/DQN/final_task1/figures/04_greedy_actions.png)

*Figure 4. Counts and shares of selected legal actions per frozen policy. Variation in action choice does not by itself show improved reasoning or productive exploration.*

![Training losses](../../../Analysis/DQN/final_task1/figures/03_training_loss.png)

*Figure 5. Raw Huber temporal-difference losses and 25-update moving means. A low loss can reflect fitting near-zero or repetitive bootstrapping targets without solving a level. Curves use gradient updates rather than environment actions; display values are floored at 10⁻¹².*

![Reward components](../../../Analysis/DQN/final_task1/figures/05_reward_components.png)

*Figure 6. Task and intrinsic reward are shown separately. Novelty reward does not measure actual task success.*

Recorded unique visual counts are {{UNIQUE_TEXT}}. V0 counts are unavailable. History-only retains the counting mechanism despite β=0, enabling a visual-coverage comparison with V1/V2, but interface effects, repeated level attempts, different trajectories and three seeds prevent a causal claim of useful exploration. None of the counts identifies abstract states or solved rules.

Repetitive policies are consistent with sparse reward, short training, weak state abstraction or value-estimation errors. Those mechanisms were not isolated experimentally, so they remain possible explanations. A four-frame window may still be too short for planning and cannot replace object relations, hypothesis testing or memory. A larger architecture alone also need not produce better data. The experiment demonstrates implementation and measurement, not that DQN cannot solve this game with different resources or representations.

## Supplementary zero-shot transfer check

{{CROSS_TEXT}}

## Reproducibility and uncertainty

All twelve main runs have saved checkpoints, 3,000-row training CSVs, configurations and frozen evaluation trajectories/summaries. Each has 588 gradient updates. New smoke and regression results are separate from the main analysis. Hash verification confirmed that all {{PRESERVED}} pre-existing experiment/report artifacts were unchanged. Package versions, source and game hashes, seed/reset diagnostics, raw curve data and derived tables accompany the figures. Missing historical fields remain unavailable rather than being invented. Full RNG/replay state was not saved in old checkpoints, so exact continuation of those old runs is not guaranteed. The old experiments also lack a pre-training dependency/source lock; the current manifest cannot retroactively establish their full software environment.

Only three training seeds, one deterministic within-game evaluation instance, and a small training budget are available. The study uses no significance tests or claim of equivalence. Zero observed successes does not prove a true success probability of zero. Repeated evaluations of a deterministic policy on the same unchanged game would not enlarge an independent test set. Supplementary transfer results must remain separate from within-game training results.

## Discussion and conclusion

The baseline, both substantive improvements and their history-only ablation were implemented and tested consistently. The comparison establishes that, at 3,000 interactions on ls20, **{{CONCLUSION}}** V1's single exploratory level event did not transfer to frozen success. Intrinsic rewards and changes in action distribution should therefore not be described as improved task performance.

Future experiments should first make episode/reset accounting explicit and add genuinely different game instances or games with trained action coverage. Subsequent hypotheses include object-centric state representations, history or recurrent memory matched for parameter count, better exploration abstractions that ignore irrelevant visual changes, and longer budgets with evaluation checkpoints. Each requires controlled testing; none is claimed as a demonstrated solution here.

## References

1. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533. [Article](https://doi.org/10.1038/nature14236); [author-hosted paper](https://deepmind-media.storage.googleapis.com/dqn/DQNNaturePaper.pdf).
2. Bellemare, M. G. et al. (2016). *Unifying Count-Based Exploration and Intrinsic Motivation*. Advances in Neural Information Processing Systems 29. [Paper](https://arxiv.org/abs/1606.01868).
