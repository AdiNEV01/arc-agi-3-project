# PPO improvements: beginner guide

Keep `ppo_baseline.py` as the reference implementation. The two improvement
experiments below are separate files so they can be trained and evaluated one
at a time.

## Improvement 1: `novelty_ppo.py`

**Limitation:** the baseline normally receives a reward only when it completes
a level. Until then, it may get a long sequence of zero rewards.

**Change:** add a small count-based bonus for visible board observations that
have been visited less often. The bonus count uses the observation, not hidden
game state.

**Hypothesis:** more varied exploration may help PPO discover actions that lead
to level completion.

**Risk:** the agent may explore without solving the task. The novelty reward is
a training signal, not the official ARC score.

Check the script first:

```bat
python novelty_ppo.py --check-only --device cuda
```

Run a short pilot:

```bat
python novelty_ppo.py --device cuda --timesteps 10000 --seed 0 --output-dir outputs/pilot_novelty_s0
```

The bonus strength can be changed with `--novelty-scale`. Change only this value
in a separate experiment if you want to test different bonus strengths.

## Improvement 2: `grid_ppo.py`

**Limitation:** the standard observation has one numeric channel, so a CNN can
treat colour IDs as if they were ordered brightness values.

**Change:** turn the visible colour IDs into 16 channels (one channel per colour)
and use `GridCNNFeaturesExtractor`, which uses small convolution filters and
preserves more local grid structure early in the network.

**Hypothesis:** a colour-aware representation may help PPO recognise small tiles
and spatial patterns more usefully.

**Risk:** more input channels increase computation, and a better representation
does not guarantee that PPO discovers the rules of a game.

Check the script first:

```bat
python grid_ppo.py --check-only --device cuda
```

Run a short pilot:

```bat
python grid_ppo.py --device cuda --timesteps 10000 --seed 0 --output-dir outputs/pilot_grid_s0
```

**Important:** the grid model must be evaluated with the same `grid` observation
encoding used during training. Use `--observation-mode grid` with both
`evaluate_simple.py` and `record_episode.py` for this model.

## Compare fairly

Run the baseline and both improvements with the same game IDs, seeds, number of
interactions, evaluation episodes, and maximum episode length. For an initial
10,000-interaction pilot, for example:

```bat
python ppo_baseline.py --version baseline --device cuda --timesteps 10000 --seed 0 --output-dir outputs/pilot_baseline_s0
python novelty_ppo.py --device cuda --timesteps 10000 --seed 0 --output-dir outputs/pilot_novelty_s0
python grid_ppo.py --device cuda --timesteps 10000 --seed 0 --output-dir outputs/pilot_grid_s0
```

Evaluate each using the same seeds:

```bat
python evaluate_simple.py --policy ppo --model outputs/pilot_baseline_s0/ppo_final.zip --seeds 0 1 2 --episodes-per-seed 1 --tag baseline_pilot
python evaluate_simple.py --policy ppo --model outputs/pilot_novelty_s0/ppo_final.zip --seeds 0 1 2 --episodes-per-seed 1 --tag novelty_pilot
python evaluate_simple.py --policy ppo --model outputs/pilot_grid_s0/ppo_final.zip --observation-mode grid --seeds 0 1 2 --episodes-per-seed 1 --tag grid_pilot
```

A 10,000-step pilot is only a pipeline and early-learning check. If the pilot
shows useful signs, repeat with a longer budget and multiple seeds. Compare
completed levels and successful episodes, not only the PPO training reward. The
official ARC score must be reported separately using the official scoring
procedure.
