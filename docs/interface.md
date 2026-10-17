# ARC-AGI-3 interface, adapter and evaluation protocol (Task 3)

Status: established from the official documentation and the package source code
(arc-agi 0.9.9, arcengine 0.9.3, Python 3.12). Items marked **[verify]** must be
confirmed on real games with `python test_env.py --game <id>`; paste the
results into the "Probe results" section at the end.

Sources: docs.arcprize.org (Game Schema, Actions, Scoring Methodology,
Competition Mode, EnvironmentWrapper, ARC Prize 2026 starter page) and the
installed source files `arcengine/enums.py`, `arcengine/base_game.py`,
`arc_agi/base.py`, `arc_agi/local_wrapper.py`, `arc_agi/scorecard.py`.

## A. Native interface

### A1. Observation format
* Every call (`env.reset()`, `env.step()`) returns a `FrameDataRaw` with:
  `game_id`, `state`, `levels_completed`, `win_levels`, `available_actions`
  (list of ints), `action_input` (the action that produced it), `guid`,
  `full_reset` (bool), and `frame`.
* `frame` is a **list** of 2D integer grids, not one grid: a single action can
  produce several animation frames. The last frame is the settled state.
* Grids are at most 64x64, cell values 0 to 15 (colours), origin (0,0) at the
  top left, coordinates given as (x, y).
* No text, no goal description, no reward. The agent must infer the rules and
  the objective from the grids alone.
* When the state is WIN or GAME_OVER, a further non-RESET action returns an
  empty frame list.
* **[verify]** typical frame size per game, frames per action, and whether the
  grid size changes between levels (the engine resizes the camera per level).

### A2. Action space
* `RESET` plus `ACTION1` to `ACTION7`.
  ACTION1 to ACTION4 are conventionally up/down/left/right, ACTION5 is a
  game-specific interaction, ACTION6 is a click with `x`, `y` in 0 to 63,
  ACTION7 is undo where supported.
* Each game exposes a subset; the subset is reported in every frame's
  `available_actions`. For ACTION6 only its availability is given, never which
  coordinates are meaningful.
* After GAME_OVER the only valid action is RESET.

### A3. Reset behaviour (from `base_game.handle_reset`)
* The first RESET starts the game.
* A RESET sent after at least one action in the current level is a **level
  reset**: the current level restarts, completed levels are kept.
* A RESET sent when no action has been taken since the last level change or
  reset (for example two RESETs in a row), or after WIN, is a **full reset**:
  back to level 1 with `levels_completed = 0`. The response has
  `full_reset = True`.
* With environment variable `ONLY_RESET_LEVELS=true`, every RESET before WIN is
  a level reset, including the first one, which then never opens a scorecard
  play (verified). Kaggle competition mode turns game resets into level resets
  on the server side; locally we reproduce this by never sending a second
  consecutive RESET.
* Each level RESET is counted as one action by the scorecard (verified in
  `arc_agi/scorecard.py`: `inc_reset_count` also increments the action count),
  so dying and resetting costs efficiency. The very first RESET that starts
  the game is not counted. `Arcade.make()` sends that first RESET itself.

### A4. Level progression
* Each game has `win_levels` levels of increasing difficulty.
  `levels_completed` increments when a level is solved and the next level
  loads automatically in the same response sequence.
* The game state becomes WIN when the final level is completed.

### A5. Feedback
* There is no reward signal. The only feedback is the next frame(s), the
  change in `levels_completed`, the `state`, and `available_actions`.

### A6. Termination
* `WIN`: all levels complete. `GAME_OVER`: the run failed (game-specific
  condition); only RESET is then valid.
* `NOT_FINISHED`: game active. `NOT_PLAYED`: not started.
* There is no built-in action limit in the local engine; any limit is ours
  (the evaluation budget below).

### A7. Scoring (RHAE, Relative Human Action Efficiency)
* Per completed level: `(human_baseline_actions / ai_actions)^2`, capped at
  1.15 (115 on the toolkit's 0 to 100 scale). Uncompleted levels score 0.
* Game score: average of level scores weighted by the 1-indexed level number,
  capped by the share of weight in completed levels.
* Total score: mean of game scores, 0 to 100%.
* Baseline: the upper median of first-time human players per level, stored in
  `EnvironmentInfo.baseline_actions`.
* Actions per level are counted across level resets, and each level reset
  itself costs one action (failed attempts are expensive).
* In local mode, a full reset opens a new run on the scorecard and the game
  score is the **maximum** over runs. This is why our evaluator forces
  level-only resets.

### A8. Competition rules that affect design
* Kaggle runs in competition mode: interaction through the API, each
  environment may be made **once**, one scorecard, no score readout while
  running, level resets only, scoring over all environments even if skipped.
* Kaggle evaluates a hidden game set; local runs only see public games.
  Local scores and Kaggle scores must be reported separately.
* Accelerated Kaggle sessions have no internet, so checkpoints and
  dependencies must be bundled with the notebook.
* The submission is the starter kit's `agent/my_agent.py`
  (`choose_action(frames, latest_frame)`); `agents/kaggle_bridge.py` connects
  our policies to it with the same encoding as training.

## B. Gymnasium adapter (`arc_env/gym_adapter.py`)

| Element | Choice | Reason |
|---|---|---|
| Observation | `Box(64,64)` uint8, last frame, value 16 = padding | Settled state after the action; fixed shape for CNNs |
| Action | `Discrete(70)`: 6 simple actions + ACTION6 on an 8x8 click lattice | Makes clicking learnable by DQN/PPO; `click_grid` configurable |
| RESET | Not an agent action; issued by `reset()` | Avoids agents learning to reset-spam |
| Legal actions | `action_masks()` from `available_actions` | Works with MaskablePPO; DQN can mask Q-values |
| Illegal action | Not sent, obs unchanged, `info["illegal_action"]=True` | Does not consume official actions |
| Reward | +1 per newly completed level (training only) | Only native progress signal; shaping via `reward_fn` |
| terminated | WIN or GAME_OVER | Native terminal states |
| truncated | `max_episode_actions` reached | Training episode limit |
| reset_mode | `level` (default, competition-legal) or `full` (training) | Full restarts allowed for training only |

Limitations to state in the report: discarding intermediate animation frames
loses motion information; the click lattice cannot hit every pixel; the
observation is not Markov (history is needed for many games), so agents that
need memory must keep it themselves.

## C. Reference policy (`agents/random_agent.py`)
Uniform over legal action types, then a uniform click cell for ACTION6. This
avoids the 64 click targets dominating a flat uniform choice. Seeded per game.

## D. Evaluation protocol (`evaluate.py`, `Configs/eval.yaml`)
* Same script, same games, same seeds, same budget for every agent version.
* One scorecard and one `make` per (agent, seed, game). `make()` starts the game;
  afterwards RESET is sent only after GAME_OVER, so every later reset is a level
  reset (competition semantics). Verified per run: `full_resets_after_start == 0`
  and `scorecard_runs == 1`. `ONLY_RESET_LEVELS` must NOT be set: it stops the
  scorecard from opening a play, leaving the official score empty.
* Budget: `max_actions_per_game` agent actions (ACTION1 to ACTION7). Scored actions
  (agent actions plus level resets) are recorded separately.
  A run stops at WIN, budget, or optional wall-clock limit.
* After GAME_OVER the evaluator sends a level reset and continues.
* Seeds: at least 3. Report mean, standard deviation and individual seeds.
* Game splits (train / dev / heldout) are fixed once in `Configs/eval.yaml`.
  Held-out games are used only for final numbers.

Recorded per (agent, seed, game): agent, agent_version, checkpoint_step,
split, game_id (with version), seed, game_seed, success, levels_completed,
win_levels, official score, recomputed score, official_actions, budget,
per-level actions and human baselines, level resets, game overs, illegal
actions, stop reason, runtime, actions per second, scorecard run count,
trajectory file. `run_meta.json` stores package versions, hardware, git commit
and the full config.

## Probe results (17 Oct 2026, arc-agi 0.9.9, arcengine 0.9.3, laptop run)

25 public games were available (`python test_env.py --list`). Tags: 4 keyboard,
7 click, 13 keyboard_click, 1 untagged. Four TRAIN games were probed with 300
random actions each (`test_env.py`, seed 0):

| Game | Tag | Levels | Frame | Actions | Frames per action | Frame changed | GAME_OVER | Levels by random |
|---|---|---|---|---|---|---|---|---|
| tr87 | keyboard | 6 | 64x64 int8 | 1, 2, 3, 4 | 1 | 100% | 2 | 0 |
| lp85 | click | 8 | 64x64 int8 | 6 | 1 | 1% | 0 | 0 |
| sc25 | keyboard_click | 6 | 64x64 int8 | 1, 2, 3, 4, 6 | 1, 13 or 22 | 72% | 4 | 0 |
| ft09 | untagged | 6 | 64x64 int8 | 6 | 1 or 5 | 7% | 0 | 0 |

Reset semantics confirmed on all four games: RESET after an action gives
`full_reset = False` (level reset); a second consecutive RESET gives
`full_reset = True` (full reset).

Findings and their consequences:
1. **All frames are 64x64.** Padding (value 16) is a safeguard, not routine.
2. **ft09 is untagged but click-only**, so the split has 3 click games in train.
3. **Multi-frame actions exist** (sc25 up to 22 frames, ft09 5 frames). Keeping only
   the last frame discards animation, which may carry information. This is a
   documented limitation and a possible improvement direction (stacking frames).
4. **Click games are very sparse.** Only 1% (lp85) and 7% (ft09) of uniformly random
   clicks changed the settled frame. Most of the click space does nothing, so
   exploration over ACTION6 is the main difficulty in click games. Measured
   lattice coverage: see `probe_clicks.py` results below.
5. **GAME_OVER is reachable by random play** in keyboard and mixed games, so
   level resets occur during evaluation and their action cost matters for RHAE.
6. **Random play completed no level in 300 actions on any probed game.** The
   +1-per-level training reward will almost never be observed early in training.
   Both DQN and PPO baselines are expected to face an extremely sparse reward;
   this is the obvious motivation for exploration or shaping improvements.

### Click coverage (`python probe_clicks.py --game <id>`)
| Game | Responsive pixels | Regions | 4x4 | 8x8 | 16x16 | 32x32 |
|---|---|---|---|---|---|---|
| lp85 | 96 (2.3%) | 2 | 0% | 100% | 100% | 100% |
| ft09 | 288 (7.0%) | 8 | 50% | 100% | 100% | 100% |

Decision: click_grid = 8 (64 targets), the smallest lattice with full coverage on level 1 of both click-only train games. Limitation: level 1 only; regions average about 6x6 pixels, so smaller targets on later levels could fall between lattice points.

## Leakage controls
* `arc.make()` downloads each game's Python source to `environment_files/`.
  That code reveals rules and win conditions (hidden game state). It is
  git-ignored, is not read when designing agents, and no agent imports it.
* Agents receive only the observation grid, legal actions, state and level count.
  Human baselines and raw frames are logged for analysis but never given to agents.
* Held-out games are not probed, played or trained on before final evaluation.
