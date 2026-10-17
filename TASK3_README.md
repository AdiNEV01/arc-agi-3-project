# Task 3: interface, adapter, reference agent, evaluation

## Setup (Ubuntu, conda)
    conda create -n arc3 python=3.12 -y && conda activate arc3
    pip install -r requirements.txt

## Order of use
1. `python test_env.py --list` and copy game ids into `Configs/eval.yaml`
   (split into train / dev / heldout as a team, then freeze).
2. `python test_env.py --game <id>` for a few games; fill the probe table in
   `docs/interface.md`.
3. `python -m pytest tests -q` (offline tests, no games needed).
4. `python probe_clicks.py --game lp85` and `--game ft09` (click coverage).
5. `python evaluate.py --agent random --split dev --save-traj` (sanity baseline).
   Add `--save-recording` for official replays with frames (Task 4 cases).

Never probe, play or train on held-out games. Never open environment_files/.

## For Task 1 (DQN) and Task 2 (PPO)
* Train with `arc_env.gym_adapter.ArcGymEnv`. Use `reset_mode="full"` if you
  want restarts during training; evaluation always uses level resets.
* Make your agent a subclass of `agents.base.Policy` whose constructor accepts
  `click_grid` and `seed` (plus your own kwargs, e.g. `checkpoint`).
* Evaluate only with `evaluate.py`, e.g.
      python evaluate.py --agent agents.ppo_agent:PPOPolicy \
        --agent-kwargs '{"checkpoint": "results/checkpoints/ppo_base.zip"}' \
        --agent-version ppo-base --checkpoint-step 200000 --split dev
* For learning curves, evaluate saved checkpoints at fixed interaction counts
  and pass `--checkpoint-step`.
* Reward shaping goes in `reward_fn`; the reported score is always the RHAE
  score from `evaluate.py`, never training return.
