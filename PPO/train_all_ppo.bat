@echo off
setlocal

REM Run from the PPO project folder with the project's .venv activated.
REM Trains baseline, novelty, and grid PPO for seeds 0, 1, and 2.
REM Each run requests 500,000 environment interactions.

for %%S in (0 1 2) do (
    echo.
    echo ============================================================
    echo SEED %%S - BASELINE PPO - 500,000 steps
    echo ============================================================
    python ppo_baseline.py --version baseline --device cuda --timesteps 500000 --seed %%S --output-dir outputs\baseline_s%%S
    if errorlevel 1 goto :failed

    echo.
    echo ============================================================
    echo SEED %%S - NOVELTY PPO - 500,000 steps
    echo ============================================================
    python novelty_ppo.py --device cuda --timesteps 500000 --seed %%S --output-dir outputs\novelty_s%%S
    if errorlevel 1 goto :failed

    echo.
    echo ============================================================
    echo SEED %%S - GRID CNN PPO - 500,000 steps
    echo ============================================================
    python grid_ppo.py --device cuda --timesteps 500000 --seed %%S --output-dir outputs\grid_s%%S
    if errorlevel 1 goto :failed
)

echo.
echo ============================================================
echo ALL NINE TRAINING RUNS COMPLETED
echo Results and checkpoints are in the outputs folder.
echo ============================================================
exit /b 0

:failed
echo.
echo A training run failed. The batch file stopped to avoid silently skipping an experiment.
echo Read the error shown above, fix it, and rerun only the failed experiment using its command.
exit /b 1
