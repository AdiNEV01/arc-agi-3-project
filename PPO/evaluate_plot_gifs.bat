@echo off
setlocal EnableExtensions

REM Run this file from the PPO project folder after all 9 training runs finish.
REM It evaluates using new game seeds (10, 11, 12), then makes tables, plots,
REM per-run training charts, and GIF replays. Grid models use grid observations.

cd /d "%~dp0"

set "EVAL_SEEDS=10 11 12"
set "EVAL_ROOT=outputs\final_evaluations"
set "ANALYSIS_ROOT=outputs\analysis_final"
set "REPLAY_ROOT=outputs\replays_final_%RANDOM%_%RANDOM%"

echo ============================================================
echo STEP 1 - CHECK THAT ALL NINE FINAL MODELS EXIST
echo ============================================================
for %%S in (0 1 2) do (
    for %%V in (baseline novelty grid) do (
        if not exist "outputs\%%V_s%%S\ppo_final.zip" (
            echo ERROR: Missing outputs\%%V_s%%S\ppo_final.zip
            echo Complete or repair that training run before evaluating.
            exit /b 2
        )
    )
)

echo.
echo ============================================================
echo STEP 2 - EVALUATE A RANDOM-ACTION REFERENCE
echo ============================================================
python evaluate_simple.py --policy random --seeds 10 11 12 --episodes-per-seed 1 --tag random_reference --output-root "%EVAL_ROOT%"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo STEP 3 - EVALUATE ALL TRAINED MODELS ON THE SAME GAME SEEDS
echo ============================================================
for %%S in (0 1 2) do (
    echo Evaluating baseline training seed %%S...
    python evaluate_simple.py --policy ppo --model "outputs\baseline_s%%S\ppo_final.zip" --seeds 10 11 12 --episodes-per-seed 1 --tag baseline_s%%S --output-root "%EVAL_ROOT%"
    if errorlevel 1 goto :failed

    echo Evaluating novelty training seed %%S...
    python evaluate_simple.py --policy ppo --model "outputs\novelty_s%%S\ppo_final.zip" --seeds 10 11 12 --episodes-per-seed 1 --tag novelty_s%%S --output-root "%EVAL_ROOT%"
    if errorlevel 1 goto :failed

    echo Evaluating grid training seed %%S...
    python evaluate_simple.py --policy ppo --model "outputs\grid_s%%S\ppo_final.zip" --observation-mode grid --seeds 10 11 12 --episodes-per-seed 1 --tag grid_s%%S --output-root "%EVAL_ROOT%"
    if errorlevel 1 goto :failed
)

echo.
echo ============================================================
echo STEP 4 - MAKE EVALUATION TABLES AND COMPARISON PLOTS
echo ============================================================
python analyze_results.py --input-root "%EVAL_ROOT%" --output-dir "%ANALYSIS_ROOT%"
if errorlevel 1 goto :failed

python plot_training_comparison.py --input-root outputs --output "%ANALYSIS_ROOT%\training_comparison.png"
if errorlevel 1 goto :failed

for %%S in (0 1 2) do (
    python plot_training.py --run-dir "outputs\baseline_s%%S"
    if errorlevel 1 goto :failed
    python plot_training.py --run-dir "outputs\novelty_s%%S"
    if errorlevel 1 goto :failed
    python plot_training.py --run-dir "outputs\grid_s%%S"
    if errorlevel 1 goto :failed
)

echo.
echo ============================================================
echo STEP 5 - RECORD RANDOM AND TRAINED TRAJECTORIES AS GIFS
echo ============================================================
python record_episode.py --policy random --seed 10 --output-dir "%REPLAY_ROOT%\random_seed10"
if errorlevel 1 goto :failed

for %%S in (0 1 2) do (
    python record_episode.py --policy ppo --model "outputs\baseline_s%%S\ppo_final.zip" --seed 10 --output-dir "%REPLAY_ROOT%\baseline_s%%S_seed10"
    if errorlevel 1 goto :failed

    python record_episode.py --policy ppo --model "outputs\novelty_s%%S\ppo_final.zip" --seed 10 --output-dir "%REPLAY_ROOT%\novelty_s%%S_seed10"
    if errorlevel 1 goto :failed

    python record_episode.py --policy ppo --model "outputs\grid_s%%S\ppo_final.zip" --observation-mode grid --seed 10 --output-dir "%REPLAY_ROOT%\grid_s%%S_seed10"
    if errorlevel 1 goto :failed
)

echo.
echo ============================================================
echo ALL EVALUATION, PLOTTING, AND REPLAY-GIF STEPS COMPLETED
echo ============================================================
echo Evaluation CSVs and summaries: %EVAL_ROOT%
echo Analysis tables and figures: %ANALYSIS_ROOT%
echo Replay GIFs and trajectories: %REPLAY_ROOT%
echo Each evaluation episode is included, including failures.
echo Local level-completion results are not the official ARC score.
exit /b 0

:failed
echo.
echo ERROR: A command failed. Read the traceback above.
echo The batch file stopped so incomplete results are not mistaken for a full analysis.
exit /b 1
