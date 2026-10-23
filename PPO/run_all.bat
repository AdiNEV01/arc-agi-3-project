@echo off
echo ========================================================
echo Starting Full ARC-AGI-3 PPO Experimental Pipeline
echo ========================================================

:: --- 1. BASELINE ---
echo.
echo [1/5] Training Baseline (3 Seeds)...
python run_experiments.py --exp baseline --seed 0 --steps 500000
python run_experiments.py --exp baseline --seed 1 --steps 500000
python run_experiments.py --exp baseline --seed 2 --steps 500000

:: --- 2. ABLATION: MEMORY ONLY ---
echo.
echo [2/5] Training Ablation: Memory Only (3 Seeds)...
python run_experiments.py --exp memory_only --seed 0 --steps 500000
python run_experiments.py --exp memory_only --seed 1 --steps 500000
python run_experiments.py --exp memory_only --seed 2 --steps 500000

:: --- 3. IMPROVEMENT 1: CURIOSITY ONLY ---
echo.
echo [3/5] Training Improvement 1: Curiosity Only (3 Seeds)...
python run_experiments.py --exp curiosity --seed 0 --steps 500000
python run_experiments.py --exp curiosity --seed 1 --steps 500000
python run_experiments.py --exp curiosity --seed 2 --steps 500000

:: --- 4. IMPROVEMENT 2: MEMORY + CURIOSITY ---
echo.
echo [4/5] Training Improvement 2: Memory + Curiosity (3 Seeds)...
python run_experiments.py --exp memory_curiosity --seed 0 --steps 500000
python run_experiments.py --exp memory_curiosity --seed 1 --steps 500000
python run_experiments.py --exp memory_curiosity --seed 2 --steps 500000

:: --- 5. EVALUATION AND PLOTTING ---
echo.
echo [5/5] 
echo Evaluating all checkpoints...
python evaluate.py

echo Generating plots and tables...
python plot_results.py

echo Recording trajectories...
python record_trajectories.py

echo.
echo ALL DONE
pause