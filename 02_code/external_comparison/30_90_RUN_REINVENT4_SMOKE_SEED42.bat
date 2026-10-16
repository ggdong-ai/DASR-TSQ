@echo off
setlocal

cd /d D:\pycharm\project\tsq_mcts_rebuild

set "PLUGIN_ROOT=D:\pycharm\project\tsq_mcts_rebuild\02_code\external_comparison\reinvent4_plugin"
set "PYTHONPATH=%PLUGIN_ROOT%;%PYTHONPATH%"

if not exist "06_paper\experiment_records\external_benchmark_v4_reinvent4_preflight\seed_42" (
    mkdir "06_paper\experiment_records\external_benchmark_v4_reinvent4_preflight\seed_42"
)

echo ============================================================
echo REINVENT4 preflight
echo ============================================================
python .\02_code\external_comparison\30_89_preflight_reinvent4_v1.py
if errorlevel 1 (
    echo PRELIGHT FAILED. Smoke not started.
    exit /b 1
)

echo ============================================================
echo REINVENT4 Mol2Mol seed42 engineering smoke
echo ============================================================
reinvent --seed 42 ^
  -l "D:\pycharm\project\tsq_mcts_rebuild\06_paper\experiment_records\external_benchmark_v4_reinvent4_preflight\seed_42\reinvent4_smoke.log" ^
  ".\02_code\external_comparison\reinvent4_smoke_seed42_v1.toml"

if errorlevel 1 (
    echo REINVENT4 SMOKE FAILED.
    exit /b 1
)

echo ============================================================
echo REINVENT4 SMOKE COMMAND COMPLETED
echo Inspect oracle_state.json, reinvent4_smoke_1.csv and log before freezing.
echo ============================================================
endlocal
