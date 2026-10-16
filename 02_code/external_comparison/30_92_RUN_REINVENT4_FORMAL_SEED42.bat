@echo off
setlocal
cd /d D:\pycharm\project\tsq_mcts_rebuild

if not exist ".venv_reinvent4\Scripts\reinvent.exe" (
  echo ERROR: .venv_reinvent4 REINVENT executable not found.
  exit /b 1
)

set "PYTHONPATH=D:\pycharm\project\tsq_mcts_rebuild\02_code\external_comparison\reinvent4_plugin;%PYTHONPATH%"
set "OUT=D:\pycharm\project\tsq_mcts_rebuild\06_paper\experiment_records\external_benchmark_v4_reinvent4_formal\seed_42"

if exist "%OUT%\TASK_COMPLETED.json" (
  echo ERROR: seed42 already has TASK_COMPLETED.json. Do not rerun.
  exit /b 2
)

if not exist "%OUT%" mkdir "%OUT%"

echo ================================================================
echo REINVENT4 FORMAL seed42
echo Protocol must already be frozen by 30_91.
echo ================================================================

".venv_reinvent4\Scripts\python.exe" ".\02_code\external_comparison\30_91_freeze_reinvent4_formal_v1.py"
if errorlevel 1 exit /b %errorlevel%

".venv_reinvent4\Scripts\reinvent.exe" --seed 42 ^
  -l "%OUT%\reinvent4_formal.log" ^
  ".\02_code\external_comparison\reinvent4_formal_seed42_v1.toml"

if errorlevel 1 (
  echo.
  echo REINVENT4 formal seed42 exited with an error.
  echo Do NOT delete or rerun the directory before audit.
  exit /b %errorlevel%
)

echo.
echo REINVENT4 seed42 engine run finished.
echo Do NOT run seeds 43-46 yet. Audit seed42 first.
endlocal
