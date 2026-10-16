@echo off
setlocal EnableExtensions
cd /d D:\pycharm\project\tsq_mcts_rebuild

set "ROOT=D:\pycharm\project\tsq_mcts_rebuild"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "CODE=%ROOT%\02_code\external_comparison"

call "%CODE%\30_96_RUN_AUDIT_REINVENT4_FORMAL_SEEDS43_46_v2.bat"
if errorlevel 1 exit /b %errorlevel%

echo.
echo ================================================================
echo Aggregating audited REINVENT4 seeds 42-46
ECHO ================================================================
"%PY%" "%CODE%\30_97_aggregate_reinvent4_formal_5seeds_v1.py"
if errorlevel 1 (
  echo ERROR: 5-seed aggregation failed.
  exit /b 20
)

echo.
echo REINVENT4 5-seed formal package COMPLETE.
exit /b 0
