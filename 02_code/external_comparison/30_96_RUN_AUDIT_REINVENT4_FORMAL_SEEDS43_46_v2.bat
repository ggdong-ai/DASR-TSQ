@echo off
setlocal EnableExtensions
cd /d D:\pycharm\project\tsq_mcts_rebuild

set "ROOT=D:\pycharm\project\tsq_mcts_rebuild"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "CODE=%ROOT%\02_code\external_comparison"

if not exist "%PY%" (
  echo ERROR: project Python not found: %PY%
  exit /b 1
)

echo ================================================================
echo REINVENT4 FORMAL AUDIT seeds 43-46
ECHO Uses the same corrected v2 audit logic already passed by seed42.
echo ================================================================

for %%S in (43 44 45 46) do (
  echo.
  echo ----------------------------------------------------------------
  echo Auditing seed %%S
  echo ----------------------------------------------------------------
  "%PY%" "%CODE%\30_96_audit_reinvent4_formal_seed%%S_v2.py"
  if errorlevel 1 (
    echo.
    echo ERROR: formal audit failed for seed %%S.
    echo Stop here. Do NOT rerun the REINVENT4 engine.
    exit /b 10
  )
)

echo.
echo ================================================================
echo ALL REINVENT4 FORMAL AUDITS 43-46 PASS
ECHO Ready for 5-seed aggregation.
echo ================================================================
exit /b 0
