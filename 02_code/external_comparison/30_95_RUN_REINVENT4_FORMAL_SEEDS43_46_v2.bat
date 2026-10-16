@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d D:\pycharm\project\tsq_mcts_rebuild

set "ROOT=D:\pycharm\project\tsq_mcts_rebuild"
set "PYREINVENT=%ROOT%\.venv_reinvent4\Scripts\python.exe"
set "REINVENT=%ROOT%\.venv_reinvent4\Scripts\reinvent.exe"
set "CODE=%ROOT%\02_code\external_comparison"
set "FORMAL=%ROOT%\06_paper\experiment_records\external_benchmark_v4_reinvent4_formal"
set "PYTHONPATH=%CODE%\reinvent4_plugin;%PYTHONPATH%"

if not exist "%PYREINVENT%" (
  echo ERROR: REINVENT4 Python not found: %PYREINVENT%
  exit /b 1
)
if not exist "%REINVENT%" (
  echo ERROR: REINVENT4 executable not found: %REINVENT%
  exit /b 1
)

echo ================================================================
echo REINVENT4 FORMAL seeds 43-46
echo Scientific protocol is already frozen; no tuning is performed.
echo ================================================================

"%PYREINVENT%" "%CODE%\30_94_materialize_reinvent4_formal_seeds43_46_v2.py"
if errorlevel 1 (
  echo ERROR: execution-config verification failed.
  exit /b 2
)

call :runseed 43
if errorlevel 1 exit /b !errorlevel!

call :runseed 44
if errorlevel 1 exit /b !errorlevel!

call :runseed 45
if errorlevel 1 exit /b !errorlevel!

call :runseed 46
if errorlevel 1 exit /b !errorlevel!

echo.
echo ================================================================
echo ALL REINVENT4 FORMAL ENGINE RUNS 43-46 FINISHED NORMALLY
echo Do NOT rerun them. Next step is formal audit/materialization.
echo ================================================================
exit /b 0


:runseed
set "SEED=%~1"
set "OUT=%FORMAL%\seed_!SEED!"
set "CFG=%CODE%\reinvent4_formal_seed!SEED!_v1.toml"
set "LOG=!OUT!\reinvent4_formal.log"

echo.
echo ----------------------------------------------------------------
echo Starting formal seed !SEED!
echo Config: !CFG!
echo ----------------------------------------------------------------

if exist "!OUT!\TASK_COMPLETED.json" (
  echo seed !SEED!: TASK_COMPLETED already exists - SKIP.
  exit /b 0
)

if exist "!LOG!" (
  echo ERROR: seed !SEED! already has reinvent4_formal.log.
  echo Refusing to overwrite an existing formal run.
  echo Audit or inspect that directory before any rerun.
  exit /b 10
)
if exist "!OUT!\oracle_state.json" (
  echo ERROR: seed !SEED! already has oracle_state.json.
  echo Refusing to overwrite an existing formal run.
  exit /b 11
)
if exist "!OUT!\reinvent4_formal_1.csv" (
  echo ERROR: seed !SEED! already has formal CSV output.
  echo Refusing to overwrite an existing formal run.
  exit /b 12
)

if not exist "!OUT!" mkdir "!OUT!"

"%REINVENT%" --seed !SEED! -l "!LOG!" "!CFG!"
if errorlevel 1 (
  echo.
  echo ERROR: REINVENT4 formal seed !SEED! exited nonzero.
  echo Preserve !OUT! exactly as-is for audit. Do NOT rerun.
  exit /b 20
)

findstr /C:"Maximum number of steps of 100 reached in stage 1" "!LOG!" >nul
if errorlevel 1 (
  echo ERROR: seed !SEED! log lacks the frozen 100-step completion marker.
  echo Preserve the directory and stop.
  exit /b 21
)

findstr /C:"Finished REINVENT" "!LOG!" >nul
if errorlevel 1 (
  echo ERROR: seed !SEED! log lacks the normal Finished REINVENT marker.
  echo Preserve the directory and stop.
  exit /b 22
)

if not exist "!OUT!\oracle_state.json" (
  echo ERROR: seed !SEED! finished but oracle_state.json is missing.
  exit /b 23
)
if not exist "!OUT!\oracle_batches.jsonl" (
  echo ERROR: seed !SEED! finished but oracle_batches.jsonl is missing.
  exit /b 24
)
if not exist "!OUT!\oracle_archive.jsonl" (
  echo ERROR: seed !SEED! finished but oracle_archive.jsonl is missing.
  exit /b 25
)
if not exist "!OUT!\reinvent4_formal_1.csv" (
  echo ERROR: seed !SEED! finished but reinvent4_formal_1.csv is missing.
  exit /b 26
)

echo seed !SEED!: ENGINE RUN COMPLETE and basic completion markers PASS.
exit /b 0
