@echo off
setlocal EnableExtensions
@chcp 65001 >nul 2>&1
cd /d "%~dp0\..\.."
set "PROJECT_ROOT=%CD%"
set "PROJECT_PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "PROTOCOL=%PROJECT_ROOT%\06_paper\experiment_records\external_benchmark_matched_step28_v3_1"
set "RESULT_ROOT=%PROJECT_ROOT%\05_results\external_comparison\matched_step28_v3_1"
set "LOGDIR=%RESULT_ROOT%\_console_logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%" >nul 2>&1
set PYTHONDONTWRITEBYTECODE=1
set PYTHONUTF8=1

set "SCRIPT=%PROJECT_ROOT%\02_code\external_comparison\30_2_run_graphga_matched.py"
if not exist "%PROJECT_PY%" goto :missing
if not exist "%SCRIPT%" goto :missing

echo ================================================================================
echo [30_12] GRAPHGA MATCHED REQUESTS - SEEDS 42 43 44 45 46
echo ================================================================================
for %%S in (42 43 44 45 46) do (
  echo.
  echo [GraphGA] seed %%S
  set "LOG=%LOGDIR%\graphga_seed_%%S.stderr.log"
  "%PROJECT_PY%" "%SCRIPT%" --seed %%S --protocol_dir "%PROTOCOL%" --output_root "%RESULT_ROOT%\graphga" 2>"%LOGDIR%\graphga_seed_%%S.stderr.log"
  if errorlevel 1 goto :fail
)
echo.
echo [30_12] GRAPHGA 5 SEEDS: TECHNICAL PASS
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 0

:missing
echo ERROR: project Python or GraphGA runner is missing.
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 1

:fail
echo.
echo ERROR: [30_12] GraphGA stopped on error.
echo Diagnostic logs: %LOGDIR%\graphga_seed_*.stderr.log
echo Completed seeds are preserved and never overwritten.
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 1
