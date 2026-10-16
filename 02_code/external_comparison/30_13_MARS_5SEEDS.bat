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

set "SCRIPT=%PROJECT_ROOT%\02_code\external_comparison\30_4_run_mars_matched.py"
if not exist "%SCRIPT%" goto :missing
set "CONDA_EXE="
if exist "D:\Miniconda\Scripts\conda.exe" set "CONDA_EXE=D:\Miniconda\Scripts\conda.exe"
if not defined CONDA_EXE for /f "delims=" %%C in ('where conda 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%C"
if not defined CONDA_EXE goto :missingconda

echo ================================================================================
echo [30_13] MARS MATCHED REQUESTS - SEEDS 42 43 44 45 46
echo ================================================================================
for %%S in (42 43 44 45 46) do (
  echo.
  echo [MARS] seed %%S
  call "%CONDA_EXE%" run --no-capture-output -n mars_legacy python "%SCRIPT%" --seed %%S --protocol_dir "%PROTOCOL%" --output_root "%RESULT_ROOT%\mars" 2>"%LOGDIR%\mars_seed_%%S.stderr.log"
  if errorlevel 1 goto :fail
)
echo.
echo [30_13] MARS 5 SEEDS: TECHNICAL PASS
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 0

:missing
echo ERROR: MARS runner is missing: %SCRIPT%
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 1

:missingconda
echo ERROR: conda executable was not found.
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 1

:fail
echo.
echo ERROR: [30_13] MARS stopped on error.
echo Diagnostic logs: %LOGDIR%\mars_seed_*.stderr.log
echo Completed seeds are preserved and never overwritten.
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 1
