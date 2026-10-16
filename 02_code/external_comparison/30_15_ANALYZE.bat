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

set "SCRIPT=%PROJECT_ROOT%\02_code\external_comparison\30_10_analyze_external_matched.py"
set "LOG=%LOGDIR%\analysis.stderr.log"
if not exist "%PROJECT_PY%" goto :missing
if not exist "%SCRIPT%" goto :missing

echo ================================================================================
echo [30_15] ANALYZE 20 PAIRED RUNS
echo ================================================================================
"%PROJECT_PY%" "%SCRIPT%" --protocol_dir "%PROTOCOL%" --result_root "%RESULT_ROOT%" 2>"%LOG%"
if errorlevel 1 goto :fail
echo.
echo [30_15] ANALYSIS PASS
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 0

:missing
echo ERROR: project Python or analysis script is missing.
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 1

:fail
echo.
echo ERROR: [30_15] analysis failed.
echo Diagnostic log: %LOG%
echo Formal run outputs were not modified.
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 1
