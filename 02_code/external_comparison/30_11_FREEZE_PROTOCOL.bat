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

set "SCRIPT=%PROJECT_ROOT%\02_code\external_comparison\30_1_freeze_external_matched_protocol.py"
set "LOG=%LOGDIR%\30_11_freeze.stderr.log"
if not exist "%PROJECT_PY%" goto :missing
if not exist "%SCRIPT%" goto :missing

echo ================================================================================
echo [30_11] FREEZE / VERIFY MATCHED STEP-28 V3.1 PROTOCOL
echo No external optimization is performed in this stage.
echo ================================================================================
"%PROJECT_PY%" "%SCRIPT%" --protocol_dir "%PROTOCOL%" 2>"%LOG%"
if errorlevel 1 goto :fail
echo.
echo [30_11] PASS
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 0

:missing
echo ERROR: project Python or freeze script is missing.
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 1

:fail
echo.
echo ERROR: [30_11] protocol freeze/verification failed.
echo Diagnostic log: %LOG%
echo Later formal stages were NOT started.
if not defined EXTERNAL_NO_PAUSE pause >nul
exit /b 1
