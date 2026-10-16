@echo off
setlocal
cd /d D:\pycharm\project\tsq_mcts_rebuild

set PROTOCOL=.\06_paper\experiment_records\external_benchmark_v4_mgp_protocol\MGP_V4_PROTOCOL.json

if not exist "%PROTOCOL%" (
    echo ERROR: Frozen MGP v4 protocol not found:
    echo %PROTOCOL%
    echo Run 30_78_freeze_mgp_nsga2_v4.py ONCE before formal seeds.
    exit /b 1
)

echo ============================================================
echo MGP v4 formal protocol already frozen.
echo IMPORTANT: 30_78 is NOT rerun here, so the protocol SHA
echo remains identical to the completed seed 42 run.
echo ============================================================

for %%S in (42 43 44 45 46) do (
    echo ============================================================
    echo Running/checking MGP v4 formal seed %%S
    echo ============================================================
    python .\02_code\external_comparison\30_79_run_mgp_nsga2_v4.py --seed %%S
    if errorlevel 1 goto :fail
)

echo ============================================================
echo ALL MGP V4 FORMAL SEEDS COMPLETED
echo ============================================================
exit /b 0

:fail
echo.
echo ERROR: MGP v4 formal run stopped.
exit /b 1