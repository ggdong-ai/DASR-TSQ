@echo off
setlocal
cd /d D:\pycharm\project\tsq_mcts_rebuild

set PROTOCOL=.\06_paper\experiment_records\external_benchmark_v4_mgp_tsq_transfer_protocol\MGP_TSQ_TRANSFER_PROTOCOL.json

if not exist "%PROTOCOL%" (
    echo ERROR: Frozen MGP+TSQ transfer protocol not found:
    echo %PROTOCOL%
    echo Run 30_85_freeze_mgp_tsq_transfer_v1.py ONCE first.
    exit /b 1
)

echo ============================================================
echo MGP + TSQ(q=5) formal transfer protocol found.
echo IMPORTANT: this batch NEVER re-freezes the protocol.
echo Completed seeds under the identical protocol will be skipped.
echo ============================================================

for %%S in (42 43 44 45 46) do (
    echo.
    echo ============================================================
    echo Running/checking MGP + TSQ(q=5) seed %%S
    echo ============================================================
    python .\02_code\external_comparison\30_86_run_mgp_tsq_transfer_v1.py --seed %%S
    if errorlevel 1 goto :fail
)

echo.
echo ============================================================
echo ALL MGP + TSQ(q=5) FORMAL TRANSFER SEEDS COMPLETED
echo ============================================================
exit /b 0

:fail
echo.
echo ERROR: MGP + TSQ(q=5) formal transfer run stopped.
exit /b 1
