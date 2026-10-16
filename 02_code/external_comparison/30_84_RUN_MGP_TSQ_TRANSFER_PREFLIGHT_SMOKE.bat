@echo off
setlocal
cd /d D:\pycharm\project\tsq_mcts_rebuild

echo ============================================================
echo MGP + TSQ TRANSFER: STATIC PREFLIGHT
echo ============================================================
python .\02_code\external_comparison\30_82_preflight_mgp_tsq_transfer_v1.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo MGP + TSQ TRANSFER: ONE-UPDATE SMOKE
echo ============================================================
python .\02_code\external_comparison\30_83_smoke_mgp_tsq_transfer_v1.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo PREFLIGHT + SMOKE COMPLETED
echo Do NOT start formal 5-seed transfer runs yet.
echo Review MGP_TSQ_ONE_UPDATE_SMOKE.json first.
echo ============================================================
exit /b 0

:fail
echo.
echo ERROR: MGP + TSQ transfer preflight/smoke stopped.
exit /b 1
