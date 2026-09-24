@echo off
REM ============================================================
REM  Old Photo Restoration - one-click start (conda env)
REM  Guarantees the service runs under the fixoldimg-gpu
REM  interpreter so subprocess stages (run.py) use the same
REM  Python + PyTorch. Running main.py with a system Python
REM  (e.g. Python 3.14) makes every task fail with exit=2.
REM ============================================================
setlocal

set ENV_NAME=fixoldimg-gpu
set FIXIMG_PORT=9502

where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] conda not found in PATH.
    pause
    exit /b 1
)

echo ============================================
echo  Starting Old Photo Restoration service
echo  conda env : %ENV_NAME%
echo  URL       : http://127.0.0.1:%FIXIMG_PORT%
echo ============================================
echo.

cd /d "%~dp0.."

call conda run --no-capture-output -n %ENV_NAME% python main.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Service exited with code %errorlevel%.
    pause
    exit /b %errorlevel%
)

pause
