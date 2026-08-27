@echo off
REM ============================================================
REM [DEPRECATED] This script corresponds to the legacy Python 3.8 environment (old_photo_restore).
REM For the current project please use setup_gpu.bat (Python 3.11 + PyTorch cu128 + Gradio 6).
REM ============================================================
echo ============================================
echo  Old Photo Restoration - Conda Setup Script
echo ============================================
echo.

:: Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] conda not found in PATH.
    echo Please install Anaconda or Miniconda first.
    echo Make sure "Add to PATH" was checked during installation.
    pause
    exit /b 1
)

echo [1/4] Creating conda environment (Python 3.9)...
call conda create -n old_photo_restore python=3.9 -y

echo [2/4] Activating environment and installing PyTorch...
call conda activate old_photo_restore
call conda install pytorch torchvision cpuonly -c pytorch -y

echo [3/4] Installing core dependencies...
call conda install -c conda-forge dlib -y
pip install scikit-image easydict PyYAML dominate dill tensorboardX scipy opencv-python einops matplotlib pyspng gradio fastapi uvicorn numpy Pillow
echo [note] PySimpleGUI (desktop GUI, GUI.py) is an optional dependency; 5.x is now commercially licensed, install separately if needed.

echo [4/4] Installing Synchronized-BatchNorm-PyTorch...
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%Face_Enhancement\models\networks\Synchronized-BatchNorm-PyTorch"
pip install -e .
cd /d "%PROJECT_DIR%Global\detection_models\Synchronized-BatchNorm-PyTorch"
pip install -e .
cd /d "%PROJECT_DIR%"

echo.
echo ============================================
echo  SETUP COMPLETE!
echo.
echo  How to use:
echo.
echo  Activate env: conda activate old_photo_restore
echo  Web UI:       python main.py  (then open http://127.0.0.1:9502)
echo  CLI (clean):  python run.py --input_folder ./test_images/old --output_folder ./output
echo  CLI (scratch):python run.py --input_folder ./test_images/old_w_scratch --output_folder ./output --with_scratch
echo ============================================
pause
