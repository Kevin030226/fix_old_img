@echo off
echo ============================================
echo  Old Photo Restoration - GPU Env Setup
echo  (Python 3.11 + PyTorch 2.7.1 cu128 + native web UI)
echo ============================================
echo.

where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] conda not found in PATH.
    pause
    exit /b 1
)

echo [1/6] Creating conda environment (Python 3.11)...
call conda create -n fixoldimg-gpu python=3.11 -y
if %errorlevel% neq 0 exit /b 1

echo [2/6] Installing PyTorch 2.7.1 cu128 (RTX 50 / sm_120 support)...
call conda run -n fixoldimg-gpu pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo [WARN] official index failed, trying SJTU mirror...
    call conda run -n fixoldimg-gpu pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128
    if %errorlevel% neq 0 exit /b 1
)

echo [3/6] Installing prebuilt dlib from conda-forge (PyPI has no Windows wheel)...
call conda install -n fixoldimg-gpu -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge --override-channels -y dlib=20.0.1
if %errorlevel% neq 0 exit /b 1

echo [4/6] Installing the pinned GPU deployment group...
call conda run -n fixoldimg-gpu pip install -r "%~dp0..\requirements\runtime.txt" -r "%~dp0..\requirements\gpu.txt" --extra-index-url https://download.pytorch.org/whl/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple
if %errorlevel% neq 0 exit /b 1

echo [5/6] Making the fiximg package importable...
call conda run -n fixoldimg-gpu pip install -e "%~dp0.." --no-deps
if %errorlevel% neq 0 exit /b 1

echo [6/6] Verifying GPU and the platform prerequisites...
call conda run -n fixoldimg-gpu python -c "import torch, dlib, timm; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); print('dlib', dlib.__version__, 'timm', timm.__version__)"

echo.
echo ============================================
echo  SETUP COMPLETE!
echo  Activate env: conda activate fixoldimg-gpu
echo  Web UI:       python main.py  (open http://127.0.0.1:9502)
echo  CLI:          python run.py --input_folder ./test_images/old --output_folder ./output
echo ============================================
pause

