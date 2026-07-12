@echo off
REM ============================================================
REM Table 2: Stable Diffusion + FID (COCO ground truth)
REM 3 watermark modes (ring / rand / zeros), 5000 samples each
REM 3 runs total
REM
REM Prerequisites:
REM   1. fid_outputs/coco/meta_data.json
REM   2. fid_outputs/coco/ground_truth/  (COCO 2014 train images)
REM ============================================================

set HF_HOME=E:/huggingface
if not exist logs mkdir logs

set WANDB_MODE=offline
set PYTHON=D:\Environment_2023\Anaconda3\envs\py3.9\python.exe
set SCRIPT=run_tree_ring_watermark_fid.py
set N=50

REM Generate log filename with timestamp
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set D=%%a%%b%%c
for /f "tokens=1-2 delims=:." %%a in ("%time: =0%") do set T=%%a%%b
set LOG=logs\table2_%D%_%T%.log

REM ---------- check ----------
echo [check] Python version:
echo [check] Python version: >> %LOG%
%PYTHON% --version
echo [check] PyTorch version:
echo [check] PyTorch version: >> %LOG%
%PYTHON% -c "import torch; print(torch.__version__)"
echo [check] CUDA available:
echo [check] CUDA available: >> %LOG%
%PYTHON% -c "import torch; print(torch.cuda.is_available())"
echo [check] GPU:
echo [check] GPU: >> %LOG%
%PYTHON% -c "import torch; print(torch.cuda.get_device_name(0))"

echo.
echo ==============================================
echo Table 2: SD + FID - 3 experiments
echo ==============================================
echo. >> %LOG%
echo ============================================== >> %LOG%
echo Table 2: SD + FID - 3 experiments >> %LOG%
echo ============================================== >> %LOG%

REM ---------- D1: ring ----------
echo.
echo --- [D1] FID - ring mode
echo. >> %LOG%
echo --- [D1] FID - ring mode >> %LOG%
%PYTHON% %SCRIPT% --run_name fid_ring --w_channel 3 --w_pattern ring --start 0 --end %N% --with_tracking --run_no_w >> %LOG% 2>&1
if errorlevel 1 goto :error

REM ---------- D2: rand ----------
echo.
echo --- [D2] FID - rand mode
echo. >> %LOG%
echo --- [D2] FID - rand mode >> %LOG%
%PYTHON% %SCRIPT% --run_name fid_rand --w_channel 1 --w_pattern rand --start 0 --end %N% --with_tracking --run_no_w >> %LOG% 2>&1
if errorlevel 1 goto :error

REM ---------- D3: zeros ----------
echo.
echo --- [D3] FID - zeros mode
echo. >> %LOG%
echo --- [D3] FID - zeros mode >> %LOG%
%PYTHON% %SCRIPT% --run_name fid_zeros --w_channel 2 --w_pattern zeros --start 0 --end %N% --with_tracking --run_no_w >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo ==============================================
echo Table 2 done! (3/3)
echo ==============================================
echo. >> %LOG%
echo ============================================== >> %LOG%
echo Table 2 done! (3/3) >> %LOG%
echo ============================================== >> %LOG%
goto :eof

:error
echo.
echo [ERROR] Command failed, script aborted.
echo. >> %LOG%
echo [ERROR] Command failed, script aborted. >> %LOG%
exit /b 1
