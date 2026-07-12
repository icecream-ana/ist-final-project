@echo off
REM ============================================================
REM Table 1: Stable Diffusion + CLIP Score
REM 3 watermark modes (ring / rand / zeros) x 7 attacks = 21 runs
REM ============================================================

set HF_HOME=E:/huggingface
if not exist logs mkdir logs

set WANDB_MODE=offline
set PYTHON=D:\Environment_2023\Anaconda3\envs\py3.9\python.exe
set SCRIPT=run_tree_ring_watermark.py
set N=10

REM Generate log filename with timestamp
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set D=%%a%%b%%c
for /f "tokens=1-2 delims=:." %%a in ("%time: =0%") do set T=%%a%%b
set LOG=logs\table1_%D%_%T%.log

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
echo Table 1: SD + CLIP Score - 21 experiments
echo ==============================================
echo. >> %LOG%
echo ============================================== >> %LOG%
echo Table 1: SD + CLIP Score - 21 experiments >> %LOG%
echo ============================================== >> %LOG%

REM ---------- 10.1 ring mode (w_channel=3) ----------
echo.
echo --- [A1] ring - no attack (with CLIP Score)
echo. >> %LOG%
echo --- [A1] ring - no attack (with CLIP Score) >> %LOG%
%PYTHON% %SCRIPT% --run_name no_attack_ring --w_channel 3 --w_pattern ring --start 0 --end %N% --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [A2] ring - rotation 75 deg
echo. >> %LOG%
echo --- [A2] ring - rotation 75 deg >> %LOG%
%PYTHON% %SCRIPT% --run_name rotation_ring --w_channel 3 --w_pattern ring --r_degree 75 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [A3] ring - JPEG Q=25
echo. >> %LOG%
echo --- [A3] ring - JPEG Q=25 >> %LOG%
%PYTHON% %SCRIPT% --run_name jpeg_ring --w_channel 3 --w_pattern ring --jpeg_ratio 25 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [A4] ring - crop 0.75/0.75
echo. >> %LOG%
echo --- [A4] ring - crop 0.75/0.75 >> %LOG%
%PYTHON% %SCRIPT% --run_name cropping_ring --w_channel 3 --w_pattern ring --crop_scale 0.75 --crop_ratio 0.75 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [A5] ring - gaussian blur r=4
echo. >> %LOG%
echo --- [A5] ring - gaussian blur r=4 >> %LOG%
%PYTHON% %SCRIPT% --run_name blurring_ring --w_channel 3 --w_pattern ring --gaussian_blur_r 4 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [A6] ring - gaussian noise std=0.1
echo. >> %LOG%
echo --- [A6] ring - gaussian noise std=0.1 >> %LOG%
%PYTHON% %SCRIPT% --run_name noise_ring --w_channel 3 --w_pattern ring --gaussian_std 0.1 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [A7] ring - brightness factor=6
echo. >> %LOG%
echo --- [A7] ring - brightness factor=6 >> %LOG%
%PYTHON% %SCRIPT% --run_name color_jitter_ring --w_channel 3 --w_pattern ring --brightness_factor 6 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

REM ---------- 10.2 rand mode (w_channel=1) ----------
echo.
echo --- [B1] rand - no attack (with CLIP Score)
echo. >> %LOG%
echo --- [B1] rand - no attack (with CLIP Score) >> %LOG%
%PYTHON% %SCRIPT% --run_name no_attack_rand --w_channel 1 --w_pattern rand --start 0 --end %N% --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [B2] rand - rotation 75 deg
echo. >> %LOG%
echo --- [B2] rand - rotation 75 deg >> %LOG%
%PYTHON% %SCRIPT% --run_name rotation_rand --w_channel 1 --w_pattern rand --r_degree 75 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [B3] rand - JPEG Q=25
echo. >> %LOG%
echo --- [B3] rand - JPEG Q=25 >> %LOG%
%PYTHON% %SCRIPT% --run_name jpeg_rand --w_channel 1 --w_pattern rand --jpeg_ratio 25 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [B4] rand - crop 0.75/0.75
echo. >> %LOG%
echo --- [B4] rand - crop 0.75/0.75 >> %LOG%
%PYTHON% %SCRIPT% --run_name cropping_rand --w_channel 1 --w_pattern rand --crop_scale 0.75 --crop_ratio 0.75 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [B5] rand - gaussian blur r=4
echo. >> %LOG%
echo --- [B5] rand - gaussian blur r=4 >> %LOG%
%PYTHON% %SCRIPT% --run_name blurring_rand --w_channel 1 --w_pattern rand --gaussian_blur_r 4 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [B6] rand - gaussian noise std=0.1
echo. >> %LOG%
echo --- [B6] rand - gaussian noise std=0.1 >> %LOG%
%PYTHON% %SCRIPT% --run_name noise_rand --w_channel 1 --w_pattern rand --gaussian_std 0.1 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [B7] rand - brightness factor=6
echo. >> %LOG%
echo --- [B7] rand - brightness factor=6 >> %LOG%
%PYTHON% %SCRIPT% --run_name color_jitter_rand --w_channel 1 --w_pattern rand --brightness_factor 6 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

REM ---------- 10.3 zeros mode (w_channel=2) ----------
echo.
echo --- [C1] zeros - no attack (with CLIP Score)
echo. >> %LOG%
echo --- [C1] zeros - no attack (with CLIP Score) >> %LOG%
%PYTHON% %SCRIPT% --run_name no_attack_zeros --w_channel 2 --w_pattern zeros --start 0 --end %N% --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [C2] zeros - rotation 75 deg
echo. >> %LOG%
echo --- [C2] zeros - rotation 75 deg >> %LOG%
%PYTHON% %SCRIPT% --run_name rotation_zeros --w_channel 2 --w_pattern zeros --r_degree 75 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [C3] zeros - JPEG Q=25
echo. >> %LOG%
echo --- [C3] zeros - JPEG Q=25 >> %LOG%
%PYTHON% %SCRIPT% --run_name jpeg_zeros --w_channel 2 --w_pattern zeros --jpeg_ratio 25 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [C4] zeros - crop 0.75/0.75
echo. >> %LOG%
echo --- [C4] zeros - crop 0.75/0.75 >> %LOG%
%PYTHON% %SCRIPT% --run_name cropping_zeros --w_channel 2 --w_pattern zeros --crop_scale 0.75 --crop_ratio 0.75 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [C5] zeros - gaussian blur r=4
echo. >> %LOG%
echo --- [C5] zeros - gaussian blur r=4 >> %LOG%
%PYTHON% %SCRIPT% --run_name blurring_zeros --w_channel 2 --w_pattern zeros --gaussian_blur_r 4 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [C6] zeros - gaussian noise std=0.1
echo. >> %LOG%
echo --- [C6] zeros - gaussian noise std=0.1 >> %LOG%
%PYTHON% %SCRIPT% --run_name noise_zeros --w_channel 2 --w_pattern zeros --gaussian_std 0.1 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [C7] zeros - brightness factor=6
echo. >> %LOG%
echo --- [C7] zeros - brightness factor=6 >> %LOG%
%PYTHON% %SCRIPT% --run_name color_jitter_zeros --w_channel 2 --w_pattern zeros --brightness_factor 6 --start 0 --end %N% --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo ==============================================
echo Table 1 done! (21/21)
echo ==============================================
echo. >> %LOG%
echo ============================================== >> %LOG%
echo Table 1 done! (21/21) >> %LOG%
echo ============================================== >> %LOG%
goto :eof

:error
echo.
echo [ERROR] Command failed, script aborted.
echo. >> %LOG%
echo [ERROR] Command failed, script aborted. >> %LOG%
exit /b 1
