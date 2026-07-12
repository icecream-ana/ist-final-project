@echo off
REM ============================================================
REM Table 3 & Table 4: ImageNet Guided Diffusion
REM Watermark detection (ring, 10 samples) + FID (10 samples)
REM 8 runs total
REM
REM Prerequisites:
REM   1. openai_config/256x256_diffusion.pt (~2GB)
REM   2. openai_config/256x256_diffusion.json
REM   3. FID needs: fid_outputs/imagenet/ground_truth/ (ImageNet val)
REM ============================================================

set HF_HOME=E:/huggingface
if not exist logs mkdir logs

set PYTHON=D:\Environment_2023\Anaconda3\envs\py3.9\python.exe

REM Generate log filename with timestamp
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set D=%%a%%b%%c
for /f "tokens=1-2 delims=:." %%a in ("%time: =0%") do set T=%%a%%b
set LOG=logs\table3_%D%_%T%.log

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

echo ==============================================
echo Table 3 and 4: ImageNet watermark detection + FID
echo ==============================================

REM ---------- 1. model checkpoint check ----------
if not exist "openai_config\256x256_diffusion.pt" (
    echo [ERROR] openai_config/256x256_diffusion.pt not found!
    echo         Download from: https://github.com/openai/guided-diffusion
    echo         Place in openai_config/
    exit /b 1
)
echo [check] openai_config/256x256_diffusion.pt  OK

REM ---------- 2. write JSON config to project root ----------
echo [prep] Writing 256x256_diffusion.json to project root
(
echo {
echo     "model_path": "openai_config/256x256_diffusion.pt",
echo     "attention_resolutions": "32,16,8",
echo     "class_cond": true,
echo     "diffusion_steps": 1000,
echo     "image_size": 256,
echo     "learn_sigma": true,
echo     "noise_schedule": "linear",
echo     "num_channels": 256,
echo     "num_head_channels": 64,
echo     "num_res_blocks": 2,
echo     "resblock_updown": true,
echo     "use_fp16": false,
echo     "use_scale_shift_norm": true,
echo     "classifier_scale": 0
echo }
) > 256x256_diffusion.json

REM ---------- 3. ensure openai_config has JSON ----------
if not exist "openai_config\256x256_diffusion.json" (
    echo [prep] Copying 256x256_diffusion.json to openai_config/
    if not exist openai_config mkdir openai_config
    copy 256x256_diffusion.json openai_config\256x256_diffusion.json >nul
) else (
    echo [prep] openai_config/256x256_diffusion.json exists, skip
)

REM ---------- 4. FID ground truth check ----------
set SKIP_FID=false
if not exist "fid_outputs\imagenet\ground_truth" (
    echo ==============================================
    echo [WARNING] fid_outputs/imagenet/ground_truth/ not found!
    echo   F1 ImageNet FID needs ImageNet val images.
    echo   If you only need E1-E7, you can ignore this.
    echo ==============================================
    echo.
    set /p REPLY="Skip FID experiment and continue? (y/n) "
    if /i not "!REPLY!"=="y" exit /b 1
    set SKIP_FID=true
) else (
    echo [check] fid_outputs/imagenet/ground_truth/  OK
)

goto :skip_e1_e7

REM =====================
REM 10.5 ImageNet watermark detection (E1-E7,10 samples)
REM =====================
set SCRIPT_IMG=run_tree_ring_watermark_imagenet.py
set MODEL_ID=256x256_diffusion
set N_DET=10

echo.
echo ==============================================
echo 10.5 ImageNet watermark detection - 7 experiments
echo ==============================================
echo. >> %LOG%
echo ============================================== >> %LOG%
echo 10.5 ImageNet watermark detection - 7 experiments >> %LOG%
echo ============================================== >> %LOG%

echo.
echo --- [E1] ImageNet - no attack
echo. >> %LOG%
echo --- [E1] ImageNet - no attack >> %LOG%
%PYTHON% %SCRIPT_IMG% --run_name imgnet_no_attack --model_id %MODEL_ID% --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end %N_DET% --with_tracking --reference_model dummy >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [E2] ImageNet - rotation 75 deg
echo. >> %LOG%
echo --- [E2] ImageNet - rotation 75 deg >> %LOG%
%PYTHON% %SCRIPT_IMG% --run_name imgnet_rotation --model_id %MODEL_ID% --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end %N_DET% --r_degree 75 --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [E3] ImageNet - JPEG Q=25
echo. >> %LOG%
echo --- [E3] ImageNet - JPEG Q=25 >> %LOG%
%PYTHON% %SCRIPT_IMG% --run_name imgnet_jpeg --model_id %MODEL_ID% --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end %N_DET% --jpeg_ratio 25 --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [E4] ImageNet - crop 0.75/0.75
echo. >> %LOG%
echo --- [E4] ImageNet - crop 0.75/0.75 >> %LOG%
%PYTHON% %SCRIPT_IMG% --run_name imgnet_cropping --model_id %MODEL_ID% --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end %N_DET% --crop_scale 0.75 --crop_ratio 0.75 --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [E5] ImageNet - gaussian blur r=4
echo. >> %LOG%
echo --- [E5] ImageNet - gaussian blur r=4 >> %LOG%
%PYTHON% %SCRIPT_IMG% --run_name imgnet_blurring --model_id %MODEL_ID% --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end %N_DET% --gaussian_blur_r 4 --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [E6] ImageNet - gaussian noise std=0.1
echo. >> %LOG%
echo --- [E6] ImageNet - gaussian noise std=0.1 >> %LOG%
%PYTHON% %SCRIPT_IMG% --run_name imgnet_noise --model_id %MODEL_ID% --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end %N_DET% --gaussian_std 0.1 --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

echo.
echo --- [E7] ImageNet - brightness factor=6
echo. >> %LOG%
echo --- [E7] ImageNet - brightness factor=6 >> %LOG%
%PYTHON% %SCRIPT_IMG% --run_name imgnet_color_jitter --model_id %MODEL_ID% --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end %N_DET% --brightness_factor 6 --with_tracking >> %LOG% 2>&1
if errorlevel 1 goto :error

:skip_e1_e7
REM =====================
REM 10.6 ImageNet FID (F1, 10 samples)
REM =====================
set SCRIPT_FID=run_tree_ring_watermark_imagenet_fid.py
set N_FID=10

if "%SKIP_FID%"=="true" (
    echo.
    echo ==============================================
    echo [SKIP] 10.6 ImageNet FID - ground truth missing, skipped
    echo ==============================================
    echo. >> %LOG%
    echo ============================================== >> %LOG%
    echo [SKIP] 10.6 ImageNet FID - ground truth missing, skipped >> %LOG%
    echo ============================================== >> %LOG%
) else (
    echo.
    echo ==============================================
    echo 10.6 ImageNet FID - 1 experiment (10 samples)
    echo ==============================================
    echo. >> %LOG%
    echo ============================================== >> %LOG%
    echo 10.6 ImageNet FID - 1 experiment (10 samples) >> %LOG%
    echo ============================================== >> %LOG%

    echo.
    echo --- [F1] ImageNet - FID
    echo. >> %LOG%
    echo --- [F1] ImageNet - FID >> %LOG%
    %PYTHON% %SCRIPT_FID% --run_name imgnet_fid_run --gt_data imagenet --model_id %MODEL_ID% --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end %N_FID% --with_tracking --run_no_w >> %LOG% 2>&1
    if errorlevel 1 goto :error
)

echo.
echo ==============================================
echo Table 3 and 4 done! (8/8)
echo ==============================================
echo. >> %LOG%
echo ============================================== >> %LOG%
echo Table 3 and 4 done! (8/8) >> %LOG%
echo ============================================== >> %LOG%
goto :eof

:error
echo.
echo [ERROR] Command failed, script aborted.
echo. >> %LOG%
echo [ERROR] Command failed, script aborted. >> %LOG%
exit /b 1