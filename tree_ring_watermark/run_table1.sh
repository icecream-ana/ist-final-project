#!/bin/bash
# ============================================================
# Table 1: Stable Diffusion + CLIP Score
# 三种水印模式 (ring / rand / zeros) x 7 种攻击场景
# 共 21 条命令, 每条生成 1000 张图片
# ============================================================

set -e

# HuggingFace 缓存路径 (模型/数据集下载到 E 盘)
export HF_HOME=E:/huggingface

# 日志记录
mkdir -p logs
LOG_FILE="logs/table1_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"
echo ""
exec > >(tee -a "$LOG_FILE") 2>&1

PYTHON=/mnt/d/Environment_2023/Anaconda3/envs/py3.9/python.exe
SCRIPT="run_tree_ring_watermark.py"
N=1000

# ---------- 前置检查 ----------
echo "[check] Python 版本: $($PYTHON --version 2>&1)"
echo "[check] PyTorch 版本: $($PYTHON -c 'import torch; print(torch.__version__)' 2>&1)"
echo "[check] CUDA 可用: $($PYTHON -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo "[check] GPU: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1 || echo 'N/A')"

echo ""
echo "=============================================="
echo "Table 1: SD + CLIP Score — 共 21 个实验"
echo "=============================================="

# ---------- 10.1 ring 模式 (w_channel=3) ----------
echo ""
echo ">>> [A1] ring — 无攻击 (含 CLIP Score)"
$PYTHON $SCRIPT --run_name no_attack_ring --w_channel 3 --w_pattern ring --start 0 --end $N --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k

echo ""
echo ">>> [A2] ring — 旋转攻击 75°"
$PYTHON $SCRIPT --run_name rotation_ring --w_channel 3 --w_pattern ring --r_degree 75 --start 0 --end $N --with_tracking

echo ""
echo ">>> [A3] ring — JPEG 压缩攻击 Q=25"
$PYTHON $SCRIPT --run_name jpeg_ring --w_channel 3 --w_pattern ring --jpeg_ratio 25 --start 0 --end $N --with_tracking

echo ""
echo ">>> [A4] ring — 裁剪攻击 0.75/0.75"
$PYTHON $SCRIPT --run_name cropping_ring --w_channel 3 --w_pattern ring --crop_scale 0.75 --crop_ratio 0.75 --start 0 --end $N --with_tracking

echo ""
echo ">>> [A5] ring — 高斯模糊攻击 r=4"
$PYTHON $SCRIPT --run_name blurring_ring --w_channel 3 --w_pattern ring --gaussian_blur_r 4 --start 0 --end $N --with_tracking

echo ""
echo ">>> [A6] ring — 高斯噪声攻击 std=0.1"
$PYTHON $SCRIPT --run_name noise_ring --w_channel 3 --w_pattern ring --gaussian_std 0.1 --start 0 --end $N --with_tracking

echo ""
echo ">>> [A7] ring — 亮度调整攻击 factor=6"
$PYTHON $SCRIPT --run_name color_jitter_ring --w_channel 3 --w_pattern ring --brightness_factor 6 --start 0 --end $N --with_tracking

# ---------- 10.2 rand 模式 (w_channel=1) ----------
echo ""
echo ">>> [B1] rand — 无攻击 (含 CLIP Score)"
$PYTHON $SCRIPT --run_name no_attack_rand --w_channel 1 --w_pattern rand --start 0 --end $N --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k

echo ""
echo ">>> [B2] rand — 旋转攻击 75°"
$PYTHON $SCRIPT --run_name rotation_rand --w_channel 1 --w_pattern rand --r_degree 75 --start 0 --end $N --with_tracking

echo ""
echo ">>> [B3] rand — JPEG 压缩攻击 Q=25"
$PYTHON $SCRIPT --run_name jpeg_rand --w_channel 1 --w_pattern rand --jpeg_ratio 25 --start 0 --end $N --with_tracking

echo ""
echo ">>> [B4] rand — 裁剪攻击 0.75/0.75"
$PYTHON $SCRIPT --run_name cropping_rand --w_channel 1 --w_pattern rand --crop_scale 0.75 --crop_ratio 0.75 --start 0 --end $N --with_tracking

echo ""
echo ">>> [B5] rand — 高斯模糊攻击 r=4"
$PYTHON $SCRIPT --run_name blurring_rand --w_channel 1 --w_pattern rand --gaussian_blur_r 4 --start 0 --end $N --with_tracking

echo ""
echo ">>> [B6] rand — 高斯噪声攻击 std=0.1"
$PYTHON $SCRIPT --run_name noise_rand --w_channel 1 --w_pattern rand --gaussian_std 0.1 --start 0 --end $N --with_tracking

echo ""
echo ">>> [B7] rand — 亮度调整攻击 factor=6"
$PYTHON $SCRIPT --run_name color_jitter_rand --w_channel 1 --w_pattern rand --brightness_factor 6 --start 0 --end $N --with_tracking

# ---------- 10.3 zeros 模式 (w_channel=2) ----------
echo ""
echo ">>> [C1] zeros — 无攻击 (含 CLIP Score)"
$PYTHON $SCRIPT --run_name no_attack_zeros --w_channel 2 --w_pattern zeros --start 0 --end $N --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k

echo ""
echo ">>> [C2] zeros — 旋转攻击 75°"
$PYTHON $SCRIPT --run_name rotation_zeros --w_channel 2 --w_pattern zeros --r_degree 75 --start 0 --end $N --with_tracking

echo ""
echo ">>> [C3] zeros — JPEG 压缩攻击 Q=25"
$PYTHON $SCRIPT --run_name jpeg_zeros --w_channel 2 --w_pattern zeros --jpeg_ratio 25 --start 0 --end $N --with_tracking

echo ""
echo ">>> [C4] zeros — 裁剪攻击 0.75/0.75"
$PYTHON $SCRIPT --run_name cropping_zeros --w_channel 2 --w_pattern zeros --crop_scale 0.75 --crop_ratio 0.75 --start 0 --end $N --with_tracking

echo ""
echo ">>> [C5] zeros — 高斯模糊攻击 r=4"
$PYTHON $SCRIPT --run_name blurring_zeros --w_channel 2 --w_pattern zeros --gaussian_blur_r 4 --start 0 --end $N --with_tracking

echo ""
echo ">>> [C6] zeros — 高斯噪声攻击 std=0.1"
$PYTHON $SCRIPT --run_name noise_zeros --w_channel 2 --w_pattern zeros --gaussian_std 0.1 --start 0 --end $N --with_tracking

echo ""
echo ">>> [C7] zeros — 亮度调整攻击 factor=6"
$PYTHON $SCRIPT --run_name color_jitter_zeros --w_channel 2 --w_pattern zeros --brightness_factor 6 --start 0 --end $N --with_tracking

echo ""
echo "=============================================="
echo "Table 1 全部完成! (21/21)"
echo "=============================================="
