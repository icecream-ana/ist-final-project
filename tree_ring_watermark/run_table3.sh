#!/bin/bash
# ============================================================
# Table 3 & Table 4: ImageNet Guided Diffusion
# 水印检测 (ring, 1000 样本) + FID (10000 样本)
# 共 8 条命令
#
# 前置条件:
#   1. openai_config/256x256_diffusion.pt (约 2GB) — 已就绪
#   2. openai_config/256x256_diffusion.json — 已就绪
#   3. FID 需要: fid_outputs/imagenet/ground_truth/ (ImageNet 验证集)
# ============================================================

set -e

# HuggingFace 缓存路径 (模型/数据集下载到 E 盘)
export HF_HOME=E:/huggingface

# 日志记录
mkdir -p logs
LOG_FILE="logs/table3_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"
echo ""
exec > >(tee -a "$LOG_FILE") 2>&1

# ---------- 前置检查 & 准备 ----------
echo "=============================================="
echo "Table 3 & 4: ImageNet 水印检测 + FID"
echo "=============================================="

# ---------- 1. 模型 checkpoint 检查 ----------
if [ ! -f "openai_config/256x256_diffusion.pt" ]; then
    echo "[ERROR] openai_config/256x256_diffusion.pt 不存在!"
    echo "        请先下载 ImageNet 预训练模型: https://github.com/openai/guided-diffusion"
    echo "        放到 openai_config/ 目录下"
    exit 1
fi
echo "[check] openai_config/256x256_diffusion.pt  OK"

# ---------- 2. 恢复项目根目录的 JSON 配置 ----------
# run_tree_ring_watermark_imagenet.py 从项目根目录读取 {model_id}.json
# 但该文件已被 git 删除。从 openai_config/ 恢复, 并修正 model_path 指向正确位置
echo "[prep] 恢复项目根目录 256x256_diffusion.json (model_path -> openai_config/)"
cat > 256x256_diffusion.json << 'JSONEOF'
{
    "model_path": "openai_config/256x256_diffusion.pt",
    "attention_resolutions": "32,16,8",
    "class_cond": true,
    "diffusion_steps": 1000,
    "image_size": 256,
    "learn_sigma": true,
    "noise_schedule": "linear",
    "num_channels": 256,
    "num_head_channels": 64,
    "num_res_blocks": 2,
    "resblock_updown": true,
    "use_fp16": false,
    "use_scale_shift_norm": true,
    "classifier_scale": 0
}
JSONEOF

# ---------- 3. 确保 openai_config 下的 JSON (供 FID 脚本使用) ----------
if [ ! -f "openai_config/256x256_diffusion.json" ]; then
    echo "[prep] 复制 256x256_diffusion.json -> openai_config/"
    mkdir -p openai_config
    cp 256x256_diffusion.json openai_config/256x256_diffusion.json
else
    echo "[prep] openai_config/256x256_diffusion.json 已存在, 跳过"
fi

# ---------- 4. FID ground truth 检查 ----------
if [ ! -d "fid_outputs/imagenet/ground_truth" ]; then
    echo "=============================================="
    echo "[WARNING] fid_outputs/imagenet/ground_truth/ 不存在!"
    echo "  F1 (ImageNet FID) 需要 ImageNet 验证集图片。"
    echo "  如果只需要跑 E1-E7 水印检测，可以忽略此警告。"
    echo "  如需跑 FID: 将 ImageNet 验证集放入 fid_outputs/imagenet/ground_truth/"
    echo "=============================================="
    echo ""
    read -p "跳过 FID 实验继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    SKIP_FID=true
else
    echo "[check] fid_outputs/imagenet/ground_truth/  OK"
    SKIP_FID=false
    COUNT=$(ls fid_outputs/imagenet/ground_truth/ | wc -l)
    echo "        图片数量: $COUNT"
fi

# =====================
# 10.5 ImageNet 水印检测 (E1-E7, 1000 样本)
# =====================
SCRIPT_IMG="run_tree_ring_watermark_imagenet.py"
MODEL_ID="256x256_diffusion"
N_DET=1000

echo ""
echo "=============================================="
echo "10.5 ImageNet 水印检测 — 共 7 个实验"
echo "=============================================="

echo ""
echo ">>> [E1] ImageNet — 无攻击"
python $SCRIPT_IMG --run_name imgnet_no_attack --model_id $MODEL_ID --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end $N_DET --with_tracking --reference_model dummy

echo ""
echo ">>> [E2] ImageNet — 旋转攻击 75°"
python $SCRIPT_IMG --run_name imgnet_rotation --model_id $MODEL_ID --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end $N_DET --r_degree 75 --with_tracking

echo ""
echo ">>> [E3] ImageNet — JPEG 压缩攻击 Q=25"
python $SCRIPT_IMG --run_name imgnet_jpeg --model_id $MODEL_ID --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end $N_DET --jpeg_ratio 25 --with_tracking

echo ""
echo ">>> [E4] ImageNet — 裁剪攻击 0.75/0.75"
python $SCRIPT_IMG --run_name imgnet_cropping --model_id $MODEL_ID --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end $N_DET --crop_scale 0.75 --crop_ratio 0.75 --with_tracking

echo ""
echo ">>> [E5] ImageNet — 高斯模糊攻击 r=4"
python $SCRIPT_IMG --run_name imgnet_blurring --model_id $MODEL_ID --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end $N_DET --gaussian_blur_r 4 --with_tracking

echo ""
echo ">>> [E6] ImageNet — 高斯噪声攻击 std=0.1"
python $SCRIPT_IMG --run_name imgnet_noise --model_id $MODEL_ID --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end $N_DET --gaussian_std 0.1 --with_tracking

echo ""
echo ">>> [E7] ImageNet — 亮度调整攻击 factor=6"
python $SCRIPT_IMG --run_name imgnet_color_jitter --model_id $MODEL_ID --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end $N_DET --brightness_factor 6 --with_tracking

# =====================
# 10.6 ImageNet FID (F1, 10000 样本)
# =====================
SCRIPT_FID="run_tree_ring_watermark_imagenet_fid.py"
N_FID=10000

if [ "$SKIP_FID" = true ]; then
    echo ""
    echo "=============================================="
    echo "[SKIP] 10.6 ImageNet FID — ground truth 缺失, 已跳过"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "10.6 ImageNet FID — 共 1 个实验 (10000 样本)"
    echo "=============================================="

    echo ""
    echo ">>> [F1] ImageNet — FID"
    python $SCRIPT_FID --run_name imgnet_fid_run --gt_data imagenet --model_id $MODEL_ID --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end $N_FID --with_tracking --run_no_w
fi

echo ""
echo "=============================================="
echo "Table 3 & 4 全部完成! (8/8)"
echo "=============================================="
