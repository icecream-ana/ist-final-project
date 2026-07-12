#!/bin/bash
# ============================================================
# Table 2: Stable Diffusion + FID (COCO ground truth)
# 三种水印模式 (ring / rand / zeros), 各 5000 样本
# 共 3 条命令
#
# 前置条件:
#   1. fid_outputs/coco/meta_data.json
#   2. fid_outputs/coco/ground_truth/  (COCO 2014 训练集图片)
# ============================================================

set -e

# HuggingFace 缓存路径 (模型/数据集下载到 E 盘)
export HF_HOME=E:/huggingface

# 日志记录
mkdir -p logs
LOG_FILE="logs/table2_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"
echo ""
exec > >(tee -a "$LOG_FILE") 2>&1

SCRIPT="run_tree_ring_watermark_fid.py"
N=5000

# ---------- 前置检查 ----------
if [ ! -f "fid_outputs/coco/meta_data.json" ]; then
    echo "[ERROR] fid_outputs/coco/meta_data.json 不存在!"
    echo "        请先准备 COCO 数据 (参见 setup.md 4.2 节)"
    exit 1
fi
if [ ! -d "fid_outputs/coco/ground_truth" ]; then
    echo "[ERROR] fid_outputs/coco/ground_truth/ 不存在!"
    echo "        请先放入 COCO 2014 训练集图片"
    exit 1
fi
GT_COUNT=$(ls fid_outputs/coco/ground_truth/ | wc -l)
echo "[check] fid_outputs/coco/meta_data.json  OK"
echo "[check] fid_outputs/coco/ground_truth/   OK ($GT_COUNT 张图片)"

echo "=============================================="
echo "Table 2: SD + FID — 共 3 个实验"
echo "=============================================="

# ---------- D1: ring 模式 ----------
echo ""
echo ">>> [D1] FID — ring 模式"
python $SCRIPT --run_name fid_ring --w_channel 3 --w_pattern ring --start 0 --end $N --with_tracking --run_no_w

# ---------- D2: rand 模式 ----------
echo ""
echo ">>> [D2] FID — rand 模式"
python $SCRIPT --run_name fid_rand --w_channel 1 --w_pattern rand --start 0 --end $N --with_tracking --run_no_w

# ---------- D3: zeros 模式 ----------
echo ""
echo ">>> [D3] FID — zeros 模式"
python $SCRIPT --run_name fid_zeros --w_channel 2 --w_pattern zeros --start 0 --end $N --with_tracking --run_no_w

echo ""
echo "=============================================="
echo "Table 2 全部完成! (3/3)"
echo "=============================================="
