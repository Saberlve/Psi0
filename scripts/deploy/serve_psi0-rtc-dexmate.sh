#!/bin/bash
# ========================================================================
# Psi0 RTC 服务器启动脚本 (Dexmate 机器人版本)
# ========================================================================
# 使用方法:
#   bash scripts/deploy/serve_psi0-rtc-dexmate.sh <checkpoint_dir> <ckpt_step>
#
# 示例:
#   bash scripts/deploy/serve_psi0-rtc-dexmate.sh ~/Psi0/runs/pretrain/xxx 5000
# ========================================================================

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: bash $0 <checkpoint_dir> <ckpt_step>"
    echo "示例: bash $0 ~/Psi0/runs/pretrain/xxx 5000"
    exit 1
fi

CHECKPOINT_DIR="$1"
CHECKPOINT_STEP="$2"

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES=0
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Loading checkpoint: $CHECKPOINT_DIR @ step $CHECKPOINT_STEP"

python src/psi/deploy/psi_serve_rtc-dexmate.py \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir="$CHECKPOINT_DIR" \
    --ckpt-step="$CHECKPOINT_STEP"
