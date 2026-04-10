#!/bin/bash
# ========================================================================
# Psi0 RTC 服务器启动脚本 (在 GPU 服务器上运行)
# ========================================================================
# 作用: 启动 Psi0 RTC 推理服务器，加载模型并等待机器人客户端连接
#
# 使用方法:
#   方式1 (推荐): 指定 task 自动查找最新 run
#     bash scripts/deploy/serve_psi0-rtc.sh <task> <ckpt_step>
#     示例: bash scripts/deploy/serve_psi0-rtc.sh Hug_box_and_move 10000
#
#   方式2: 指定完整 checkpoint 路径
#     bash scripts/deploy/serve_psi0-rtc.sh /path/to/run_dir <ckpt_step>
#
# 参数:
#   checkpoint_dir 或 task: task 名称 或 checkpoint 完整路径
#   ckpt_step:                checkpoint 步数 (如 10000)
#
# 注意: 与 finetune-real-psi0.sh 对齐，使用相同的 PSI_HOME 和输出路径
#   finetune 输出: $PSI_HOME/runs/finetune/<task>.<timestamp>/
# ========================================================================

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: bash $0 <task|checkpoint_dir> <ckpt_step>"
    echo ""
    echo "示例 (方式1 - 推荐): 指定 task 自动查找"
    echo "  bash scripts/deploy/serve_psi0-rtc.sh Hug_box_and_move 10000"
    echo ""
    echo "示例 (方式2): 指定完整路径"
    echo "  bash scripts/deploy/serve_psi0-rtc.sh ~/Psi0/runs/finetune/hug_box_and_move.2601091803 10000"
    exit 1
fi

INPUT_PATH="$1"
CKPT_STEP="$2"

# PSI_HOME 默认值 (与 finetune.sh 对齐)
PSI_HOME="${PSI_HOME:-$HOME/Psi0}"

# 判断是 task 名称还是完整路径
if [[ "$INPUT_PATH" == /* ]] || [[ "$INPUT_PATH" == ./* ]]; then
    # 完整路径
    CHECKPOINT_DIR="$INPUT_PATH"
else
    # task 名称，自动查找最新的 run
    TASK="$INPUT_PATH"
    TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    RUNS_DIR="$PSI_HOME/runs/finetune"

    if [ ! -d "$RUNS_DIR" ]; then
        echo "错误: 找不到 runs 目录: $RUNS_DIR"
        exit 1
    fi

    # 查找匹配 task 的最新目录
    LATEST_RUN=$(ls -td "$RUNS_DIR"/${TASK_LOWER}.* 2>/dev/null | head -1)

    if [ -z "$LATEST_RUN" ]; then
        echo "错误: 找不到 task '$TASK' 对应的 run 目录"
        echo "请检查: $RUNS_DIR/"
        echo "或使用方式2直接指定完整路径"
        exit 1
    fi

    CHECKPOINT_DIR="$LATEST_RUN"
fi

# 检查 checkpoint 是否存在
if [ ! -d "$CHECKPOINT_DIR/checkpoints" ]; then
    echo "错误: checkpoint 目录不存在: $CHECKPOINT_DIR/checkpoints"
    exit 1
fi

# 激活 Python 虚拟环境
source .venv-psi/bin/activate

# 指定使用的 GPU
export CUDA_VISIBLE_DEVICES=0
echo "============================================"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Task: $(basename $CHECKPOINT_DIR)"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Step: $CKPT_STEP"
echo "============================================"

# 启动 FastAPI RTC 服务器
python src/psi/deploy/psi_serve_rtc-trainingtimertc.py \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir="$CHECKPOINT_DIR" \
    --ckpt-step="$CKPT_STEP"
