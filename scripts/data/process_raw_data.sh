#!/bin/bash
# =============================================================================
# 将原始遥操作数据转换为 LeRobot 格式
# =============================================================================
# 该脚本的每个步骤都可以独立复制运行。
# 只需要设置好相应的环境变量即可。
#
# 使用方法:
#   ./process_raw_data.sh <任务名称> [步骤编号]
#
# 示例:
#   ./process_raw_data.sh Hug_box_and_move        # 运行所有步骤
#   ./process_raw_data.sh Hug_box_and_move 1     # 仅运行步骤1
#
# 环境要求:
#   - HF_TOKEN 环境变量（仅用于下载数据）
# =============================================================================

set -e

TASK=${1:-""}
STEP=${2:-"all"}

# -----------------------------------------------------------------------------
# 步骤1: 下载原始数据,自己手机的数据可以跳过
# -----------------------------------------------------------------------------
step1_download() {
    # -------------------- 复制以下内容运行 --------------------
    # 设置环境变量（根据你的实际情况修改）
    export TASK=${TASK:-"Hug_box_and_move"}
    export REPO_ID="USC-PSI-Lab/psi-data"

    DATA_ROOT="${PSI_HOME}/data/real_teleop_g1"
    mkdir -p "${DATA_ROOT}/g1_real_raw"

    # 如果你需要代理，先在 shell 中设置 PROXY 或 ALL_PROXY
    export PROXY="socks5h://192.168.88.10:2084"
    if [[ -n "${PROXY:-}" ]]; then
        export http_proxy="$PROXY"
        export https_proxy="$PROXY"
        export HTTP_PROXY="$PROXY"
        export HTTPS_PROXY="$PROXY"
        export all_proxy="$PROXY"
        export ALL_PROXY="$PROXY"
        echo "使用代理: $PROXY"
    fi

    hf download "${REPO_ID}" \
        "g1_real_raw/${TASK}.zip" \
        --local-dir="${DATA_ROOT}" \
        --repo-type=dataset
   
    # ----------------------------------------------------------
}

# -----------------------------------------------------------------------------
# 步骤2: 解压数据,真机数据从这一步开始运行
# -----------------------------------------------------------------------------
step2_unzip() {
    # -------------------- 复制以下内容运行 --------------------
    export TASK=${TASK:-"Hug_box_and_move"}  # 修改为你的任务名称
    
    # 所有数据放到${PSI_HOME}/data/real_teleop_g1目录下
    DATA_ROOT="${PSI_HOME}/data/real_teleop_g1"
    unzip -o "${DATA_ROOT}/g1_real_raw/${TASK}.zip" \
        -d "${DATA_ROOT}/g1_real_raw/${TASK}"
    # ----------------------------------------------------------
}

# -----------------------------------------------------------------------------
# 步骤3: 转换为 LeRobot 格式
# -----------------------------------------------------------------------------
step3_convert() {
    # -------------------- 复制以下内容运行 --------------------
    export TASK=${TASK:-"Hug_box_and_move"} # 修改为你的任务名称

    # 如果使用自己的数据，修改 DATA_ROOT 为你的原始数据路径
    # 原始数据应符合以下结构:
    # DATA_ROOT/g1_real_raw/$TASK/episode_0/color/frame_xxx.jpg
    # DATA_ROOT/g1_real_raw/$TASK/episode_0/data.json
    # 如果不一样请新建转换脚本
    DATA_ROOT="${PSI_HOME}/data/real_teleop_g1"
    WORK_DIR="${PSI_HOME}/data/real"

    python scripts/data/raw_to_lerobot.py \
        --data-root="${DATA_ROOT}/g1_real_raw" \
        --work-dir="${WORK_DIR}" \
        --repo-id=psi0-real-g1 \
        --robot-type=g1 \
        --task=${TASK}
    # ----------------------------------------------------------
}

# -----------------------------------------------------------------------------
# 步骤4: 计算模态统计信息
# -----------------------------------------------------------------------------
step4_calc_stats() {
    # -------------------- 复制以下内容运行 --------------------
    export TASK=${TASK:-"Hug_box_and_move"}

    WORK_DIR="${PSI_HOME}/data/real"

    python scripts/data/calc_modality_stats.py \
        --work-dir="${WORK_DIR}" \
        --task=${TASK}
    # ----------------------------------------------------------
}

# -----------------------------------------------------------------------------
# 步骤5: 创建 Ψ₀ 格式统计文件
# -----------------------------------------------------------------------------
step5_copy_psi0() {
    # -------------------- 复制以下内容运行 --------------------
    export TASK=${TASK:-"Hug_box_and_move"}

    WORK_DIR="${PSI_HOME}/data/real"
    cp "${WORK_DIR}/${TASK}/meta/stats.json" \
       "${WORK_DIR}/${TASK}/meta/stats_psi0.json"
    # ----------------------------------------------------------
}

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
show_help() {
    echo "使用方法: $0 <任务名称> [步骤编号]"
    echo ""
    echo "步骤:"
    echo "  1 - 下载原始数据"
    echo "  2 - 解压数据"
    echo "  3 - 转换为 LeRobot 格式"
    echo "  4 - 计算模态统计信息"
    echo "  5 - 创建 Ψ₀ 格式统计文件"
    echo ""
    echo "示例:"
    echo "  $0 Hug_box_and_move      # 运行所有步骤"
    echo "  $0 Hug_box_and_move 3    # 仅运行步骤3"
}

if [[ -z "$TASK" ]]; then
    show_help
    exit 1
fi

case "$STEP" in
    1) step1_download ;;
    2) step2_unzip ;;
    3) step3_convert ;;
    4) step4_calc_stats ;;
    5) step5_copy_psi0 ;;
    all)
        step1_download
        step2_unzip
        step3_convert
        step4_calc_stats
        step5_copy_psi0
        echo ""
        echo "✓ 全部完成！数据位于: ${PSI_HOME}/data/real/${TASK}"
        ;;
    *) show_help ;;
esac
