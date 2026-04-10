#!/bin/bash
# ========================================================================
# Psi0 RTC 客户端启动脚本 (在机器人上运行)
# ========================================================================
# 作用: 启动机器人端的 Psi0 RTC 推理客户端，连接到服务器进行实时控制
# 流程:
#   1. 设置任务名称和端口
#   2. 切换到 teleop 目录
#   3. 启动 psi-inference_rtc.py 客户端程序
#
# 使用方法 (与 finetune-real-psi0.sh 对齐):
#   bash scripts/pipeline/real/deploy_psi0-rtc.sh <task> [server_ip]
#
# 示例:
#   bash scripts/pipeline/real/deploy_psi0-rtc.sh 
#   bash scripts/pipeline/real/deploy_psi0-rtc.sh Hug_box_and_move 192.168.1.100
#
# 参数:
#   task:     任务名称 (与训练时一致，如 Hug_box_and_move)
#   server_ip: 服务器 IP (默认 127.0.0.1)
#
# 使用前提:
#   - 服务器端 (serve_psi0-rtc.sh) 必须先启动
#   - 机器人相机、传感器等硬件已配置
# ========================================================================

# 通信端口 (需与服务器端一致)
PORT=8014

# 默认任务名称 (与 finetune 对齐)
DEFAULT_TASK="Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw"
# 默认服务器 IP
DEFAULT_SERVER_IP="127.0.0.1"

# 解析参数
if [ $# -lt 1 ]; then
    echo "用法: bash $0 <task> [server_ip]"
    echo "示例: bash $0 Hug_box_and_move 192.168.1.100"
    echo "默认 task: $DEFAULT_TASK"
    echo "默认 server: $DEFAULT_SERVER_IP"
    exit 1
fi

TASK="$1"
SERVER_IP="${2:-$DEFAULT_SERVER_IP}"

# 切换到 teleop 目录
# 使用 readlink -f 解析脚本真实路径，避免 symlink 导致 cd 错误
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$SCRIPT_DIR/../teleop"
export PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"

# Avoid ROS python packages shadowing deployment dependencies such as pinocchio.
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset ROS_LOCALHOST_ONLY
if [ -n "${PYTHONPATH:-}" ]; then
    CLEANED_PYTHONPATH="$(printf '%s' "$PYTHONPATH" | tr ':' '\n' | grep -v '^/opt/ros/' | paste -sd ':' -)"
    export PYTHONPATH="$CLEANED_PYTHONPATH"
fi

# 启动机器人端推理程序
# psi-inference_rtc.py:
#   - 接收相机图像和指令
#   - 发送到服务器进行推理
#   - 接收预测的 action 并执行
echo "============================================"
echo "Connecting to server: $SERVER_IP:$PORT"
echo "Task: $TASK"
echo "============================================"

python ../deploy/psi-inference_rtc.py \
    --port "$PORT" \
    --task "$TASK"

# 使用方法:
#   1. 在 GPU 服务器端运行:
#        bash scripts/pipeline/deploy/serve_psi0-rtc.sh Hug_box_and_move 10000
#   2. 在机器人端运行:
#        bash scripts/pipeline/real/deploy_psi0-rtc.sh Hug_box_and_move
#   3. 服务器监听 0.0.0.0:8014，客户端连接到该地址进行 RTC 控制
