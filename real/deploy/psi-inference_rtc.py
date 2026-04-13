# =============================================================================
# Psi0 RTC 客户端 (Sonic 机器人版本)
# =============================================================================
# 功能: 运行在 Sonic 机器人端, 通过 WebSocket 与 Psi0 RTC 服务器通信
# 流程:
#   1. 采集相机图像和机器人状态 (关节角度)
#   2. 通过 WebSocket 发送到 Psi0 服务器
#   3. 接收服务器返回的 action
#   4. 通过 IK (逆运动学) 控制机器人执行 action
# =============================================================================
import os
import time
import threading
import json

import cv2
import numpy as np
import requests
import json_numpy

from multiprocessing import Array, Event
from teleop.master_whole_body import RobotTaskmaster
from teleop.robot_control.compute_tau import GetTauer
import zmq
from websocket import WebSocketApp

# 任务指令 - 可通过命令行 --task 参数覆盖
TASK_INSTRUCTION = "Spray the bowl and wipe it and stack it up."

# 控制频率 (Hz)
FREQ_CTRL = 60
# 观察发送间隔 (秒) - 目前未使用
OBS_SEND_INTERVAL = 0.01

# =============================================================================
# Step 1: JSON NumPy 序列化支持
# =============================================================================
# Psi0 服务器返回的 action 是 numpy array, 需要通过 base64 编码的 JSON 传输
json_numpy.patch()

from base64 import b64encode, b64decode
from numpy.lib.format import dtype_to_descr, descr_to_dtype

# =============================================================================
# Step 2: 相机接口初始化
# =============================================================================
# RSCamera: Intel RealSense 相机的 ZMQ 接口
# 相机通过 TCP 连接 (192.168.123.164:5556) 获取图像
class RSCamera:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        # 相机服务器地址 - 需要与实际相机配置匹配
        self.socket.connect("tcp://192.168.123.164:5556")

    def get_frame(self):
        """通过 ZMQ 请求一帧图像"""
        self.socket.send(b"get_frame")
        rgb_bytes, _, _ = self.socket.recv_multipart()
        # 将字节流解码为 numpy array (BGR 格式, OpenCV 默认)
        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        return rgb_image


# =============================================================================
# Step 3: 构建观察 (Observation)
# =============================================================================
def get_observation(camera, state):
    """
    构建发送给服务器的观察数据

    Args:
        camera: RSCamera 实例
        state: 机器人状态字典, 包含 arm_joints 和 hand_joints

    Returns:
        img_obs: 图像观察 (用于发送)
        state_obs: 状态观察 (用于发送)
    """
    # 获取原始图像 (H, W, 3) BGR 格式
    frame = camera.get_frame()
    # OpenCV BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.uint8)

    # 图像观察: 服务器端会提取 egocentric 视角
    img_obs = {
        "observation.images.egocentric": frame,
    }
    # 状态观察: 手臂关节 + 手指关节
    state_obs = {
        "arm_joints": np.array(state["arm_joints"]),
        "hand_joints": np.array(state["hand_joints"]),
    }
    return img_obs, state_obs


# =============================================================================
# Step 4: 共享内存与事件初始化
# =============================================================================
# 这些数据结构用于进程间/线程间通信
shared_data = {
    "kill_event": Event(),           # 强制终止事件
    "session_start_event": Event(),  # 会话开始事件
    "failure_event": Event(),       # 失败事件
    "end_event": Event(),            # 结束事件
    "dirname": None,
}
kill_event = shared_data["kill_event"]

# 共享内存数组 - 用于与下位机通信
# robot_shm_array: 512维, 存储机器人控制指令
robot_shm_array = Array("d", 512, lock=False)
# teleop_shm_array: 64维, 存储遥操作数据
teleop_shm_array = Array("d", 64, lock=False)

# =============================================================================
# Step 5: 机器人接口初始化 (RobotTaskmaster)
# =============================================================================
# RobotTaskmaster 是机器人控制的核心类, 负责:
#   - 获取机器人当前状态 (motorstate, handstate)
#   - 维护站立姿势
#   - IK 求解 (body_ik.solve_whole_body_ik)
#   - 下位机通信 (body_ctrl.ctrl_whole_body)
master = RobotTaskmaster(
    task_name="inference",
    shared_data=shared_data,
    robot_shm_array=robot_shm_array,
    teleop_shm_array=teleop_shm_array,
    robot="g1",  # Unitree G1 人形机器人
)

# GetTauer: 将关节角度指令转换为力矩指令
get_tauer = GetTauer()
camera = RSCamera()

# Action 缓冲区 - 线程安全地存储从服务器接收的最新 action
pred_action_buffer = {"actions": None}
pred_action_lock = threading.Lock()

running = Event()
running.set()


# =============================================================================
# Step 6: NumPy 序列化/反序列化工具
# =============================================================================
# WebSocket 只能传输 JSON, 因此 numpy array 需要序列化
def numpy_serialize(o):
    """将 numpy array 序列化为 JSON 可序列化的字典"""
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),  # base64 编码
            "dtype": dtype_to_descr(o.dtype),      # 数据类型描述
            "shape": o.shape,                       # 形状
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    """从 JSON 字典反序列化回 numpy array"""
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(dct["shape"]) if dct["shape"] else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    """递归地对字典/列表中的 numpy array 应用函数"""
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    else:
        return data


# =============================================================================
# Step 7: WebSocket 客户端 (与 Psi0 服务器通信)
# =============================================================================
# RealTime Control WebSocket 客户端
# 功能:
#   - 接收服务器下发的 action
#   - 发送机器人观察 (图像 + 状态) 给服务器
class RTCWebSocketClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self._running = True
        self._connected = threading.Event()
        self._ws = None
        self._send_lock = threading.Lock()
        self.start_time = time.time()

    # -------------------------------------------------------------------------
    # Step 7a: 执行接收到的 action
    # -------------------------------------------------------------------------
    def execute_action(self, action: np.ndarray):
        """
        实际只是将 action 存入缓冲区, 由控制线程读取并执行
        """
        with pred_action_lock:
            pred_action_buffer["actions"] = action

    def _on_open(self, ws):
        """WebSocket 连接建立回调"""
        print("[client] Connected!")
        self._connected.set()

    # -------------------------------------------------------------------------
    # Step 7b: 接收服务器消息 (action)
    # -------------------------------------------------------------------------
    # 服务器会发送 JSON 格式的 action, 包含:
    #   - action: numpy array (经过 base64 编码)
    #   - version: 版本号用于追踪
    def _on_message(self, ws, message):
        """Receive message from server (runs in WebSocketApp thread)"""
        interval = time.time() - self.start_time
        self.start_time = time.time()
        print(f"[client] recv_action interval: {interval} seconds")

        try:
            data = json.loads(message)
            action_data = data.get("action")
            version = data.get("version", -1)

            if action_data is not None:
                # 反序列化 action (从 base64 JSON 还原为 numpy array)
                action = convert_numpy_in_dict(action_data, numpy_deserialize)
                if isinstance(action, np.ndarray):
                    self.execute_action(action)
                    print(f"[client] Received action, version={version}")

        except Exception as e:
            print(f"[client] Message processing error: {e}")

    def _on_error(self, ws, error):
        """WebSocket 错误回调"""
        print(f"[client] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket 连接关闭回调"""
        print(f"[client] Connection closed: {close_status_code} - {close_msg}")
        self._running = False
        running.clear()

    # -------------------------------------------------------------------------
    # Step 7c: 发送观察线程
    # -------------------------------------------------------------------------
    # 持续采集机器人状态和图像, 通过 WebSocket 发送给服务器
    def _send_thread(self):
        """Send observations at high frequency"""
        print("[client] Send thread started")

        # 等待连接建立
        self._connected.wait()

        prev_tick = time.perf_counter()

        while self._running and running.is_set():
            start = time.time()
            try:
                # 获取机器人当前状态
                state = {
                    "arm_joints": master.motorstate[15:29],   # 手臂关节 (15-29)
                    "hand_joints": master.handstate,          # 手指关节
                }
                # 获取图像观察和状态观察
                img_obs, state_obs = get_observation(camera, state)

                # 构建发送给服务器的 payload
                # 注意: Psi0 服务器期望的格式由 RequestMessage 定义
                payload = {
                    "image": img_obs,
                    "state": state_obs,
                    "gt_action": None,           # 推理模式无 ground truth
                    "dataset_name": None,
                    "instruction": TASK_INSTRUCTION,  # 任务指令
                    "history": None,
                    "condition": None,
                    "timestamp": None,
                }
                # 序列化 numpy array 为 JSON
                payload = convert_numpy_in_dict(payload, numpy_serialize)
                message = json.dumps(payload)

                # 线程安全地发送消息
                with self._send_lock:
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                    else:
                        print("[client] WebSocket not connected, skipping send")
                        break

            except Exception as e:
                print(f"[client] Send error: {e}")
                break

            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            print(f"send thread interval: {interval} seconds")

        print("[client] Send thread stopped")

    # -------------------------------------------------------------------------
    # Step 7d: 启动 WebSocket 客户端
    # -------------------------------------------------------------------------
    def run(self):
        """Main client loop - runs WebSocketApp in current thread"""
        print(f"[client] Connecting to {self.server_url}")

        # 创建 WebSocketApp
        self._ws = WebSocketApp(
            self.server_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

        # 启动发送线程
        send_thread = threading.Thread(target=self._send_thread, daemon=True)
        send_thread.start()

        # 运行 WebSocketApp (阻塞直到连接关闭)
        self._ws.run_forever()

        # 等待发送线程结束
        self._running = False
        send_thread.join(timeout=2.0)

        print("[client] Client stopped")

    def stop(self):
        """停止客户端"""
        self._running = False
        if self._ws:
            self._ws.close()


# =============================================================================
# Step 8: 主控制循环 (Action 执行)
# =============================================================================
def main(server_url):
    """
    主入口函数

    流程:
    1. 初始化机器人站立姿势
    2. 启动控制线程和 WebSocket 线程
    3. 从 action 缓冲区读取 action 并执行
    """
    master.reset_yaw_offset = True

    _last_target_yaw = None

    # -------------------------------------------------------------------------
    # Step 8a: 从缓冲区读取 action 并应用到机器人
    # -------------------------------------------------------------------------
    def apply_action_from_buffer(last_pd_target):
        """
        这是控制循环的核心, 每次迭代都会:
        1. 读取最新的 action
        2. 解析 action 各部分含义
        3. 更新 master 的 IK 参数
        4. 调用 IK 求解
        5. 发送控制指令到下位机
        """
        current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()

        have_vla = False

        # 从缓冲区获取 action (线程安全)
        with pred_action_lock:
            action = pred_action_buffer["actions"]

            if action is not None:
                have_vla = True
                # action shape: (1, D) -> (D,) 取第一行
                action = action[0]

        arm_cmd = None
        hand_cmd = None
        if have_vla:
            """
            Step 8b: 解析 action 向量

            Psi0 action 向量布局 (D=36):
            - action[0:14]    = hand_cmd (14维) - 手指关节指令
            - action[14:28]   = arm_cmd (14维) - 手臂关节指令
            - action[28:32]   = rpyh (4维) - 躯干 roll, pitch, yaw, height
            - action[32]      = vx (1维) - 底盘 x 方向速度
            - action[33]      = vy (1维) - 底盘 y 方向速度
            - action[34]      = vyaw (1维) - 底盘角速度
            - action[35]      = target_yaw (1维) - 目标偏航角
            """
            if action.shape[0] < 36:
                print("[CTRL] Invalid action shape:", action.shape)
            else:
                # 底盘运动指令
                vx = action[32]
                vy = action[33]
                vyaw = action[34]
                target_yaw = action[35]

                # 阈值过滤 - 去除微小动作
                vx = 0.6 if vx > 0.25 else 0
                vy = 0 if abs(vy) < 0.3 else 0.5 * (1 if vy > 0 else -1)

                # 关节指令
                rpyh = action[28:32]   # 躯干姿态
                arm_cmd = action[14:28]  # 手臂关节
                hand_cmd = action[:14]   # 手指关节

                # 更新 master 的 IK 参数
                master.torso_roll = rpyh[0]
                master.torso_pitch = rpyh[1]
                master.torso_yaw = rpyh[2]
                master.torso_height = rpyh[3]

                print("torso_roll, pitch, yaw, height:", master.torso_roll, master.torso_pitch, master.torso_yaw, master.torso_height)

                # 底盘速度指令
                master.vx = vx
                master.vy = vy
                master.vyaw = vyaw
                master.target_yaw = target_yaw

                # 保存上一步的状态 (用于平滑过渡)
                master.prev_torso_roll = master.torso_roll
                master.prev_torso_pitch = master.torso_pitch
                master.prev_torso_yaw = master.torso_yaw
                master.prev_torso_height = master.torso_height

                master.prev_vx = master.vx
                master.prev_vy = master.vy
                master.prev_vyaw = master.vyaw
                master.prev_target_yaw = master.target_yaw

                master.prev_arm = arm_cmd
                master.prev_hand = hand_cmd

        # 如果没有收到 VLA action, 保持上一步的姿态
        if not have_vla:
            master.torso_roll = master.prev_torso_roll
            master.torso_pitch = master.prev_torso_pitch
            master.torso_yaw = master.prev_torso_yaw
            master.torso_height = master.prev_torso_height

            master.vx = master.prev_vx
            master.vy = 0  # 无 action 时停止移动
            master.vyaw = master.prev_vyaw
            master.target_yaw = master.prev_target_yaw

        """
        Step 8c: IK 求解

        调用 body_ik.solve_whole_body_ik 求解全身逆运动学:
        - 输入: 当前关节位置、观察数据、期望末端执行器位置
        - 输出: pd_target (位置目标), pd_tauff (力矩前馈)
        """
        master.get_ik_observation(record=False)

        pd_target, pd_tauff, raw_action = master.body_ik.solve_whole_body_ik(
            left_wrist=None,
            right_wrist=None,
            current_lr_arm_q=current_lr_arm_q,
            current_lr_arm_dq=current_lr_arm_dq,
            observation=master.observation,
            extra_hist=master.extra_hist,
            is_teleop=False,
        )

        # 保存原始 action 用于下次迭代
        master.last_action = np.concatenate([
            raw_action.copy(),
            (master.motorstate - master.default_dof_pos)[15:] / master.action_scale,
        ])

        # 如果有手臂/手指指令, 覆盖 IK 输出
        if arm_cmd is not None:
            pd_target[15:] = arm_cmd
            # 将关节角度转换为力矩
            tau_arm = np.asarray(get_tauer(arm_cmd), dtype=np.float64).reshape(-1)
            pd_tauff[15:] = tau_arm

        if hand_cmd is not None:
            # 写入共享内存, 手指控制器会读取
            with master.dual_hand_data_lock:
                master.hand_shm_array[:] = hand_cmd

        """
        Step 8d: 下位机控制

        将控制指令发送到下位机:
        - pd_target[15:]: 手臂关节位置目标
        - pd_tauff[15:]: 手臂关节力矩前馈
        - pd_target[:15]: 腿部关节位置目标
        - pd_tauff[:15]: 腿部关节力矩前馈
        """
        master.body_ctrl.ctrl_whole_body(
            pd_target[15:], pd_tauff[15:], pd_target[:15], pd_tauff[:15]
        )

        return pd_target

    # -------------------------------------------------------------------------
    # Step 8e: 控制循环线程
    # -------------------------------------------------------------------------
    # 以固定频率 (60Hz) 调用 apply_action_from_buffer
    def control_loop_thread():
        dt = 1.0 / FREQ_CTRL
        last_pd_target = None
        while running.is_set() and not kill_event.is_set():
            try:
                last_pd_target = apply_action_from_buffer(last_pd_target)
            except Exception as e:
                print("[CTRL] loop error:", e)
            time.sleep(dt)
        print("[CTRL] Control loop stopped.")

    # -------------------------------------------------------------------------
    # Step 8f: WebSocket 通信线程
    # -------------------------------------------------------------------------
    def websocket_thread():
        client = RTCWebSocketClient(server_url=server_url)
        client.run()
        print("[WS] WebSocket thread stopped")

    # -------------------------------------------------------------------------
    # Step 8g: 主初始化和运行
    # -------------------------------------------------------------------------
    try:
        # 初始化阶段 - 保持站立姿势 30 秒
        stabilize_thread = threading.Thread(target=master.maintain_standing, daemon=True)
        stabilize_thread.start()
        master.episode_kill_event.set()
        print("[MAIN] Initialize with standing pose...")
        time.sleep(30)
        master.episode_kill_event.clear()

        master.reset_yaw_offset = True

        # 启动控制线程和 WebSocket 线程
        t_ctrl = threading.Thread(target=control_loop_thread, daemon=True)
        t_ctrl.start()

        t_ws = threading.Thread(target=websocket_thread, daemon=True)
        t_ws.start()

        print("[MAIN] Running. Ctrl+C to stop.")

        # 主循环等待终止事件
        while not kill_event.is_set() and running.is_set():
            time.sleep(0.5)

        print("[MAIN] kill_event set, preparing to stop...")
        running.clear()
        time.sleep(0.5)

        # 返回站立姿势
        master.episode_kill_event.set()
        print("[MAIN] Returning to standing pose for 5s...")
        time.sleep(5)
        master.episode_kill_event.clear()

    except KeyboardInterrupt:
        print("[MAIN] Caught Ctrl+C, exiting...")
        running.clear()
        kill_event.set()
    finally:
        shared_data["end_event"].set()
        master.stop()
        print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--task", type=str, default=TASK_INSTRUCTION)
    args = parser.parse_args()
    TASK_INSTRUCTION = args.task
    server_url = f"ws://{args.host}:{args.port}/ws"
    main(server_url)
