# =============================================================================
# Psi0 RTC 服务器 - Sonic 机器人版本
# =============================================================================
# 功能: 作为 WebSocket 服务器, 接收机器人客户端的观察并返回 action
#
# 架构流程:
#   客户端 (Sonic 机器人)                    服务器 (Psi0 RTC)
#   ┌─────────────────────┐                ┌─────────────────────┐
#   │ 1. 采集图像+状态    │                │                     │
#   │ 2. 发送 WebSocket   │ ──────────────► │  接收观察           │
#   │                    │                │                     │
#   │                    │                │  3. 预处理          │
#   │                    │                │     (图像 + 状态)   │
#   │                    │                │                     │
#   │                    │                │  4. 推理            │
#   │                    │                │     (Psi0 模型)     │
#   │                    │                │                     │
#   │ 6. 执行 action     │ ◄────────────── │  5. 返回 action     │
#   │   (IK 控制)        │                │                     │
#   └─────────────────────┘                └─────────────────────┘
#
# Psi0 模型推理流程:
#   observations (图像) + states + instructions
#         │
#         ▼
#   ┌─────────────────────────────────┐
#   │   Qwen3-VL-2B (视觉编码器)       │
#   │   图像 → 视觉特征               │
#   └─────────────────────────────────┘
#         │
#         ▼
#   ┌─────────────────────────────────┐
#   │   特征融合层                     │
#   │   视觉 + 文本指令 + 状态         │
#   └─────────────────────────────────┘
#         │
#         ▼
#   ┌─────────────────────────────────┐
#   │   Flow-based Diffusion Transformer │
#   │   动作预测 (denoise 过程)       │
#   └─────────────────────────────────┘
#         │
#         ▼
#   action (36维向量)
# =============================================================================
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from pydantic import BaseModel
from collections import deque
import threading
import time
import copy

import os
import sys
import json
import tyro
import uvicorn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List
import os.path as osp

from psi.models.psi0 import Psi0Model

# 确保导入路径正确
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from psi.config.config import LaunchConfig, ServerConfig
from psi.deploy.helpers import *

from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything
from psi.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


# =============================================================================
# RTC 控制参数
# =============================================================================
PREDICT_HORIZON = 30          # H: 预测视野, 即每次预测多少步 action
MIN_EXEC_HORIZON = 15         # s_min: 最小执行视野, 用于延迟补偿
DELAY_BUFFER_SIZE = 6         # 延迟缓冲区大小
D_INIT = 6                    # d_init: 初始延迟估计
CTRL_PERIOD_SEC = 1. / 30     # 控制频率 30Hz



# =============================================================================
# Step 2: Real-Time Chunk Controller (RTC) - 实时动作控制器
# =============================================================================
class RealTimeChunkController:
    """
    实时动作控制器 - 管理 Psi0 模型的推理循环

    核心思想: 每次推理生成 H 步 action, 但只执行前 s_min 步,
    然后利用已执行的动作作为先验, 继续推理接下来的 H 步。
    这样可以有效减少推理延迟的影响。

    延迟补偿机制:
    - 客户端发送 obs[t] 时, 服务器实际执行的是 action[t-d]
    - d 是网络延迟 + 推理延迟的估计
    - 通过滑动窗口 Q 动态估计延迟
    """
    def __init__(self,
                 policy: Psi0Model,
                 prediction_horizon: int = PREDICT_HORIZON,
                 min_exec_horizon: int = MIN_EXEC_HORIZON,
                 delay_buf_size: int = DELAY_BUFFER_SIZE,
                 d_init: int = D_INIT,
                 o_first: np.ndarray | None = None): # type: ignore

        self.policy : Psi0Model = policy
        self.device = self.policy.device
        self.H     = prediction_horizon  # 预测视野 = 30

        self.s_min = min_exec_horizon   # 最小执行步数 = 15

        self.t: int = 0  # 当前时间步

        assert o_first != None, "please provide o_first"

        # =======================================================================
        # Step 2a: 初始化推理 - 首次预测
        # =======================================================================
        # 使用第一个观察进行首次预测, 得到 H 步 action
        A_first = self._predict_action(o_first)  # shape: (H, D) = (30, 36)

        # =======================================================================
        # Step 2b: 模型预热 (Warmup)
        # =======================================================================
        # 预热 2 次, 让模型状态稳定, 减少首次推理的延迟波动
        for i in range (2):
            # 构建之前的 action 序列: 将已执行的 action 置零
            A_prev = np.concatenate([copy.deepcopy(A_first[self.s_min:, :]), np.zeros((self.s_min, A_first.shape[1]), dtype=A_first.dtype)], axis=0)
            _ = self._predict_action_rtc(copy.deepcopy(o_first), A_prev, d_init)
        print("Model warmed up")

        self.A_cur = A_first  # 当前预测的 action 序列 (H, D)
        self.o_cur: Dict[str, Any] | None = None  # 当前观察

        # =======================================================================
        # Step 2c: 延迟估计队列
        # =======================================================================
        # 使用 deque 维护最近 delay_buf_size 个延迟估计
        self.Q = deque([d_init], maxlen=delay_buf_size)

        # =======================================================================
        # Step 2d: 线程同步原语
        # =======================================================================
        self.M = threading.Lock()           # 主锁
        self.C = threading.Condition(self.M)  # 条件变量用于线程通信

        # =======================================================================
        # Step 2e: 启动推理线程
        # =======================================================================
        # 推理线程独立运行, 不断预取下一个 action 序列
        self._infer_th = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_th.start()

    # -------------------------------------------------------------------------
    # Step 2f: 主 step 函数 - 被控制循环调用
    # -------------------------------------------------------------------------
    def step(self, obs_next: Dict[str, Any]): # consume a_(t-1) and provide o_t
        """
        Args:
            obs_next: 下一个观察 (来自客户端执行完上一 action 后的状态)

        Returns:
            单步 action, shape: (1, D)
        """
        with self.C:
            self.t += 1
            self.o_cur = obs_next
            self.C.notify()  # 通知推理线程有新的观察

            # 返回当前时间步对应的 action
            if self.t - 1 >= len(self.A_cur):
                single_action = self.A_cur[-1]
                print("failed")
            else:
                single_action = self.A_cur[self.t - 1]
            return single_action[np.newaxis, :]  # (D,) -> (1, D)

    # -------------------------------------------------------------------------
    # Step 2g: 推理线程主循环
    # -------------------------------------------------------------------------
    def _inference_loop(self):
        """
        流程:
        1. 等待新观察到达 (Condition.wait)
        2. 积累足够多的观察后开始推理 (t >= s_min)
        3. 调用 _predict_action_rtc 进行带延迟补偿的推理
        4. 更新 A_cur 和 t
        """
        while True:
            with self.C:
                try:
                    # 等待直到有足够的观察 (t >= s_min)
                    while self.t < self.s_min:
                        self.C.wait()

                    s = self.t  # 当前时间步

                    assert (s-2) >= 0

                    # 深拷贝当前观察
                    o = copy.deepcopy(self.o_cur)
                    # 获取当前延迟估计 (使用最近延迟的最大值)
                    d = max(self.Q)

                    # 构建之前的 action 序列
                    # 将已执行的 action 置零, 未执行的保持
                    A_prev = np.concatenate([copy.deepcopy(self.A_cur[s:, :]), np.zeros((s, self.A_cur.shape[1]), dtype=self.A_cur.dtype)], axis=0)

                    # =================================================================
                    # Step 2h: 调用 Psi0 进行 RTC 推理 (带延迟补偿)
                    # =================================================================
                    inference_start = time.perf_counter()
                    self.C.release()  # 释放锁让主线程可以继续
                    A_new = self._predict_action_rtc(o, A_prev, d)
                    self.C.acquire()

                    self.A_cur = A_new
                    self.t = self.t - s
                    self.Q.append(self.t)
                    print(f"[inference]  latency={time.perf_counter()-inference_start:.4f}s  s={s}  d={d}  self.t={self.t}")
                except Exception as e:
                    print(f"\n[ERROR] Inference loop crashed!")
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print("\n[FATAL] Stopping program...")
                    os._exit(1)  # 强制退出整个程序

    # -------------------------------------------------------------------------
    # Step 2i: Psi0 RTC 推理 (带延迟补偿的 action 预测)
    # -------------------------------------------------------------------------
    def _predict_action_rtc(self, o, A_prev, d):
        """
        Args:
            o: 观察 (包含图像、状态、指令)
            A_prev: 之前的 action 序列 (用于作为 diffusion 的条件)
            d: 当前估计的延迟

        Returns:
            A_new: 预测的 action 序列, shape: (H, D)
        """
        # 调用 Psi0 模型的 predict_action_with_training_rtc_flow 方法
        # 该方法内部:
        #   1. 使用 Qwen3-VL 编码图像
        #   2. 融合文本指令和状态
        #   3. 使用 Flow-based Diffusion Transformer 生成 action
        A_new = self.policy.predict_action_with_training_rtc_flow(
                    observations=o['imgs'],           # 图像列表
                    states=torch.from_numpy(o['obs']).to(self.device),  # 状态向量
                    traj2ds=None,                      # 2D 轨迹 (不使用)
                    instructions=o['text_instructions'],  # 文本指令
                    num_inference_steps=8,            # Diffusion 步数
                    prev_actions=torch.from_numpy(A_prev[np.newaxis, :, :]).to(self.device),  # (H, D) -> (1, H, D)
                    inference_delay=d,                # 延迟估计
                    max_delay=8
                )[0].float().detach().cpu().numpy()  # (1, H, D) -> (H, D)
        return A_new

    # -------------------------------------------------------------------------
    # Step 2j: 普通 Psi0 推理 (无延迟补偿)
    # -------------------------------------------------------------------------
    def _predict_action(self, o):
        """
        用于首次预测和 warmup
        """
        normalized_actions = self.policy.predict_action(
                    observations=o['imgs'],
                    states=torch.from_numpy(o['obs']).to(self.device),
                    traj2ds=None,
                    instructions=o['text_instructions'],
                    num_inference_steps = 8,
                )[0].float().detach().cpu().numpy()  # (1, H, D) -> (H, D)

        return normalized_actions


# =============================================================================
# Step 3: Server 类 - WebSocket 服务器主类
# =============================================================================
class Server:
    """
    Psi0 RTC WebSocket 服务器

    功能:
    - 加载训练好的 Psi0 模型
    - 接收客户端的观察 (图像 + 状态)
    - 通过 RealTimeChunkController 管理推理
    - 通过 WebSocket 发送 action 给客户端
    """

    def __init__(
        self,
        policy: str,
        run_dir: Path,
        ckpt_step: int | str = "latest",
        device: str = "cuda:0",
        enable_rtc: bool = False,
        action_exec_horizon: int | None = None
    ):
        """
        初始化服务器

        Args:
            policy: 策略名称 (如 "psi")
            run_dir: checkpoint 目录
            ckpt_step: checkpoint 步数
            device: 运行设备
            enable_rtc: 是否启用 RTC 模式
            action_exec_horizon: action 执行视野
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

        self.device = torch.device(device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {policy}")

        # =========================================================================
        # Step 3a: 检查路径和加载配置
        # =========================================================================
        assert osp.exists(run_dir), f"run_dir {run_dir} does not exist!"
        assert osp.exists(run_dir / "checkpoints" / f"ckpt_{ckpt_step}"), f"ckpt {ckpt_step} does not exist!"
        assert osp.exists(run_dir / "run_config.json"), f"run config does not exist!"

        # 从 argv.txt 动态构建配置
        config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")
        # 从保存的 JSON 加载完整配置
        conf = (run_dir / "run_config.json").open("r").read()
        launch_config = config_.model_validate_json(conf)
        seed_everything(launch_config.seed or 42)

        # =========================================================================
        # Step 3b: 加载 Psi0 模型
        # =========================================================================
        overwatch.info("loading action model...")
        from psi.models.psi0 import Psi0Model
        self.model = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device=device)
        self.model.to(device)
        self.model.eval()

        # 加载数据变换 (用于归一化/反归一化)
        from psi.config.transform import SimpleRepackTransform, Psi0ModelTransform, ActionStateTransform
        self.maxmin: ActionStateTransform = launch_config.data.transform.field
        self.model_transform: Psi0ModelTransform = launch_config.data.transform.model

        # =========================================================================
        # Step 3c: 设置模型参数
        # =========================================================================
        self.Da = launch_config.model.action_dim           # action 维度 (36)
        self.Tp = launch_config.model.action_chunk_size    # action chunk 大小
        self.Ta = action_exec_horizon or launch_config.model.action_exec_horizon  # 执行视野
        assert self.Ta <= self.Tp, "action_exec_horizon is too big"
        self.launch_cfg = launch_config
        self.count = 0

        # =========================================================================
        # Step 3d: 初始化线程安全的共享状态
        # =========================================================================
        self.latest_obs = None        # 最新接收的观察
        self.latest_action = None      # 最新预测的 action
        self.action_version = 0       # action 版本号 (用于客户端追踪)

        self.obs_lock = threading.Lock()     # 观察锁
        self.action_lock = threading.Lock()  # action 锁

        self.controller = None              # RealTimeChunkController 实例
        self._control_loop_started = False  # 控制循环是否已启动

        # =========================================================================
        # Step 3e: 初始化 FastAPI 和 WebSocket
        # =========================================================================
        self.app = FastAPI()
        self._setup_routes()

        self._action_ready_event: asyncio.Event = None
        self._active_websocket: WebSocket = None
        self._loop = None
        self.start_time = time.time()
        self.start_time_obs = time.time()

    # =============================================================================
    # Step 3f: 初始化 RTC 控制器
    # =============================================================================
    def _init_controller(self, o_first):
        """使用第一个观察初始化 RealTimeChunkController"""
        controller = RealTimeChunkController(policy=self.model, o_first=o_first)
        return controller

    # =============================================================================
    # Step 3g: Action 后处理 - 反归一化
    # =============================================================================
    def _postprocess_action(self, action):
        """对预测的 action 进行反归一化"""
        return self.maxmin.denormalize(action)  # 反归一化在 pipeline 中完成

    # =============================================================================
    # Step 3h: 图像预处理
    # =============================================================================
    def preprocess_image(self, image_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        对输入图像进行预处理

        预处理步骤:
        1. 调整大小 (resize)
        2. 中心裁剪 (center crop)
        """
        imgs = {}
        for k in image_dict.keys():
            imgs[k] = self._process_img(image_dict[k])
        return imgs

    def _process_img(self, img):
        """单张图像的预处理流水线"""
        from torchvision.transforms import v2
        transforms = [self.model_transform.resize(), self.model_transform.center_crop()]
        t = v2.Compose(transforms)
        return [t(img)]

    # =============================================================================
    # Step 3i: 解析客户端发送的观察数据
    # =============================================================================
    def _parse_obs_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析从客户端接收的 JSON payload, 转换为模型输入格式

        Returns:
            解析后的观察字典, 包含:
            - imgs: 预处理后的图像列表
            - text_instructions: 文本指令列表
            - obs: 归一化后的状态向量
            - conditions: 额外条件 (当前为空)
        """
        request = RequestMessage.deserialize(payload)
        image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
                    request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name
        condition_dict = request.condition
        overwatch.info(f"Instruction: {instruction}")

        # 指令小写处理
        instruction = instruction.lower()

        # =========================================================================
        # 处理图像: numpy array -> PIL Image
        # =========================================================================
        imgs = {}
        for cam_idx, img in enumerate(image_dict.values()):
            imgs[f"cam{cam_idx}"] = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        # =========================================================================
        # 处理状态: 归一化
        # =========================================================================
        obs = state_dict['states'].copy()  # shape: (33,) 原始关节状态

        # 状态归一化
        assert self.maxmin.normalize_state, "check"
        # 填充到标准维度
        if self.maxmin.pad_state_dim != len(obs):
            obs = pad_to_len(obs, self.maxmin.pad_state_dim, dim=0)[0]
        # 归一化函数
        obs = self.maxmin.normalize_state_func(obs)  # shape: (32,)
        # 调整形状为 (1, 1, 32) 以匹配模型输入
        obs = obs[np.newaxis, np.newaxis, :]

        # 预处理图像
        image_input = self.preprocess_image(imgs)
        batch_images = [image_input['cam0']]  # batch size == 1

        conditions = {}
        text_instructions = [instruction]  # len == 1

        return {'imgs': batch_images, 'text_instructions': text_instructions, 'obs': obs, 'conditions': conditions}

    # =============================================================================
    # Step 3j: WebSocket 处理器
    # =============================================================================
    async def websocket_handler(self, websocket: WebSocket):
        """
        WebSocket handler for bidirectional communication:
        - Receive obs from client at high frequency
        - Send action to client immediately when new action is ready

        双向通信处理:
        1. receive_obs: 持续接收客户端发送的观察数据
        2. send_action: 当有新的 action 时立即发送给客户端
        """
        await websocket.accept()
        self._active_websocket = websocket

        # 创建 asyncio Event 用于 action 就绪通知
        self._action_ready_event = asyncio.Event()

        print("[WebSocket] Client connected")

        # -------------------------------------------------------------------------
        # 接收观察线程 - 持续接收客户端发送的观察数据
        # -------------------------------------------------------------------------
        async def receive_obs():
            """
            流程:
            1. 接收 JSON payload
            2. 解析为观察格式
            3. 更新 latest_obs
            4. 如果控制循环未启动, 则启动它
            """
            try:
                while True:
                    # 接收客户端数据
                    data = await websocket.receive_text()
                    payload = json.loads(data)
                    interval = time.time() - self.start_time_obs
                    self.start_time_obs = time.time()
                    print(f"[WebSocket] receive_obs interval: {interval} seconds")

                    # 解析 payload 为观察格式
                    this_o = self._parse_obs_payload(payload)

                    # 线程安全地更新最新观察
                    with self.obs_lock:
                        self.latest_obs = this_o

                    # 如果控制循环未启动, 启动它
                    if not self._control_loop_started and self.latest_obs is not None:
                        self._start_control_loop()

            except WebSocketDisconnect:
                print("[WebSocket] Client disconnected (receive)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[WebSocket] Receive error: {e}")

        # -------------------------------------------------------------------------
        # 发送 Action 线程 - 当有新的 action 时发送给客户端
        # -------------------------------------------------------------------------
        async def send_action():
            """
            流程:
            1. 等待 action 就绪事件
            2. 获取最新的 action
            3. 序列化为 JSON 并发送
            """
            try:
                while True:
                    # 等待新的 action 就绪
                    await self._action_ready_event.wait()
                    self._action_ready_event.clear()

                    interval = time.time() - self.start_time
                    self.start_time = time.time()
                    print(f"[WebSocket] send_action interval: {interval} seconds")

                    # 获取 action
                    with self.action_lock:
                        action = self.latest_action
                        version = self.action_version
                        self.latest_action = None  # 发送后重置

                    if action is not None:
                        # 使用 ResponseMessage 序列化 action
                        response = ResponseMessage(action, err=0.0)
                        resp_dict = response.serialize()
                        resp_dict["version"] = version
                        await websocket.send_text(json.dumps(resp_dict))
                        print(f"[WebSocket] Sent action, version={version}")
                    else:
                        assert False, "action is None"

            except WebSocketDisconnect:
                print("[WebSocket] Client disconnected (send)")
            except Exception as e:
                print(f"[WebSocket] Send error: {e}")

        try:
            # Run both tasks concurrently
            await asyncio.gather(receive_obs(), send_action())
        except Exception as e:
            print(f"[WebSocket] Connection closed: {e}")
        finally:
            self._active_websocket = None
            print("[WebSocket] Handler finished")

    # =============================================================================
    # Step 3k: 启动控制循环
    # =============================================================================
    def _start_control_loop(self):
        """
        首次调用时会初始化 RealTimeChunkController
        """
        if self._control_loop_started:
            return
        self._control_loop_started = True

        # 使用第一个观察初始化控制器
        with self.obs_lock:
            o_first = copy.deepcopy(self.latest_obs)

        # 初始化 RTC 控制器 (内部会 warmup 模型)
        self.controller = self._init_controller(o_first)

        # 启动控制循环线程
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        print("[control loop] started")

    # =============================================================================
    # Step 3l: 控制循环 - 定时调用 controller.step
    # =============================================================================
    def _control_loop(self):
        """
        控制循环 - 以固定频率 (30Hz) 运行

        流程:
        1. 获取最新观察
        2. 调用 controller.step 获取 action
        3. 对 action 反归一化
        4. 更新 latest_action 并通知 WebSocket
        5. 等待下一个控制周期
        """
        next_tick = time.perf_counter()
        prev_tick = time.perf_counter()

        while True:
            # =================================================================
            # Step 3l-1: 获取最新观察
            # =================================================================
            with self.obs_lock:
                obs_next = copy.deepcopy(self.latest_obs)

            # =================================================================
            # Step 3l-2: 调用 controller.step 获取 action
            # =================================================================
            action = self.controller.step(obs_next)  # (1, D)
            pred_action = self._postprocess_action(action)  # (1, D) 反归一化

            # =================================================================
            # Step 3l-3: 更新 latest_action
            # =================================================================
            with self.action_lock:
                self.latest_action = pred_action
                self.action_version += 1

            # =================================================================
            # Step 3l-4: 通知 WebSocket 有新的 action
            # =================================================================
            if self._action_ready_event is not None:
                try:
                    # 线程安全地设置 asyncio event
                    self._loop.call_soon_threadsafe(self._action_ready_event.set)
                except Exception as e:
                    print(f"[control loop] Failed to notify WebSocket: {e}")

            # =================================================================
            # Step 3l-5: 等待下一个控制周期
            # =================================================================
            next_tick += CTRL_PERIOD_SEC
            sleep_time = next_tick - time.perf_counter()
            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            print(f"[control loop] interval: {interval} seconds")
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[control loop] WARNING: missed tick by {-sleep_time*1000:.1f}ms")
                next_tick = time.perf_counter()


    # =============================================================================
    # Step 3m: 设置 FastAPI 路由
    # =============================================================================
    def _setup_routes(self):
        """设置 FastAPI 路由"""
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket 端点 /ws"""
            self._loop = asyncio.get_event_loop()
            await self.websocket_handler(websocket)

        @self.app.get("/health")
        async def health_check():
            """健康检查端点"""
            return {"status": "ok"}

    # =============================================================================
    # Step 3n: 启动服务器
    # =============================================================================
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        启动 uvicorn 服务器

        Args:
            host: 监听地址
            port: 监听端口
        """
        print(f"Server listens on {host}:{port}")
        print(f"WebSocket endpoint: ws://{host}:{port}/ws")
        try:
            uvicorn.run(self.app, host=host, port=port)
        except Exception as e:
            print(f"Server crashed, {e}")
        finally:
            print("Server stopped.")
            exit(1)


# ==============================================================================
# Step 4: 入口函数
# ==============================================================================
def serve(cfg: ServerConfig) -> None:
    """
    服务入口函数

    Args:
        cfg: ServerConfig 配置对象, 包含:
            - policy: 策略名称
            - run_dir: checkpoint 目录
            - ckpt_step: checkpoint 步数
            - device: 运行设备
            - rtc: 是否启用 RTC 模式
            - action_exec_horizon: action 执行视野
            - host: 服务器监听地址
            - port: 服务器监听端口
    """
    overwatch.info("Server :: Initializing Policy")
    assert cfg.policy is not None, "which policy to serve?"
    assert cfg.rtc, "this server is for rtc"

    # 创建 Server 实例
    server = Server(
        cfg.policy,
        Path(cfg.run_dir),
        cfg.ckpt_step,
        cfg.device,
        cfg.rtc,
        cfg.action_exec_horizon)

    print("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)


# ==============================================================================
# Step 5: 命令行入口
# ==============================================================================
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()  # 从 .env 文件加载环境变量

    # 使用 tyro 解析命令行参数为 ServerConfig
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)