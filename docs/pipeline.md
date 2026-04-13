# Psi0 训练 Pipeline 详解

> 基于 `finetune-real-psi0.sh` 配置，从数据构造到模型前向的完整流程。

---

## 阶段 0: 数据准备

### 0.1 数据存储格式 (LeRobot Format)

```
$PSI_HOME/data/real/<task>/
├── meta/
│   ├── info.json
│   └── stats_psi0.json          # 归一化统计量 (min/max)
├── train/
│   └── ...
└── val/
    └── ...
```

### 0.2 原始数据字段

每条样本 (从 LeRobotDataset 读取) 包含:
- `observation.images.egocentric`: 相机图像 (H×W×3)
- `states`: 机器人本体状态 (To×Da)
- `action`: 动作序列 (Tp×Da)
- `task`: 文本指令 (str)

---

## 阶段 1: Dataset 创建 (`finetune.py:284`)

```
LerobotDataConfig.__call__(split="train")
    │
    ├── 1. LeRobotDatasetWrapper(data_cfg, split="train")
    │       └── src/psi/data/lerobot/lerobot_ext.py:47
    │       └── 内部调用 lerobot 包的 LeRobotDataset
    │
    └── 2. Dataset(data_cfg, wrapper, transform_kwargs)
            └── src/psi/data/dataset.py:15
```

---

## 阶段 2: DataLoader 创建 (`finetune.py:326`)

```
torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,               # train_batch_size=16
    collate_fn=PaddedCollatorForTogether(...),
    num_workers=12,
    drop_last=True,
    shuffle=True,
    persistent_workers=True,
)
```

---

## 阶段 3: 单条数据迭代 `Dataset.__getitem__` (`dataset.py:23`)

### 步骤 1: 从磁盘读取原始数据

```
raw = raw_dataset[idx]
```

从 LeRobotDataset 读取，原始字段:
- `observation.images.egocentric`: 图像 tensor
- `states`: numpy
- `action`: numpy
- `task`: str

---

## 阶段 4: Transform 流水线 (`DataTransform.__call__`) (`transform.py:46`)

---

### Transform 步骤 1: RepackTransform → `RealRepackTransform.__call__` (`transform.py:338`)

**目的:** 提取字段 + padding 到固定维度

**配置 (`finetune-real-psi0.sh`):**
```
pad_action_dim=36, pad_state_dim=36
action_chunk_size=30
```

**流程:**

```
输入: LeRobotDataset 原始数据
│
├── 步骤 1a: padding states
│   states → pad_to_len → (To, 36)
│
├── 步骤 1b: padding actions + 生成 mask
│   actions → pad_to_len → (30, 36)
│   mask: 有效位置=True, pad位置=False
│
└── 步骤 1c: 构建输出 dict
    observations: [PIL Image list]      # 图像列表
    states: np.array (30, 36)            # padded
    actions: np.array (30, 36)           # padded
    instruction: str                     # 任务描述
    actions_mask: (30, 36)               # 有效位掩码
```

---

### Transform 步骤 2: FieldTransform → `ActionStateTransform.__call__` (`transform.py:485`)

**目的:** Bounds 归一化 actions/states 到 [-1, 1]

**配置:**
```
stat_path=meta/stats_psi0.json
action_norm_type=bounds
normalize_state=True
use_norm_mask=True
```

**流程:**

```
输入: RealRepackTransform 输出
│
├── 步骤 2a: 处理近零范围 (防止除零)
│   ill_mask = |max - min| < threshold
│
├── 步骤 2b: bounds 归一化
│   action_normalized = (action - min) / (max - min) * 2 - 1
│
├── 步骤 2c: 应用归一化 mask (gripper 等某些维度保留原始值)
│   actions = where(mask, normalized, original)
│
└── 步骤 2d: clip 到 [-1, 1]，保存原始值
    data["raw_actions"] = original
    data["actions"] = clip(normalized, -1, 1)
```

**关键:** 归一化后 actions 范围 [-1, 1]，便于 diffusion 模型处理。

---

### Transform 步骤 3: ModelTransform → `Qwen3vlModelTransform.__call__` (`transform.py:628`)

**目的:** VLM 图像预处理 + 构建模型输入

**配置:**
```
resize.size=(240, 320)
img_aug=True
adaptive_resize=True
```

**流程:**

```
输入: ActionStateTransform 输出
│
├── 步骤 3a: 图像预处理
│   observations (List[PIL])
│     → Resize((240, 320))
│     → ColorJitter (可选)
│   images: List[PIL]
│
├── 步骤 3b: 构建 VLM 输入 (`build_qwenvl_inputs`)
│   │
│   ├── 步骤 3b-1: 构建 messages
│   │   user: [image, image, instruction]
│   │   assistant: [tokenized_action]
│   │
│   ├── 步骤 3b-2: 应用 chat template
│   │   messages → prompt texts
│   │
│   ├── 步骤 3b-3: process_vision_info
│   │   提取 image patch 信息 (image_patch_size=16)
│   │
│   └── 步骤 3b-4: vlm_processor
│       texts + images → input_ids, pixel_values, image_grid_thw
│
├── 步骤 3c: 构建 labels
│   labels = input_ids.clone()
│   labels[:, :-(num_answer_tokens+2)] = IGNORE_INDEX
│   # 仅对 answer + EOS 计算 loss
│
└── 步骤 3d: 附加元数据
    raw_actions: 归一化前的 actions
    raw_images: 预处理后的图像 (可视化用)
    dataset_name: 数据集名称
```

**输出:**
```
input_ids:       (seq_len,)        # tokenized 文本
attention_mask:  (seq_len,)        # 有效位掩码
pixel_values:    (2, 1536)         # 2张图，每张1536维
image_grid_thw:  (2, 3)            # 每张图的 grid 尺寸
actions:         (30, 36)         # 归一化后的 action
states:          (1, 36)           # 归一化后的状态
actions_mask:    (30, 36)          # 有效位掩码
raw_actions:     (30, 36)          # 原始 action
raw_images:      (2, H, W, 3)      # 原始图像
```

---

## 阶段 5: Batch 构造 `PaddedCollatorForTogether.__call__` (`qwen3vl_mixin.py:94`)

**目的:** 将多个 sample 合并为一个 batch，处理变长序列

```
输入: list of samples (batch_size=16)
│
├── 步骤 5a: 提取各字段
│   input_ids: [instance["input_ids"] for instance in instances]
│   pixel_values: [instance["pixel_values"] for instance in instances]
│
├── 步骤 5b: pad_sequence (input_ids)
│   input_ids: 右填充到 model_max_length
│   attention_mask: 有效位置=1, padding=0
│
├── 步骤 5c: torch.stack
│   pixel_values: 堆叠 → (B, 2, 1536)
│   image_grid_thw: 堆叠 → (B, 2, 3)
│   actions: 堆叠 → (B, 30, 36)
│   states: 堆叠 → (B, 1, 36)
│   actions_mask: 堆叠 → (B, 30, 36)
│   raw_images: 堆叠 → (B, 2, H, W, 3)
│
└── 输出 batch dict:
    {
        "input_ids": (B, seq_len),
        "attention_mask": (B, seq_len),
        "pixel_values": (B, 2, 1536),
        "image_grid_thw": (B, 2, 3),
        "actions": (B, 30, 36),
        "states": (B, 1, 36),
        "actions_mask": (B, 30, 36),
        "raw_actions": (B, 30, 36),
        "raw_images": (B, 2, H, W, 3),
    }
```

---

## 阶段 6: 训练循环 (`train.py:233`)

```
for epoch in range(...):
    for local_step, batch in enumerate(train_dataloader):
        │
        ├── 步骤 6a: batch_str_to_tensor (utils.py)
        │   将 batch 中 "_str" 结尾的序列化字段反序列化
        │
        ├── 步骤 6b: trainer.step(batch, global_step, local_step)
        │
        └── ...
```

---

## 阶段 7: 模型前向传播

### 7.1 Trainer.step (`finetune.py:346`)

```
step()
  └── training_step(batch_input)
        └── forward_and_loss(model, batch)
```

### 7.2 加噪 (`forward_and_loss`) (`finetune.py:627`)

**配置:**
```
noise_scheduler=flow
train_diffusion_steps=1000
rtc=True
```

```
输入: batch["actions"] (B, 30, 36)
│
├── 步骤 7a: 采样 timesteps
│   sigmas = rand(B,) ∈ [0, 1)
│   timesteps = sigmas * 1000
│   # RTC 模式: 生成 prefix_mask
│
├── 步骤 7b: Flow matching 加噪
│   noisy_actions = (1-sigma) * clean_actions + sigma * noise
│   target_action = noise - clean_actions  # velocity prediction
│
└── 输出:
    noisy_actions: (B, 30, 36)
    target_action: (B, 30, 36)
    timesteps: (B,)
```

### 7.3 Psi0Model.forward (`psi0.py:1602`)

```
输入:
  input_ids, attention_mask,
  pixel_values, image_grid_thw,
  action_samples=noisy_actions,
  states, timestep, traj2ds=None
│
├── 步骤 7c: VLM 提取视觉-语言特征
│   vlm_hidden_states = vlm_model(
│       input_ids, attention_mask, pixel_values, image_grid_thw
│   ).hidden_states[-1]  # (B, seq_len, hidden_dim)
│
└── 步骤 7d: Action Header 前向
    └── ActionTransformerModel.forward
```

### 7.4 ActionTransformerModel.forward (`psi0.py:1080`)

```
输入:
  action_samples: (B, 30, 36)  # noisy actions
  views: (B, V, N, D)         # VLM visual tokens
  obs: (B, 1, 36)             # robot states
  timestep: (B,)              # noise level
│
├── 步骤 7e: 时间步嵌入
│   temb = time_ins_embed(timestep)  # (B, action_hidden_dim)
│
├── 步骤 7f: Action 输入投影
│   action_hidden = action_proj_in(noisy_actions)
│   # (B, 30, 36) → (B, 30, action_hidden_dim)
│
├── 步骤 7g: 观测编码 (ObservationProjection.forward)
│   │
│   │   你的配置: n_conditions=0, use_film=False
│   │   │
│   │   views (B, V, N, 1920)
│   │     → views_proj → (B, V, N, action_hidden_dim)
│   │     → view_tokens.view(B, V*N, D)
│   │     → concat obs_token (B, 1, action_hidden_dim)
│   │     → post_proc → obs_hidden (B, S, action_hidden_dim)
│   │
│   └── obs_hidden, obs_mask = obs_proj(...)
│       # obs_hidden: (B, S, action_hidden_dim)
│
├── 步骤 7h: VLA Transformer Block 堆叠 (N=6)
│   for block in transformer_blocks:
│       action_hidden, obs_hidden = block(
│           action_hidden,   # A: action 特征
│           obs_hidden,      # O: 观测特征
│           temb,           # T: 时间步嵌入
│           obs_mask        # mask
│       )
│   │
│   └── 核心交叉注意力:
│       action (query) attend to obs (key/value)
│       obs attend to action
│
└── 步骤 7i: Action 输出投影
    action_pred = action_proj_out(action_hidden)
    # (B, 30, action_hidden_dim) → (B, 30, 36)
```

### 7.5 Loss 计算 (`forward_and_loss`) (`finetune.py:627`)

```
输入:
  action_pred: (B, 30, 36)    # 模型预测
  target_action: (B, 30, 36)  # 真实 velocity
│
├── MSE Loss: F.mse_loss(action_pred, target_action)
│   # (B, 30, 36)
│
├── 步骤 7j: 应用 mask + 加权
│   loss = (mse * actions_mask).sum(1)  # 跨时间步
│   loss = (loss.mean(0) * loss_w).sum()  # 跨 batch + 加权
│
└── 输出:
    loss: scalar
```

---

## 阶段 8: 反向传播 + 参数更新 (`training_step`) (`finetune.py:399`)

```
training_step(batch_input)
│
├── 步骤 8a: 梯度累积上下文
│   gac = accelerator.accumulate(model)
│
├── with gac:
│   ├── 步骤 8b: 前向传播
│   │   losses = forward_and_loss(model, batch)
│   │
│   ├── 步骤 8c: 反向传播
│   │   accelerator.backward(losses["loss"])
│   │
│   └── if sync_gradients:
│       ├── 步骤 8d: 梯度裁剪
│       │   clip_grad_norm_(max_grad_norm=1.0)
│       │
│       ├── 步骤 8e: 优化器更新
│       │   optimizer.step()
│       │   lr_scheduler.step()
│       │   optimizer.zero_grad()
│       │
│       └── 步骤 8f: 日志记录
│           trainer.log(losses)
│
└── 返回: (sync_gradients, metrics)
```

---

## 完整数据流图

```
磁盘 (LeRobot Format)
    │
    ▼
LeRobotDataset[idx]
    │
    ▼
RealRepackTransform
    ├── 提取图像/状态/action
    ├── pad_to_len (36维)
    │
    ▼
ActionStateTransform
    ├── bounds 归一化 (min/max → [-1,1])
    │
    ▼
Qwen3vlModelTransform
    ├── 图像 resize + 增强
    ├── vlm_processor → input_ids, pixel_values
    │
    ▼
PaddedCollatorForTogether
    ├── pad_sequence (input_ids)
    ├── torch.stack (其他字段)
    │
    ▼
DataLoader ─── batch_size=16
    │
    ▼
batch
    │
    ▼
forward_and_loss
    ├── Flow matching 加噪
    │
    ▼
Psi0Model.forward
    ├── VLM (Qwen3-VL) → visual features
    │
    ▼
ActionTransformerModel.forward
    ├── time_ins_embed (timestep)
    ├── action_proj_in (noisy actions)
    ├── obs_proj (views + states)
    ├── VLATransformerBlock × 6 (交叉注意力)
    ├── action_proj_out
    │
    ▼
action_pred (B, 30, 36)
    │
    ▼
MSE Loss + backward
```

---

## 关键配置汇总

| 配置项 | 值 | 阶段 |
|--------|-----|------|
| `train_batch_size` | 16 | DataLoader |
| `pad_action_dim` | 36 | Repack |
| `pad_state_dim` | 36 | Repack |
| `action_chunk_size` | 30 | Repack |
| `noise_scheduler` | flow | forward_and_loss |
| `train_diffusion_steps` | 1000 | forward_and_loss |
| `normalize_state` | True | ActionStateTransform |
| `n_conditions` | 0 | ObservationProjection |
| `use_film` | False | ObservationProjection |
| `resize.size` | (240, 320) | Qwen3vlModelTransform |
| `tune_vlm` | False | init_models |
| `action_hidden_dim` | 1536 | ActionTransformerModel |
| `action_num_blocks` | 6 | ActionTransformerModel |

---

# Psi0 RTC 部署 Pipeline (Sonic 机器人)

> 基于 `serve_psi0-rtc-sonic.sh` 脚本，Psi0 模型在 Sonic 机器人上的实时控制部署流程。

---

## 部署架构概览

```
Sonic 机器人 (客户端)                              Psi0 RTC 服务器
┌──────────────────────┐                    ┌──────────────────────────────────┐
│ psi-inference_rtc.py │                    │ psi_serve_rtc-trainingtimertc-   │
│                      │                    │ sonic.py                         │
│                      │                    │                                  │
│ ┌──────────────────┐ │   WebSocket /ws   │ ┌────────────────────────────┐ │
│ │ 1. 相机采集        │ │ ─────────────────► │ │ 5. 解析观察 payload       │ │
│ │ RSCamera 获取     │ │                    │ │ - 图像预处理              │ │
│ │ RGB 图像          │ │                    │ │ - 状态归一化              │ │
│ └──────────────────┘ │                    │ └─────────────┬───────────────┘ │
│ ┌──────────────────┐ │                    │               │                 │
│ │ 2. 读取机器人状态 │ │                    │               ▼                 │
│ │ arm_joints (14维) │ │                    │ ┌────────────────────────────┐ │
│ │ hand_joints (14维)│ │                    │ │ 6. RealTimeChunkController │ │
│ └──────────────────┘ │                    │ │ - 推理线程循环            │ │
│ ┌──────────────────┐ │                    │ │ - 延迟补偿                │ │
│ │ 3. 构建 payload  │ │                    │ └─────────────┬───────────────┘ │
│ │ JSON 序列化       │ │                    │               │                 │
│ └────────┬─────────┘ │                    │               ▼                 │
│          │           │                    │ ┌────────────────────────────┐ │
│          ▼           │                    │ │ 7. 模型推理               │ │
│ ┌──────────────────┐ │                    │ │ predict_action_with_      │ │
│ │ 4. WebSocket     │ │                    │ │ training_rtc_flow          │ │
│ │ 发送观察         │ │                    │ │ - Qwen3-VL 视觉编码        │ │
│ └──────────────────┘ │                    │ │ - Flow Diffusion (8步)    │ │
│                      │                    │ │ - 输出 action[30,36]      │ │
│ ┌──────────────────┐ │                    │ └─────────────┬───────────────┘ │
│ │ 10. 接收 action   │ │ ◄──────────────── │               │                 │
│ │ (36维向量)       │ │                    │               ▼                 │
│ └────────┬─────────┘ │                    │ ┌────────────────────────────┐ │
│          │           │                    │ │ 8. 反归一化 action         │ │
│          ▼           │                    │ └─────────────┬───────────────┘ │
│ ┌──────────────────┐ │                    │               │                 │
│ │ 11. 解析 action   │ │                    │               ▼                 │
│ │ [0:14]   hand     │ │                    │ ┌────────────────────────────┐ │
│ │ [14:28]  arm      │ │                    │ │ 9. WebSocket              │ │
│ │ [28:32]  rpyh     │ │                    │ │ 发送 action               │ │
│ │ [32:36]  vx/vy    │ │                    │ └────────────────────────────┘ │
│ └────────┬─────────┘ │                    └──────────────────────────────────┘
│          │
│          ▼
│ ┌──────────────────┐
│ │ 12. IK 求解       │
│ │ solve_whole_body_ │
│ │ ik                │
│ └────────┬─────────┘
│          │
│          ▼
│ ┌──────────────────┐
│ │ 13. 下位机控制    │
│ │ ctrl_whole_body  │
│ └──────────────────┘
└──────────────────────┘
```

---

## 启动流程

### 1. 服务器启动 (`serve_psi0-rtc-sonic.sh`)

```bash
bash scripts/deploy/serve_psi0-rtc-sonic.sh <checkpoint_dir> <ckpt_step>
```

关键参数:
- `--rtc`: 启用 RTC 模式
- `--action_exec_horizon 30`: 执行视野
- `--port 8014`: WebSocket 端口

### 2. 客户端启动 (`psi-inference_rtc.py`)

```bash
cd real
python deploy/psi-inference_rtc.py --host <server_ip> --port 8014
```

---

## 核心组件详解

### 6. RealTimeChunkController (`psi_serve_rtc-trainingtimertc-sonic.py`)

实时动作控制器，管理 Psi0 模型的推理循环。

**核心参数:**
```python
PREDICT_HORIZON = 30      # H: 每次预测 30 步 action
MIN_EXEC_HORIZON = 15    # s_min: 最小执行步数，用于延迟补偿
DELAY_BUFFER_SIZE = 6     # 延迟缓冲区大小
D_INIT = 6               # d_init: 初始延迟估计
CTRL_PERIOD_SEC = 1/30   # 控制频率 30Hz
```

---

#### 初始化流程

```
1. 首次预测: A_first = _predict_action(o_first)  # (30, 36)
   └── 使用第一个观察调用普通 predict_action，生成初始 action 序列

2. 模型预热: 执行 2 次 _predict_action_rtc
   └── 让 GPU/CUDA 环境稳定，减少首次推理延迟波动

3. 启动推理线程: _inference_loop (daemon thread)
   └── 独立运行，不断预取下一个 action 序列
```

---

#### 推理循环详解 (_inference_loop)

**step() 方法是什么?**

```python
def step(self, obs_next):
    """
    被控制循环 (30Hz) 调用:
    1. 消费 obs_next (执行完上一 action 后的观察)
    2. 返回下一时刻的 action
    3. t += 1
    4. 通知推理线程有新的观察
    """
    with self.C:
        self.t += 1
        self.o_cur = obs_next
        self.C.notify()  # 通知推理线程
        single_action = self.A_cur[self.t - 1]
        return single_action[np.newaxis, :]  # (1, D)
```

**时间步 t 与控制循环的关系:**

```
时间步 t: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...
         └────────────────────────────────────────►
         ↑
         控制循环每 33ms (30Hz) 调用一次 step()

step() 的作用:
    - 每次调用返回 A_cur[t-1]
    - t += 1
    - 当 t >= s_min (15) 时，推理线程被唤醒
```

**完整流程:**

```
t=0:  控制循环调用 step()，t 变为 1，返回 A_cur[0]
t=1:  控制循环调用 step()，t 变为 2，返回 A_cur[1]
...
t=14: 控制循环调用 step()，t 变为 15，返回 A_cur[14]
      此时推理线程被唤醒 (t >= s_min)
t=15: 控制循环调用 step()，t 变为 16，返回 A_cur[15]
      推理线程同时开始执行推理...
```

**推理线程执行:**

```python
def _inference_loop(self):
    with self.C:
        while self.t < self.s_min:
            self.C.wait()  # 等待 t >= 15
        
        s = self.t  # 记录当前时间步 s
        
        # 深拷贝当前观察
        o = copy.deepcopy(self.o_cur)
        
        # 获取延迟估计 (deque Q 中的最大值)
        d = max(self.Q)
        
        # 构建之前的 action 序列 A_prev
        # A_cur[s:] 保持不变 (未执行的 action)
        # A_cur[:s] 置零 (已执行的 action)
        A_prev = concatenate([A_cur[s:], zeros(s, D)], axis=0)
        
        # 调用模型推理 (带延迟补偿)
        A_new = self._predict_action_rtc(o, A_prev, d)
        
        # 更新当前 action 序列
        self.A_cur = A_new
        
        # 重置 t (相对于新 action 序列的索引)
        self.t = self.t - s
        
        # 将实际执行步数添加到延迟估计队列
        self.Q.append(self.t)
```

---

#### 延迟补偿机制详解

**问题背景:**

```
客户端时间线:
t=0    t=1    t=2    t=3    t=4    t=5    t=6    ...
│      │      │      │      │      │      │
▼      ▼      ▼      ▼      ▼      ▼      ▼
obs0  obs1  obs2  obs3  obs4  obs5  obs6  ...
       │
       │ 网络延迟 ~50ms
       │ 推理延迟 ~100ms
       ▼
服务器接收到 obs1 时，实际时刻可能是 t=3
服务器执行 action[1] 时，实际对应客户端状态已经是 obs3
```

**延迟补偿公式:**

```
服务器在时刻 t 发送 action[t]
客户端在时刻 t + d 执行 action[t]
客户端在时刻 t + d + Δt 发送 obs[t+d+Δt] 给服务器

服务器收到的 obs 对应的是 action[t] 执行后的状态
因此服务器需要预测 action[t+d] 而不是 action[t]
```

**延迟估计滑动窗口:**

```python
self.Q = deque([d_init], maxlen=6)  # 初始延迟 6

# 每次推理完成后
actual_delay = self.t  # 从上一轮推理到现在执行的步数
self.Q.append(actual_delay)

# 估计延迟取最大值 (保守策略)
d = max(self.Q)  # 确保延迟不会低估
```

**为什么用最大值而不是平均值?**

```
延迟估计 = max(Q) 的原因:
1. 如果低估延迟，机器人可能执行"过时"的动作，导致不稳定
2. 使用最大值可以确保 action 更加"前瞻"
3. 牺牲一点响应速度，换取稳定性
```

**prev_actions 的作用:**

```python
# A_prev 构建
A_prev = concatenate([A_cur[s:], zeros(s, D)], axis=0)
#           ↑ 未执行部分    ↑ 已执行部分置零

# 作用: 告诉模型"之前执行了哪些 action"
# 模型据此预测"接下来应该执行什么 action"
# 这实现了 action 序列的连续性和一致性
```

---

### 7. Psi0 模型推理 (`predict_action_with_training_rtc_flow`)

```python
def _predict_action_rtc(self, o, A_prev, d):
    A_new = self.policy.predict_action_with_training_rtc_flow(
        observations=o['imgs'],           # 图像列表
        states=torch.from_numpy(o['obs']).to(self.device),  # 状态向量
        traj2ds=None,                     # 2D 轨迹 (不使用)
        instructions=o['text_instructions'],  # 文本指令
        num_inference_steps=8,            # Diffusion 步数
        prev_actions=torch.from_numpy(A_prev[np.newaxis, :, :]).to(self.device),  # (30, 36) -> (1, 30, 36)
        inference_delay=d,               # 延迟估计
        max_delay=8
    )[0].float().detach().cpu().numpy()  # (1, 30, 36) -> (30, 36)
    return A_new
```

**模型内部流程:**
```
输入:
  observations (图像)     → Qwen3-VL-2B 视觉编码
  states (32维)           → 状态编码
  instructions (文本)      → 文本编码

融合 → Flow-based Diffusion Transformer (8步 denoise)

输出: action[36维], shape (30, 36)
```

---

### 10. 客户端 Action 解析 (`psi-inference_rtc.py`)

**Action 向量布局 (36维):**

| 索引 | 维度 | 名称 | 说明 |
|------|------|------|------|
| `[0:14]` | 14 | `hand_cmd` | 手指关节指令 |
| `[14:28]` | 14 | `arm_cmd` | 手臂关节指令 |
| `[28:32]` | 4 | `rpyh` | 躯干 (roll, pitch, yaw, height) |
| `[32]` | 1 | `vx` | 底盘 x 方向速度 |
| `[33]` | 1 | `vy` | 底盘 y 方向速度 |
| `[34]` | 1 | `vyaw` | 底盘角速度 |
| `[35]` | 1 | `target_yaw` | 目标偏航角 |

**解析代码:**
```python
if have_vla:
    # 底盘运动指令
    vx = action[32]
    vy = action[33]
    vyaw = action[34]
    target_yaw = action[35]

    # 阈值过滤
    vx = 0.6 if vx > 0.25 else 0
    vy = 0 if abs(vy) < 0.3 else 0.5 * (1 if vy > 0 else -1)

    # 关节指令
    rpyh = action[28:32]   # 躯干姿态
    arm_cmd = action[14:28]  # 手臂关节
    hand_cmd = action[:14]   # 手指关节
```

---

### 12. IK 求解与 13. 下位机控制

**IK 求解 (`solve_whole_body_ik`):**
```python
pd_target, pd_tauff, raw_action = master.body_ik.solve_whole_body_ik(
    left_wrist=None,
    right_wrist=None,
    current_lr_arm_q=current_lr_arm_q,
    current_lr_arm_dq=current_lr_arm_dq,
    observation=master.observation,
    extra_hist=master.extra_hist,
    is_teleop=False,
)
```

**下位机控制 (`ctrl_whole_body`):**
```python
# 手臂/手指指令覆盖 IK 输出
if arm_cmd is not None:
    pd_target[15:] = arm_cmd
    tau_arm = get_tauer(arm_cmd)  # 关节角度 → 力矩
    pd_tauff[15:] = tau_arm

if hand_cmd is not None:
    master.hand_shm_array[:] = hand_cmd  # 写入共享内存

# 发送控制指令
master.body_ctrl.ctrl_whole_body(
    pd_target[15:], pd_tauff[15:],  # 手臂
    pd_target[:15], pd_tauff[:15]    # 腿部
)
```

---

## 数据流总结

```
客户端 (Sonic 机器人)
│
├── 1. 相机采集 → RGB 图像 (H, W, 3)
├── 2. 读取状态 → arm_joints[14], hand_joints[14]
├── 3. 构建 payload
├── 4. WebSocket 发送
│
▼ 网络传输 (JSON over WebSocket)
│
服务器 (Psi0 RTC)
├── 5. 解析 payload
├── 6. RealTimeChunkController 管理推理
├── 7. Psi0 模型推理 → action[30, 36]
├── 8. action 反归一化
└── 9. WebSocket 发送
│
▼ 网络传输 (JSON over WebSocket)
│
客户端
├── 10. 接收 action
├── 11. 解析 action 各部分
├── 12. IK 求解
└── 13. 下位机控制
```

---

## 关键配置汇总

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `PREDICT_HORIZON` | 30 | 预测视野 |
| `MIN_EXEC_HORIZON` | 15 | 最小执行步数 |
| `CTRL_PERIOD_SEC` | 1/30 | 控制频率 30Hz |
| `num_inference_steps` | 8 | Diffusion 步数 |
| `maxmin.normalize_state` | True | 状态归一化 |
| `action_dim` | 36 | action 维度 |
