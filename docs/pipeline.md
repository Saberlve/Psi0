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
