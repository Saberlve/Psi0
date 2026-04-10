# Psi0 Pipeline Guide

## Data And Training

先处理数据，再训练，再部署。

1. 看 `scripts/pipeline/data/` 下面的脚本，把原始数据转成 LeRobot 格式。
2. 计算 norm stats，并放到对应数据目录或配置里。
3. 修改 task 配置，确认训练使用的 task 名称、数据路径、stats 路径一致。
4. 用 `scripts/pipeline/finetune/` 下面的脚本训练。
5. 部署时按同一个 task 或直接按 run 目录启动。

## Serve Side

服务端在 GPU 机器上启动，负责加载 checkpoint 并提供 RTC 推理接口。

示例：

```bash
cd /home/ubuntu/Psi0
source .venv-psi/bin/activate

bash scripts/deploy/serve_psi0-rtc.sh \
  /home/ubuntu/Psi0/.runs/finetune/pick-toys.real.flow1000.cosine.lr1.0e-04.b16.gpus1.2604101000 \
  40000
```

也可以传 task 名称，让脚本自动找最新 run：

```bash
cd /home/ubuntu/Psi0
source .venv-psi/bin/activate

bash scripts/deploy/serve_psi0-rtc.sh \
  Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw \
  40000
```

注意：

- `serve_psi0-rtc.sh` 会读取 run 目录下的 `checkpoints/ckpt_<step>/model.safetensors`。
- 推理阶段除了加载 checkpoint，还需要加载 Qwen3-VL 的 `config` 和 `processor`。
- 当前代码已经改成优先使用 `run_config.json` 里的 `model.model_name_or_path`，因此离线部署时应确保这个路径是本地可读目录。

## Client Side

机器人端 RTC client 不建议使用 `.venv-psi`。推荐严格按照 `real/README.md` 使用单独的 `conda` 环境。

### Recommended Environment

```bash
cd /home/ubuntu/Psi0/real
conda env create -f psi_deploy_env.yaml
conda activate psi_deploy

pip install -e .
git clone https://github.com/physical-superintelligence-lab/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
cd /home/ubuntu/Psi0
```

然后启动 client：

```bash
conda activate psi_deploy
cd /home/ubuntu/Psi0

bash real/scripts/deploy_psi0-rtc.sh \
  Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw \
  127.0.0.1
```

## Why `.venv-psi` Often Fails For Real Deploy

`.venv-psi` 主要是训练 / 服务侧环境，不是 `real/README.md` 那套机器人部署环境。

常见问题：

- 缺少 `pinocchio`
- 缺少 `casadi`
- 缺少 `pin-pink`
- 缺少 `mujoco`
- 缺少 `unitree_sdk2py`

所以机器人端不要默认复用 `.venv-psi`。

## ROS Conflict

如果当前 shell source 过 ROS，例如 `/opt/ros/humble/setup.bash`，很容易污染 Python 路径。

典型现象：

```python
ImportError: cannot import name 'casadi' from 'pinocchio'
```

这通常不是项目代码错，而是 Python 先导入了：

```text
/opt/ros/humble/lib/python3.10/site-packages/pinocchio
```

而不是 `psi_deploy` 环境里的 Pinocchio。

当前 `real/scripts/deploy_psi0-rtc.sh` 已经加入了 ROS Python 路径清理逻辑，会在启动前清掉这些环境变量和 `/opt/ros/*` 的 `PYTHONPATH` 项。

但前提仍然是：你实际激活的环境里必须真的装了正确的 Pinocchio 及相关依赖。

## Known Working Rule

可以按下面的原则记：

- 服务端：用 `/home/ubuntu/Psi0/.venv-psi`
- 机器人端：用 `real/psi_deploy_env.yaml` 创建的 `conda` 环境

不要反过来。

## Quick Troubleshooting

### 1. 服务端启动时去 Hugging Face 下载

先检查 run 目录里的 `run_config.json`：

```bash
rg -n "model_name_or_path" /home/ubuntu/Psi0/.runs/finetune/<your_run>/run_config.json
```

要求这个路径指向本地 Qwen3-VL 基座目录，并且目录下至少有：

- `config.json`
- `tokenizer.json` 或其他 tokenizer 文件
- `preprocessor_config.json`

### 2. Client 报 `No module named teleop`

优先使用：

```bash
bash real/scripts/deploy_psi0-rtc.sh ...
```

不要直接在错误目录下手敲：

```bash
python real/deploy/psi-inference_rtc.py
```

### 3. Client 报 `cannot import name 'casadi' from 'pinocchio'`

先确认你不是在 `.venv-psi` 里跑，也不是在 ROS 污染过的 shell 里跑。

正确做法：

```bash
conda activate psi_deploy
cd /home/ubuntu/Psi0
bash real/scripts/deploy_psi0-rtc.sh ...
```

### 4. Client 报 `No module named unitree_sdk2py`

安装仓库里要求的修改版：

```bash
git clone https://github.com/physical-superintelligence-lab/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

## Summary

一条最重要的经验：

- 训练和服务，用 `.venv-psi`
- 真机 deploy，用 `psi_deploy` conda 环境

如果混用，通常会在 `teleop`、`pinocchio`、`mujoco`、`unitree_sdk2py` 这些依赖上反复出错。
