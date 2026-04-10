# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ψ₀ (Psi0) is a vision-language-action (VLA) foundation model for dexterous humanoid loco-manipulation. The architecture consists of:
- **System-2**: Qwen3-VL-2B-Instruct vision-language backbone
- **System-1**: Flow-based multimodal diffusion transformer action expert (~500M params)
- **System-0**: RL-based tracking controller for lower-body control

## Environment Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate

# Install dependencies (skip lfs to avoid large file issues)
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match --active

# Install flash attention separately
uv pip install flash_attn==2.7.4.post1 --no-build-isolation

# Verify installation
python -c "import psi; print(psi.__version__)"
python -c "from psi.data.lerobot.compat import LEROBOT_LAYOUT; print(LEROBOT_LAYOUT)"
```

## Key Environment Variables

Set up `.env` file (copy from `.env.sample`):
- `HF_TOKEN` - HuggingFace token for model/data download
- `WANDB_API_KEY`, `WANDB_ENTITY` - Weights & Biases logging
- `PSI_HOME` - Root directory for checkpoints, data (default: `~/Psi0`)
- `HF_LEROBOT_HOME=$PSI_HOME/data/lerobot`

## Training Commands

### Fine-tuning on Real Robot Data
```bash
# Real-world data (Unitree G1)
bash scripts/train/psi0/finetune-real-psi0.sh <task>

# Example
bash scripts/train/psi0/finetune-real-psi0.sh Hug_box_and_move
```

### Fine-tuning on Simulation Data (SIMPLE)
```bash
# Simulation data
bash scripts/train/psi0/finetune-simple-psi0.sh <task>
```

### Pre-training VLM Backbone
```bash
# Pre-train on EgoDex
bash scripts/train/psi0/pretrain-egodex-psi0-fast.sh

# Pre-train on Humanoid Everyday dataset
bash scripts/train/psi0/pretrain-he-psi0-fast.sh
```

### Post-training Action Expert
```bash
bash scripts/train/psi0/posttrain-he-psi0.sh
```

## Configuration System

Training uses a tyro-based config system. Configs are defined in `src/psi/config/train/`:
- `finetune_real_psi0_config.py` - Fine-tuning on real data
- `finetune_simple_psi0_config.py` - Fine-tuning on simulation
- `pretrain_*_config.py` - Pre-training configs
- `posttrain_*_config.py` - Post-training configs

Training is launched via `scripts/train.py` which:
1. Imports the config module (first argument, e.g., `finetune_real_psi0_config`)
2. Parses command-line overrides with tyro
3. Instantiates a `Trainer` and runs training

## Data Processing

### Download Data from HuggingFace
```bash
hf download USC-PSI-Lab/psi-data \
  real/<task>.zip \
  --local-dir=$PSI_HOME/data \
  --repo-type=dataset
```

### Convert Raw Data to LeRobot Format
```bash
python scripts/data/raw_to_lerobot.py \
  --data-root=$PSI_HOME/data/real_teleop_g1/g1_real_raw \
  --work-dir=$PSI_HOME/data/real \
  --repo-id=psi0-real-g1 \
  --robot-type=g1 \
  --task=$task
```

### Calculate Stats
```bash
python scripts/data/calc_modality_stats.py \
  --work-dir=$PSI_HOME/data/real \
  --task=$task
```

## Deployment

### Serve Psi0 (RTC mode - real-time control)
```bash
bash ./scripts/deploy/serve_psi0-rtc.sh
```

### Start Psi0 Client (on robot)
```bash
cd real && bash ./scripts/deploy_psi0-rtc.sh
```

### Serve for SIMPLE evaluation
```bash
uv run --active --group psi --group serve serve_psi0 \
  --host 0.0.0.0 --port 22085 \
  --run-dir=<run_dir> --ckpt-step=<step> \
  --action-exec-horizon=24 --rtc
```

## Evaluation

### Open-Loop Evaluation (offline)
See `examples/simple/openloop_eval.ipynb` for notebook-based evaluation.

### SIMPLE Simulation Evaluation
Requires Docker with Isaac Sim. See `examples/simple/README.md` for details.

## Directory Structure

```
src/psi/
├── config/           # Configuration schemas (data, model, training)
│   └── train/        # Training config modules
├── data/             # Data handling
│   ├── lerobot/      # LeRobot dataset format support
│   ├── egodex/       # EgoDex dataset processing
│   └── humanoid/     # Humanoid everyday dataset
├── models/           # Model definitions (psi0.py, qwen3vl_mixin.py, etc.)
├── trainers/         # Training loop implementation
├── deploy/           # Deployment/inference code
├── utils/            # Utilities
└── __init__.py       # Package init with version

scripts/
├── train.py          # Main training entry point
├── train/            # Training shell scripts
│   └── psi0/         # Psi0-specific training scripts
├── deploy/           # Deployment scripts
└── data/             # Data processing scripts

real/                 # Real-world deployment
├── teleop/           # Teleoperation system (AVP, PICO, etc.)
├── deploy/           # Real robot deployment scripts
└── README.md         # Detailed real-world setup guide

third_party/SIMPLE/   # Simulation benchmark (git submodule)
```

## Key Models

- **Psi0Model** (`src/psi/models/psi0.py`) - Main VLA model combining VLM + diffusion action expert
- **Qwen3VLMixin** (`src/psi/models/qwen3vl_mixin.py`) - Vision-language backbone
- **DiffusionPolicyG1**, **ACT_G1**, **DiffusionPolicyG1** - Baseline policies

## LeRobot Dataset Format

Data is stored in LeRobot format under `$PSI_HOME/data/`:
```
task_name/
├── meta/
│   ├── info.json
│   └── stats_psi0.json     # Normalization statistics
├── train/
│   └── ...
└── val/
    └── ...
```

## Troubleshooting

1. **Lerobot import error (stack issue)**: Re-sync PSI env
   ```bash
   uv sync --group psi --active
   ```

2. **Missing Python.h for evdev**: Install python3-dev and build essentials
   ```bash
   sudo apt install python3-dev build-essential linux-headers-$(uname -r)
   ```

3. **libtorchcodec load error**: Install ffmpeg
   ```bash
   sudo apt-get install ffmpeg
   ```
