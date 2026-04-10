#!/bin/bash

export OMP_NUM_THREADS=24
export CUDA_VISIBLE_DEVICES=0

source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 <task> [exp]"
    echo "Example: bash /home/ubuntu/Psi0/scripts/train/psi0/finetune-real-psi0.sh Hug_box_and_move box-move"
    exit 1
fi

export task="$1"  # ⚠️ 修改或命令行传入任务名称
task_words=$(echo "$task" | tr '[:upper:]' '[:lower:]' | tr '_' ' ')
default_exp=$(echo "$task_words" | awk '{if (NF>=2) print $1 "-" $2; else print $1}')
export exp=${2:-$default_exp}  # ⚠️ 修改或命令行传入实验名称

echo "Task: $task"
echo "Experiment name: $exp"



# ============ config ============
config="finetune_real_psi0_config"  # 配置模块名

# ============ train ============
seed="--seed=292285"  
exp="--exp=$exp"  # 实验名称
train_name="--train.name=finetune"  
train_data_parallel="--train.data_parallel=ddp"  
train_mixed_precision="--train.mixed_precision=bf16"  
train_train_batch_size="--train.train_batch_size=16"  # ⚠️训练批次大小
train_max_checkpoints_to_keep="--train.max_checkpoints_to_keep=5"  # 最多保留checkpoint数
train_gradient_accumulation_steps="--train.gradient_accumulation_steps=1" 
train_learning_rate="--train.learning_rate=1e-4"  # 学习率
train_max_training_steps="--train.max_training_steps=40000"  # ⚠️最大训练步数
train_warmup_ratio="--train.warmup_ratio=None"  # 预热比例
train_warmup_steps="--train.warmup_steps=1000"  # 预热步数
train_checkpointing_steps="--train.checkpointing_steps=5000"  # ⚠️checkpoint保存间隔
train_validation_steps="--train.validation_steps=1000"  # ⚠️验证间隔
train_val_num_batches="--train.val_num_batches=20"  # 验证批次数量
train_max_grad_norm="--train.max_grad_norm=1.0"  
train_lr_scheduler_type="--train.lr_scheduler_type=cosine"  
train_lr_scheduler_kwargs_weight_decay="--train.lr_scheduler_kwargs.weight_decay=1e-6" 
train_lr_scheduler_kwargs_betas="--train.lr_scheduler_kwargs.betas 0.95 0.999"  

# ============ log ============
log_report_to="--log.report_to=wandb"  # 日志报告工具

# ============ data ============
data_root_dir="--data.root_dir=$PSI_HOME/data/real"  # 数据根目录
data_train_repo_ids="--data.train_repo_ids=$task"  # 训练任务ID,启动脚本时动态传入
data_transform_repack_pad_action_dim="--data.transform.repack.pad-action-dim=36"  # action维度padding
data_transform_repack_pad_state_dim="--data.transform.repack.pad-state-dim=36"  # state维度padding
data_transform_field_stat_path="--data.transform.field.stat-path=meta/stats_psi0.json"  # 统计数据路径
data_transform_field_stat_action_key="--data.transform.field.stat-action-key=action"  # action统计key
data_transform_field_stat_state_key="--data.transform.field.stat-state-key=states"  # state统计key
data_transform_field_action_norm_type="--data.transform.field.action_norm_type=bounds"  # action归一化类型
data_transform_field_no_use_norm_mask="--data.transform.field.no-use-norm-mask"  # 不使用归一化mask
data_transform_field_normalize_state="--data.transform.field.normalize-state"  # 归一化state
data_transform_field_pad_action_dim="--data.transform.field.pad-action-dim=36"  # action padding维度
data_transform_field_pad_state_dim="--data.transform.field.pad-state-dim=36"  # state padding维度
data_transform_model_img_aug="--data.transform.model.img-aug"  # 启用图像增强
data_transform_model_resize_size="--data.transform.model.resize.size 240 320"  # 图像resize尺寸
data_transform_model_center_crop_size="--data.transform.model.center_crop.size 240 320"  # 中心裁剪尺寸

# ============ model ============
model_model_name_or_path="--model.model_name_or_path=$PSI_HOME/model/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"  
model_pretrained_action_header_path="--model.pretrained-action-header-path=$PSI_HOME/model/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"  
model_noise_scheduler="--model.noise-scheduler=flow"  # 噪声调度器类型
model_train_diffusion_steps="--model.train-diffusion-steps=1000"  # 扩散步数
model_n_conditions="--model.n_conditions=0"  # 条件数量
model_action_chunk_size="--model.action-chunk-size=30"  # ⚠️action分块大小
model_action_dim="--model.action-dim=36"  # action维度
model_action_exec_horizon="--model.action-exec-horizon=30"  # action执行视野
model_observation_horizon="--model.observation-horizon=1"  # 观测视野,未被使用 
model_odim="--model.odim=36"  # 观测维度
model_view_feature_dim="--model.view-feature-dim=2048"  
model_no_tune_vlm="--model.no-tune-vlm"  # 不微调VLM
model_no_use_film="--model.no-use_film"  # 不使用FiLM
model_no_combined_temb="--model.no-combined_temb"  # 不合并时间嵌入
model_rtc="--model.rtc"  # 启用RTC
model_max_delay="--model.max-delay=8"  # 最大延迟

# ============ launch ============
torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${config} \
    ${seed} \
    ${exp} \
    ${train_name} \
    ${train_data_parallel} \
    ${train_mixed_precision} \
    ${train_train_batch_size} \
    ${train_max_checkpoints_to_keep} \
    ${train_gradient_accumulation_steps} \
    ${train_learning_rate} \
    ${train_max_training_steps} \
    ${train_warmup_ratio} \
    ${train_warmup_steps} \
    ${train_checkpointing_steps} \
    ${train_validation_steps} \
    ${train_val_num_batches} \
    ${train_max_grad_norm} \
    ${train_lr_scheduler_type} \
    ${train_lr_scheduler_kwargs_weight_decay} \
    ${train_lr_scheduler_kwargs_betas} \
    ${log_report_to} \
    ${data_root_dir} \
    ${data_train_repo_ids} \
    ${data_transform_repack_pad_action_dim} \
    ${data_transform_repack_pad_state_dim} \
    ${data_transform_field_stat_path} \
    ${data_transform_field_stat_action_key} \
    ${data_transform_field_stat_state_key} \
    ${data_transform_field_action_norm_type} \
    ${data_transform_field_no_use_norm_mask} \
    ${data_transform_field_normalize_state} \
    ${data_transform_field_pad_action_dim} \
    ${data_transform_field_pad_state_dim} \
    ${data_transform_model_img_aug} \
    ${data_transform_model_resize_size} \
    ${data_transform_model_center_crop_size} \
    ${model_model_name_or_path} \
    ${model_pretrained_action_header_path} \
    ${model_noise_scheduler} \
    ${model_train_diffusion_steps} \
    ${model_n_conditions} \
    ${model_action_chunk_size} \
    ${model_action_dim} \
    ${model_action_exec_horizon} \
    ${model_observation_horizon} \
    ${model_odim} \
    ${model_view_feature_dim} \
    ${model_no_tune_vlm} \
    ${model_no_use_film} \
    ${model_no_combined_temb} \
    ${model_rtc} \
    ${model_max_delay}

