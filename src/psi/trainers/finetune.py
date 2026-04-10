from __future__ import annotations
import os
import json
import torch
import wandb
import pickle
import numpy as np
from tqdm import tqdm
from copy import copy, deepcopy
import contextlib
import torch.nn.functional as F
from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING
from torch.utils.data import DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLProcessor
from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether

if TYPE_CHECKING:    
    from psi.config.config import TrainConfig
    from psi.config.model_psi0 import Psi0ModelConfig
    from psi.config.transform import Psi0ModelTransform

from accelerate import Accelerator

# from utils.overwatch import initialize_overwatch
from psi.utils import (
    initialize_overwatch,
    shorten
)
from psi.utils.utils import batch_str_to_tensor

overwatch = initialize_overwatch(__name__)

# from .base import Trainer
from .trainer import Trainer, worker_init_fn

from psi.utils import flatten, shorten, move_to_device, rmse, seed_everything
from psi.models.psi0 import Psi0Model

class FinetuneTrainer(Trainer):

    def __init__(self, cfg, device: torch.device):
        # ------------------------------------------------------------------
        # 步骤 1: 调用父类初始化，获取基础配置
        # ------------------------------------------------------------------
        super().__init__(cfg, device)

        # ------------------------------------------------------------------
        # 步骤 2: 从模型配置中读取 diffusion 相关参数
        # ------------------------------------------------------------------
        self.noise_scheduler_name = self.model_cfg.noise_scheduler
        self.train_diffusion_steps = self.model_cfg.train_diffusion_steps
        self.eval_diffusion_steps = self.model_cfg.eval_diffusion_steps
        self.ac_chunk = self.model_cfg.action_chunk_size
        self.ac_dim = self.model_cfg.action_dim
        self.maxmin = self.data_cfg.transform.field

        # ------------------------------------------------------------------
        # 步骤 3: 初始化 action 各维度的 loss 权重 (xyz, rpy, gripper)
        # ------------------------------------------------------------------
        if self.model_cfg.action_dim == 7:
            w_xyz, w_rpy, w_gripper = self.model_cfg.loss_w
            self.loss_w = (
                torch.from_numpy(np.array([w_xyz] * 3 + [w_rpy] * 3 + [w_gripper]))
                .float()
                .to(self.device)
            )  # (7,)
        else:
            self.loss_w = torch.tensor([1.0 / self.model_cfg.action_dim] * self.model_cfg.action_dim, dtype=torch.float32).to(self.device)
        assert self.loss_w.sum() == 1.0, "Weights better sum to 1.0 to keep loss range consistent"

        # ------------------------------------------------------------------
        # 步骤 4: 初始化 noise scheduler (ddpm / flow matching)
        # ------------------------------------------------------------------
        if self.noise_scheduler_name == "ddpm":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            self.noise_scheduler = DDIMScheduler(  # FIXME configure args
                num_train_timesteps=self.train_diffusion_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
        elif self.noise_scheduler_name == "flow":
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler,
            )
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.train_diffusion_steps,  # MUST be 1000 as per pretrained SD3
            )
        # assert self.task_cfg.mixed_precision == "no", "other options not tested"
        # assert self.model_cfg.n_conditions == len(self.data_cfg.transform.repack.conditions), "inconsistent confs" # type: ignore

    @property
    def task_cfg(self) -> TrainConfig:
        return self.cfg.train

    @property
    def model_cfg(self) -> Psi0ModelConfig:
        return self.cfg.model # type: ignore
    
    # ========================================================================
    # 初始化 VLM (Vision-Language Model): Qwen3-VL-2B-Instruct
    # ========================================================================
    def init_qwen3vl_models(self):
        # 使用 flash_attention_2 加载预训练 VLM
        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_cfg.model_name_or_path,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16
        )
        overwatch.info(f"Load pretrained VLM model from {self.model_cfg.model_name_or_path}")

        # 加载对应的 processor (包含 tokenizer 和 image processor)
        self.vlm_processor = AutoProcessor.from_pretrained(self.model_cfg.model_name_or_path)
        self.tokenizer = self.vlm_processor.tokenizer

        # 确保不使用 LoRA (当前不支持)
        assert self.cfg.train.lora == False

        return vlm_model

    # ========================================================================
    # 初始化完整模型: Psi0Model (VLM + Action Header)
    # ========================================================================
    def init_models(self):
        # ------------------------------------------------------------------
        # 步骤 1: 初始化 VLM (Qwen3-VL-2B-Instruct)
        # ------------------------------------------------------------------
        vlm_model = self.init_qwen3vl_models()

        # ------------------------------------------------------------------
        # 步骤 2: 构建 Psi0Model，组合 VLM 和 Action Header
        # ------------------------------------------------------------------
        self.model = Psi0Model(
            model_cfg=self.model_cfg,
            vlm_model=vlm_model,
        )

        # ------------------------------------------------------------------
        # 步骤 3: 加载预训练的 Action Header (可选)
        # ------------------------------------------------------------------
        if self.model_cfg.pretrained_action_header_path is not None:
            from safetensors.torch import load_file
            ckpt_path = self.model_cfg.pretrained_action_header_path
            state_dict = load_file(f"{ckpt_path}/action_header.safetensors")
            if state_dict["action_proj_in.dec_pos"].shape[0] != self.model_cfg.action_chunk_size:
                # 若 action chunk size 不匹配，仅加载 transformer blocks
                reduced_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("transformer_blocks"):
                        reduced_state_dict[k] = v
                overwatch.info(f"Loading pretrained action header from {ckpt_path}")
                self.model.action_header.load_state_dict(reduced_state_dict, strict=False)
                overwatch.warning("action_proj_in.dec_pos size mismatch, only loaded transformer blocks.")
            else:
                self.model.action_header.load_state_dict(state_dict, strict=False)
            overwatch.info("loaded pretrained action header successfully.")
        else:
            overwatch.info("No pretrained action header specified, training from scratch.")

        # ------------------------------------------------------------------
        # 步骤 4: 配置 DeepSpeed (若启用)
        # ------------------------------------------------------------------
        if self.train_cfg.data_parallel == "deepspeed":
            # HACK: 设置正确的 config 以便 DeepSpeed 初始化
            self.model.config = vlm_model.config
            self.model.config.hidden_size = vlm_model.config.text_config.hidden_size

        # ------------------------------------------------------------------
        # 步骤 5: 根据 tune_vlm 标志设置 VLM 参数是否可训练
        # ------------------------------------------------------------------
        if not self.model_cfg.tune_vlm:
            for p in self.model.vlm_model.parameters():
                p.requires_grad = False
            overwatch.info("VLM parameters are frozen (tune_vlm=False)")
        else:
            for p in self.model.vlm_model.parameters():
                p.requires_grad = True
            # 冻结最后一层 norm (因为使用 hidden_states[-1] 不需要它)
            if hasattr(self.model.vlm_model.model, 'language_model') and hasattr(self.model.vlm_model.model.language_model, 'norm'):
                for p in self.model.vlm_model.model.language_model.norm.parameters():
                    p.requires_grad = False
                overwatch.info("VLM final norm layer frozen (not used in forward pass)")
            overwatch.info("VLM parameters are trainable (tune_vlm=True)")

        # ------------------------------------------------------------------
        # 步骤 6: 打印可训练参数数量
        # ------------------------------------------------------------------
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        overwatch.info(f"Model has {num_parameters:,} trainable parameters")


    # ========================================================================
    # 创建优化器，按参数组设置不同的学习率
    # ========================================================================
    def create_optimizers(self):
        # ------------------------------------------------------------------
        # 步骤 1: 将模型参数按模块分组 (action_header / vlm_model / other)
        # ------------------------------------------------------------------
        action_header_params = []
        vlm_model_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if name.startswith("action_header."):
                action_header_params.append(param)
            elif name.startswith("vlm_model."):
                if self.model_cfg.tune_vlm:
                    vlm_model_params.append(param)
            else:
                other_params.append(param)

        # ------------------------------------------------------------------
        # 步骤 2: 构建参数组，指定各自的学习率
        # ------------------------------------------------------------------
        param_groups = []
        if action_header_params:
            param_groups.append({
                'params': action_header_params,
                'lr': self.cfg.train.learning_rate,
                'group_name': 'action_header'
            })
        if vlm_model_params:
            param_groups.append({
                'params': vlm_model_params,
                'lr': self.model_cfg.lang_backbone_lr,
                'group_name': 'vlm_model'
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.cfg.train.learning_rate,  # Default group
                'group_name': 'other'
            })

        # ------------------------------------------------------------------
        # 步骤 3: 创建 AdamW 优化器
        # ------------------------------------------------------------------
        optimizer_kwargs = dict(self.cfg.train.lr_scheduler_kwargs)
        if self.cfg.train.optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = self.cfg.train.optimizer_foreach
        self.optimizer = torch.optim.AdamW(
            param_groups,
            **optimizer_kwargs, # type: ignore
        )

    # ========================================================================
    # 创建学习率调度器 (LR Scheduler)
    # ========================================================================
    def create_lr_schedulers(
        self,
        num_training_steps: int | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        from transformers.optimization import get_scheduler
        assert num_training_steps is not None
        self.lr_scheduler = get_scheduler(
            self.cfg.train.lr_scheduler_type,
            num_training_steps=num_training_steps * self.world_size,
            optimizer=optimizer if optimizer is not None else self.optimizer,
            num_warmup_steps=self.num_warmup_steps * self.world_size,
            scheduler_specific_kwargs=self.cfg.train.scheduler_specific_kwargs,
        )

    # ========================================================================
    # 组合创建优化器和 LR 调度器
    # ========================================================================
    def create_optimizer_and_scheduler(self, num_training_steps: int | None = None):
        optimizer = self.create_optimizers()
        self.create_lr_schedulers(
            num_training_steps=num_training_steps, optimizer=optimizer
        )

    # ========================================================================
    # 创建训练和验证数据集
    # ========================================================================
    def create_datasets(self):  # TODO use parent impl.
        # 传入 vlm_processor 用于图像预处理
        transform_kwargs=dict(
            vlm_processor=self.vlm_processor,
        )
        self.train_dataset = self.data_cfg(split="train", transform_kwargs=transform_kwargs)
        self.val_dataset = self.data_cfg(split="val", transform_kwargs=transform_kwargs)
        return self.train_dataset, self.val_dataset

    # ========================================================================
    # 创建训练和验证 DataLoader (带 padding collator)
    # ========================================================================
    def create_dataloaders(self, train_dataset, val_dataset):
        # ------------------------------------------------------------------
        # 步骤 1: 配置 DataLoader 公共参数 (num_workers, shuffle, drop_last 等)
        # ------------------------------------------------------------------
        g = torch.Generator()
        g.manual_seed(self.cfg.seed or 42)
        train_dataloader_kwargs = {
            "num_workers": 12,
            "drop_last": True,
            "shuffle": True,
            "generator": g,
            "worker_init_fn": worker_init_fn,
            "persistent_workers": True,  # prefetch_factor=4
        }

        val_dataloader_kwargs = {
            "num_workers": 12,
            "drop_last": False,
            # "pin_memory": True,
            "persistent_workers": True,
            "shuffle": True,
        }  # 1 small enough to fit im Mem. 2. no need distributed sampler

        # ------------------------------------------------------------------
        # 步骤 2: 创建 Collator (对 batch 内的变长序列进行 padding)
        # ------------------------------------------------------------------
        collator = PaddedCollatorForTogether(
            model_max_length=self.tokenizer.model_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="right",
        )

        # ------------------------------------------------------------------
        # 步骤 3: 构建训练和验证 DataLoader
        # ------------------------------------------------------------------
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.task_cfg.train_batch_size,
            collate_fn=collator,
            **train_dataloader_kwargs,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.task_cfg.val_batch_size,
            collate_fn=collator,
            **val_dataloader_kwargs,
        )
        return self.train_dataloader, self.val_dataloader

    @property
    def task_run_name(self):
        dataset_name = shorten(self.data_cfg.transform.repack.dataset_name) # type:ignore
        return (
            f".{dataset_name}"
            f".{self.noise_scheduler_name}{self.train_diffusion_steps}"
            f".{shorten(self.task_cfg.lr_scheduler_type)}"
            f".lr{self.task_cfg.learning_rate:.1e}"
        )

    # ========================================================================
    # 使用 Accelerator 准备模型、优化器、调度器和 DataLoader
    # ========================================================================
    def prepare(self, accelerator: Accelerator) -> DataLoader:
        # ------------------------------------------------------------------
        # 步骤 1: 将 model, optimizer, lr_scheduler, train_dataloader 注册到 accelerator
        # NOTE: 使用 DeepSpeed 时，model/optimizer/lr_scheduler 必须一起 prepare
        # ------------------------------------------------------------------
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader
        )

        # ------------------------------------------------------------------
        # 步骤 2: 配置 DeepSpeed Gradient Checkpointing (若启用)
        # ------------------------------------------------------------------
        if self.train_cfg.data_parallel == "deepspeed":
            if self.model_cfg.gradient_checkpointing and self.model_cfg.tune_vlm:
                assert "DeepSpeedEngine" in self.model.__class__.__name__, "deepspeed is not properly initialized!"
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.vlm_model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    self.model.vlm_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # ------------------------------------------------------------------
        # 步骤 3: 单 batch overfitting 模式 (用于 debug)
        # ------------------------------------------------------------------
        if self.cfg.train.overfit_single_batch:
            overwatch.warning("Overfitting a single batch: reusing first batch every step. set cfg.data.image_aug = False for true memorization.")
            first_batch = next(iter(self.train_dataloader))
            class SingleBatchLoader:
                def __iter__(self):
                    while True:
                        yield first_batch
                def __len__(self):
                    return 1
            self.train_dataloader = SingleBatchLoader()

        # ------------------------------------------------------------------
        # 步骤 4: 准备验证 DataLoader (若存在)
        # ------------------------------------------------------------------
        val_dataloader = getattr(self, "val_dataloader", None)
        if val_dataloader is not None: # not using if self.val_dataloader to avoid DataLoader.__len__() being called on iterable dataset
            self.val_dataloader = accelerator.prepare(self.val_dataloader)

        self.accelerator = accelerator
        return self.train_dataloader # type: ignore

    # ========================================================================
    # 单步训练: 前向传播 → 反向传播 → 参数更新 → 日志记录
    # ========================================================================
    def training_step(
        self,
        batch: dict[str, Union[torch.Tensor, Any]],
    ) -> tuple[bool, dict[str, Any]]:

        # ------------------------------------------------------------------
        # 步骤 1: 梯度累积上下文 (DeepSpeed 自己管理梯度累积)
        # ------------------------------------------------------------------
        if self.train_cfg.data_parallel == "deepspeed":
            gac = contextlib.nullcontext()
        else:
            # 否则由 accelerator 管理梯度累积
            gac = self.accelerator.accumulate(self.model)
        with gac:
            # ------------------------------------------------------------------
            # 步骤 2: 前向传播计算 loss (在 autocast 上下文中使用 bfloat16)
            # ------------------------------------------------------------------
            with self.accelerator.autocast():
                losses = self.forward_and_loss(self.model, batch)

            # ------------------------------------------------------------------
            # 步骤 3: 反向传播计算梯度
            # ------------------------------------------------------------------
            self.accelerator.backward(losses["loss"])

            # ------------------------------------------------------------------
            # 步骤 4: 梯度同步后执行裁剪 (若启用) 和参数更新
            # ------------------------------------------------------------------
            if self.accelerator.sync_gradients:
                # 梯度裁剪，防止梯度爆炸
                if self.train_cfg.max_grad_norm is not None:
                    self._grad_norm_act = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.max_grad_norm
                    )
                else:
                    self._grad_norm_act = self.get_total_grad_norm(self.model.parameters())

                # DDP 模式下检查参数是否获得梯度 (Deepspeed/FSDP 正常为 None)
                if self.train_cfg.data_parallel == "ddp":
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is None:
                            overwatch.critical(f"[Unused] {name} did not receive a gradient.")

            # ------------------------------------------------------------------
            # 步骤 5: 优化器步进 + LR 调度器步进 + 梯度清零
            # ------------------------------------------------------------------
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # ------------------------------------------------------------------
        # 步骤 6: 可视化日志 (按 log_freq 频率记录 raw_images 到 wandb)
        # ------------------------------------------------------------------
        if (
            hasattr(self, "global_step")
            and self.global_step % self.cfg.log.log_freq == 0
            and "raw_images" in batch
            and self.accelerator.is_main_process
            and self.cfg.log.report_to == "wandb"
        ):
            raw_imgs = batch["raw_images"]
            # Support both torch.Tensor and numpy
            if isinstance(raw_imgs, torch.Tensor):
                raw_imgs = raw_imgs.detach().cpu().numpy()
            # Log up to 4 images for visualization: concat them horizontally into one image
            img_arrays = []
            for i in range(min(10, len(raw_imgs))):
                img = raw_imgs[i][0]  # t=0
                img_arrays.append(img)

            # Images are assumed to have same shape and 3 channels; concat directly
            concat_img = np.concatenate(img_arrays, axis=1)
            wandb.log({"raw_images": [wandb.Image(concat_img, caption=f"raw images {self.global_step}")]}, step=self.global_step)

        # ------------------------------------------------------------------
        # 步骤 7: 记录 step loss 并返回同步标志和 metrics
        # ------------------------------------------------------------------
        self.train_loss_tracker = step_loss = losses["loss"].detach().item()

        return (self.accelerator.sync_gradients, {
            "lr_act": self.lr,
            "grad_norm_act": self._grad_norm_act, # type: ignore
            "loss": step_loss
        })

    @torch.no_grad()
    def inference(self, eval_model, repacked_batch):

        traj2ds = repacked_batch["traj2ds"] if "traj2ds" in repacked_batch else None  # (B, 3, H, W)
        obs = repacked_batch["states"]  # (B,1,M)

        # imgs_in = {"cam0": view_features}  # views # TODO support multi-view
        bsz = obs.shape[0]

        action_samples = torch.randn(
            bsz, self.ac_chunk, self.ac_dim, device=self.device
        )
        self.noise_scheduler.set_timesteps(self.eval_diffusion_steps)

        for timestep in self.noise_scheduler.timesteps:
            batched_timestep = timestep.expand(bsz).to(self.device)

            model_pred = eval_model(
                input_ids=repacked_batch["input_ids"],#####
                attention_mask=repacked_batch["attention_mask"], # vlm related
                pixel_values=repacked_batch["pixel_values"],
                image_grid_thw=repacked_batch["image_grid_thw"], ####
                action_samples=action_samples, # (B,Tp,Da)
                states=obs,
                timestep=batched_timestep,
                traj2ds=traj2ds,
                return_dict=True,
            ).action

            action_samples = self.noise_scheduler.step(
                model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
            ).prev_sample

        return action_samples
    
    def save_checkpoint(self, global_step: int) -> str | None:
        save_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_dir = os.path.join(save_dir, f"ckpt_{global_step}")
        
        self.accelerator.save_model(self.model, ckpt_dir)
        return super().save_checkpoint(global_step)

    def evaluate(self) -> dict[str, float] | None:
        accelerator = self.accelerator
        global_step = self.global_step
        eval_model = self.unwrap_model()

        total_val_batches = (
            len(self.val_dataloader)
            if self.task_cfg.val_num_batches == -1
            else min(self.task_cfg.val_num_batches, len(self.val_dataloader))
        )
        val_progress_bar = tqdm(
            self.val_dataloader,
            total=total_val_batches,
            disable=not accelerator.is_local_main_process,
            position=1,
            leave=False,
        )
        val_progress_bar.set_description(f"Eval at global step {global_step}")

        val_loss_list = []
        action_l1_err_list = []

        for val_step, val_batch in enumerate(val_progress_bar):
            val_batch = batch_str_to_tensor(val_batch)
            # mask = val_batch["mask"]
            mask = torch.ones_like(val_batch["actions"]) # FIXME
            gt_actions = val_batch["actions"]  # (B, Tp, Da)
            # gt_actions = self.data_cfg.data_transforms.normalize_action(repacked_batch[1])

            # Tp -> predicted action horizon, Da -> action dim
            B, Tp, Da = gt_actions.shape

            # validation loss
            with accelerator.autocast():
                val_loss = self.forward_and_loss(eval_model, val_batch)
                val_loss_list.append(accelerator.gather(val_loss["loss"]))

                # action prediction loss
                pred_actions = self.inference(eval_model, val_batch)
                err_action_l1 = pred_actions - gt_actions  # (B, Tp, Da)

                err_action_l1_all = accelerator.gather(
                    err_action_l1[:, :Tp].contiguous()
                )
                err_action_masks_all = accelerator.gather(mask[:, :Tp].contiguous())
                err_action_l1 = err_action_l1_all[
                    err_action_masks_all.to(torch.bool) # type: ignore
                ].abs()
                action_l1_err_list.append(err_action_l1.reshape(-1, Da).float().cpu().numpy())  # (B*world_size*Ta, 7)

            if val_step + 1 >= total_val_batches:
                if accelerator.is_local_main_process:
                    val_progress_bar.close()
                self.val_dataloader.end() # type: ignore
                break

        avg_val_loss = torch.cat(val_loss_list).mean().item()
        action_l1_err_list = np.concatenate(action_l1_err_list, axis=0)  # (len_val_dataset*Ta, Da)
        action_l1_err_list_denormed = (
            self.maxmin.denormalize_L1_action_err( # type: ignore FIXME
                action_l1_err_list
            )
        )

        # action L1 errors
        avg_action_errors_denormed = action_l1_err_list_denormed.mean(0)  # (Da,) NOTE only if the error is L1 (linear)
        # Define dimension splits: hand_joints(14) + arm_joints(14) + rpy(3) + height(1) = 32
        hand_joints_start, hand_joints_end = 0, 14
        arm_joints_start, arm_joints_end = 14, 28
        rpy_start, rpy_end = 28, 31
        height_start, height_end = 31, 32
        torso_vx_start, torso_vx_end = 32, 33
        torso_vy_start, torso_vy_end = 33, 34
        torso_vyaw_start, torso_vyaw_end = 34, 35
        torso_dyaw_start, torso_dyaw_end = 35, 36
    
        labels_denormed = [
            "err_l1_hand_joints",
            "err_l1_arm_joints",
            "err_l1_torso_rpy",
            "err_l1_height",
            "err_l1_vx",
            "err_l1_vy",
            "err_l1_vyaw",
            "err_l1_target_yaw",
        ]
    
        avg_lr_action_err_denormed = np.split(
            avg_action_errors_denormed, [hand_joints_end, arm_joints_end, rpy_end, height_end, torso_vx_end, torso_vy_end, torso_vyaw_end], axis=-1
        )

        # log metrics
        return {
            "loss": avg_val_loss,
            **dict(zip(labels_denormed, map(np.linalg.norm, avg_lr_action_err_denormed)))
        }

    # ========================================================================
    # 前向传播 + Loss 计算: 加噪 → 模型预测 → MSE Loss
    # ========================================================================
    def forward_and_loss(self, model, batch) -> dict[str, torch.Tensor]:
        bsz, Tp, Da = batch["actions"].shape

        # ------------------------------------------------------------------
        # 步骤 1: 采样 timesteps (noise level)
        # ------------------------------------------------------------------
        if self.noise_scheduler_name == "ddpm":
            # DDPM: 均匀采样 [0, train_diffusion_steps) 的整数 timesteps
            timesteps = torch.randint(
                low=0, high=self.train_diffusion_steps, size=(bsz,), device=self.device
            ).long()

        elif self.noise_scheduler_name == "flow":
            # 主要路径: Flow matching: 采样 [0, 1) 的 sigma，然后缩放到扩散步数
            sigmas = torch.rand((bsz,), device=self.device)
            timesteps = sigmas * self.train_diffusion_steps

            # RTC 模式: 生成随机 delay，创建 prefix_mask 标记被遮蔽的 action
            if self.model_cfg.rtc:
                delay = torch.randint(
                    low=0,
                    high=self.model_cfg.max_delay,
                    size=(bsz,),
                    device=self.device,
                    dtype=torch.long
                )
                # prefix_mask[i][t] = True 表示 action[t] 被遮蔽 (delay 范围内)
                prefix_mask = torch.arange(Tp, device=self.device)[None, :] < delay[:, None]
                # 被遮蔽部分的 sigma 强制为 0 (即 clean action)
                sigmas = torch.where(
                    prefix_mask,
                    torch.tensor(0.0, device=self.device, dtype=sigmas.dtype),
                    sigmas[:, None]
                )
                timesteps = sigmas * self.train_diffusion_steps

        else:
            raise ValueError(f"Unknown noise scheduler: {self.noise_scheduler_name}")

        # ------------------------------------------------------------------
        # 步骤 2: 对 actions 加噪，生成 noisy_actions
        # 同时确定 target_action (模型需要预测的目标)
        # ------------------------------------------------------------------
        noise_action = torch.randn_like(batch["actions"])
        if self.noise_scheduler_name == "ddpm":
            # DDPM: 直接用 scheduler 添加噪声
            noisy_actions = self.noise_scheduler.add_noise(
                batch["actions"], noise_action, timesteps # type: ignore
            )
            target_action = noise_action  # DDPM 预测噪声 epsilon
        elif self.noise_scheduler_name == "flow":
            # Flow matching: 线性插值 (1-sigma)*clean + sigma*noise
            sigmas_action = sigmas  # type: ignore
            while len(sigmas_action.shape) < len(batch["actions"].shape):
                sigmas_action = sigmas_action.unsqueeze(-1)  # Expand to B, Tp, action_dim
            noisy_actions = (1 - sigmas_action) * batch["actions"] + sigmas_action * noise_action
            target_action = noise_action - batch["actions"]  # Flow matching 预测速度 (noise - clean)

        # ------------------------------------------------------------------
        # 步骤 3: 模型前向传播 (VLM 提取视觉-语言特征 + Action Header 预测 action)
        # ------------------------------------------------------------------
        model_output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],  # VLM 相关
            pixel_values=batch["pixel_values"], # 视觉embedding
            image_grid_thw=batch["image_grid_thw"],  # patch shape
            action_samples=noisy_actions,  # 加噪后的 action 作为输入
            states=batch["states"],  # (B, 1, M) 机器人状态
            timestep=timesteps,  # 时间步embeding
            traj2ds=batch["traj2ds"] if "traj2ds" in batch else None,  # (B, C, 3, H, W) 2D 轨迹图
            return_dict=True,
        )
        action_pred = model_output.action

        # ------------------------------------------------------------------
        # 步骤 4: 计算 MSE Loss，并应用 mask 和维度权重
        # ------------------------------------------------------------------
        loss_action = F.mse_loss(
            action_pred.float(), target_action.float(), reduction="none"
        )  # (B, Tp, Da)

        # mask: 标记哪些 action dimension 是有效的 (由数据集提供)
        mask = batch["actions_mask"].float() if "actions_mask" in batch else torch.ones_like(batch["actions"])
        # RTC 模式: postfix_mask 标记非遮蔽部分 (只有这部分计算 loss)
        if self.model_cfg.rtc:
            postfix_mask = (~prefix_mask)[:, :, None].float()  # (B, Tp, 1)
            mask = mask * postfix_mask

        # sum(1): 跨时间步求和 (保持不同 action chunk 训练时梯度一致)
        # mean(0): 跨 batch 求平均
        # sum(): 跨 action 维度加权求和 (xyz, rpy, gripper 各有权重)
        loss_action = (loss_action * mask).sum(1)  # (B, Da)
        loss_action = (loss_action.mean(0) * self.loss_w).sum()

        return {"loss": loss_action}
