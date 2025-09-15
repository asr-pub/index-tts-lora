import copy
import gc
import os
import random
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from indextts.BigVGAN.models import BigVGAN
from indextts.data_utils import (
    collate_finetune_fn,
    load_finetune_datasets,
    load_speaker_conditions,
)
from indextts.gpt.model import UnifiedVoice


def load_UnifiedVoice(gpt_config: DictConfig, gpt_checkpoint_path: str, device: torch.device) -> UnifiedVoice:
    """加载 UnifiedVoice 模型权重。"""
    state_dict = torch.load(gpt_checkpoint_path, map_location=device, weights_only=True)
    state_dict = state_dict["model"] if "model" in state_dict else state_dict
    model = UnifiedVoice(**gpt_config)
    model.load_state_dict(state_dict, strict=True)
    model.post_init_gpt2_config()
    del state_dict
    return model.to(device)

def clear_torch_cache():
    """清理 GPU 缓存。"""
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        # logger.info(f"{torch.cuda.memory_reserved() / (1024**2):.2f} MiB reserved")
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # logger.info("CUDA cache cleared.")
        # logger.info(f"{torch.cuda.memory_reserved() / (1024**2):.2f} MiB reserved")

def forward_gpt2(
    model: UnifiedVoice,
    inputs_embeds: torch.FloatTensor,
    text_lengths: torch.LongTensor,
    codes_lengths: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_latent: bool = False,
    output_logits: bool = True,
):
    assert attention_mask is not None, "Attention mask must be provided for UnifiedVoice forward pass."

    """UnifiedVoice GPT2 部分的前向传播。"""
    b = inputs_embeds.shape[0]
    gpt_out = model.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
    hidden_state = gpt_out.last_hidden_state

    # Vectorized implementation to replace the for loop
    conditioning_len = 32
    
    # Remove conditioning part from hidden states and attention mask
    h_no_cond = hidden_state[:, conditioning_len:]  # [b, seq_len, hidden_dim]
    attention_no_cond = attention_mask[:, conditioning_len:]  # [b, seq_len]
    
    # Apply final_norm to all samples at once
    latent = model.final_norm(h_no_cond)  # [b, seq_len, hidden_dim]
    
    # Get max lengths for efficient processing
    max_text_len = text_lengths.max().item()
    max_mel_len = codes_lengths.max().item()
    
    # Create batched tensors for text and mel latents
    batch_text_latents = torch.zeros(b, max_text_len, latent.shape[-1], device=latent.device, dtype=latent.dtype)
    batch_mel_latents = torch.zeros(b, max_mel_len, latent.shape[-1], device=latent.device, dtype=latent.dtype)
    
    ## Create masks for valid positions in batched tensors
    #text_valid_mask = torch.zeros(b, max_text_len, device=latent.device, dtype=torch.int32)
    #mel_valid_mask = torch.zeros(b, max_mel_len, device=latent.device, dtype=torch.int32)
    
    # Fill the batched tensors
    for i in range(b):
        text_len = text_lengths[i].item()
        mel_len = codes_lengths[i].item()
        
        # Extract valid latent for this sample
        sample_valid_mask = attention_no_cond[i] == 1
        sample_latent = latent[i][sample_valid_mask]  # [valid_len, hidden_dim]
        
        # Verify the expected length
        expected_len = text_len + mel_len
        assert sample_latent.shape[0] == expected_len, \
            f"Expected valid_latent shape {expected_len}, got {sample_latent.shape[0]}, " \
            f"text_len: {text_len}, mel_len: {mel_len}"
        
        # Split and assign to batched tensors
        batch_text_latents[i, :text_len] = sample_latent[:text_len]
        batch_mel_latents[i, :mel_len] = sample_latent[text_len:text_len + mel_len]
        
        ## Set valid masks
        #text_valid_mask[i, :text_len] = 1
        #mel_valid_mask[i, :mel_len] = 1
    
    # Vectorized head processing
    # Process all text latents at once
    batch_text_logits = model.text_head(batch_text_latents)  # [b, max_text_len, vocab_size]
    batch_text_logits = batch_text_logits.permute(0, 2, 1)  # [b, vocab_size, max_text_len]
    
    # Process all mel latents at once  
    batch_mel_logits = model.mel_head(batch_mel_latents)  # [b, max_mel_len, vocab_size]
    batch_mel_logits = batch_mel_logits.permute(0, 2, 1)  # [b, vocab_size, max_mel_len]
    
    # Return the processed batches directly.
    # The tensors are already padded to the max length in the batch.
    output = {}
    if output_logits:
        output["logits"] = (batch_text_logits, batch_mel_logits)
    if output_latent:
        output["latent"] = (batch_text_latents, batch_mel_latents)
    return output

def forward_UnifiedVoice(
    model: UnifiedVoice,
    mel_spec: torch.FloatTensor,
    mel_codes: torch.LongTensor,
    text_ids: torch.LongTensor,
    mel_lengths: torch.LongTensor,
    codes_lengths: torch.LongTensor,
    text_lengths: torch.LongTensor,
    speaker_ids: List[str] = None,  # 新增参数
    add_mel_stop_token: bool = True,
    output_loss: bool = True,
    output_logits: bool = True,
    output_latent: bool = False,
    loss_reduction: str = "mean",
):
    """UnifiedVoice 模型的完整前向传播。"""

    conditioning_latent = model.get_conditioning(mel_spec, mel_lengths, speaker_ids=speaker_ids)  # 传递 speaker_ids
    
    # -------- build text_inputs with start/stop tokens --------
    B, T_pad = text_ids.shape
    max_out_text = T_pad + 2  # +<start> +<stop>
    text_inputs = text_ids.new_zeros((B, max_out_text))
    for i, L in enumerate(text_lengths):
        L = L.item()
        text_inputs[i, 0] = model.start_text_token
        text_inputs[i, 1 : L + 1] = text_ids[i, :L]
        text_inputs[i, L + 1] = model.stop_text_token
    text_targets = text_inputs[:, 1:].clone().contiguous()

    # -------- build mel_inputs with start/stop tokens --------
    B, M_pad = mel_codes.shape
    extra_stop = 1 if add_mel_stop_token else 0
    max_out_mel = M_pad + 1 + extra_stop  # +<start> (+<stop>)
    mel_inputs = mel_codes.new_zeros((B, max_out_mel))
    for i, L in enumerate(codes_lengths):
        L = L.item()
        mel_inputs[i, 0] = model.start_mel_token
        mel_inputs[i, 1 : L + 1] = mel_codes[i, :L]
        if add_mel_stop_token:
            mel_inputs[i, L + 1] = model.stop_mel_token
    mel_targets = mel_inputs[:, 1:].clone().contiguous()

    # Embeddings
    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    # for later use in loss and lengths
    mel_codes = mel_inputs
    
    inputs_embeds = torch.cat([conditioning_latent, text_emb, mel_emb], dim=1)
    
    # Create attention mask for the combined sequence
    batch_size, total_seq_len = inputs_embeds.shape[:2]
    attention_mask = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device=inputs_embeds.device)
    
    # Calculate actual sequence lengths for each sample
    conditioning_len = conditioning_latent.shape[1]
    actual_text_lengths = text_lengths + 2  # +2 for start/stop tokens
    actual_mel_lengths = codes_lengths + 1 + int(add_mel_stop_token)  # +1 for start token + optional stop token
    
    for i in range(batch_size):
        # Set conditioning part (always valid)
        attention_mask[i, :conditioning_len] = 1
        
        # Set text part (considering actual text length without padding)
        text_start = conditioning_len
        text_end = text_start + actual_text_lengths[i].item()
        attention_mask[i, text_start:text_end] = 1
        
        # Set mel part (considering actual mel length without padding)
        # mel_start should be based on the actual text_emb length, not text_end
        mel_start = conditioning_len + text_emb.shape[1]
        mel_end = mel_start + actual_mel_lengths[i].item()
        attention_mask[i, mel_start:mel_end] = 1

    gpt2_outputs = forward_gpt2(
        model,
        inputs_embeds,
        text_lengths + 2,
        codes_lengths + 1 + int(add_mel_stop_token),
        attention_mask=attention_mask,
        output_latent=output_latent,
        output_logits=output_logits or output_loss,
    )
    
    outputs = {}
    if output_logits or output_loss:
        text_logits, mel_logits = gpt2_outputs["logits"]
        text_logits = text_logits[:, :, :-1].contiguous()
        mel_logits = mel_logits[:, :, :-1].contiguous()
        if output_loss:
            # Create masks based on actual sequence lengths
            batch_size = text_targets.size(0)
            
            # Text mask: consider actual text length + 1 (stop token)
            text_mask = torch.zeros_like(text_targets, dtype=torch.bool)
            for i in range(batch_size):
                actual_text_len = text_lengths[i].item() + 1  # +1 for stop token
                text_mask[i, :actual_text_len] = True
            
            # Mel mask: consider actual codes length + stop token (if any)
            mel_mask = torch.zeros_like(mel_targets, dtype=torch.bool)
            for i in range(batch_size):
                actual_mel_len = codes_lengths[i].item() + int(add_mel_stop_token)
                mel_mask[i, :actual_mel_len] = True
            
            loss_text = F.cross_entropy(text_logits, text_targets.long(), reduction='none')
            loss_mel = F.cross_entropy(mel_logits, mel_targets.long(), reduction='none')
            
            # Apply masking and reduce - only calculate loss on valid positions
            loss_text = (loss_text * text_mask).sum() / text_mask.sum() if text_mask.sum() > 0 else torch.tensor(0.0, device=text_logits.device)
            loss_mel = (loss_mel * mel_mask).sum() / mel_mask.sum() if mel_mask.sum() > 0 else torch.tensor(0.0, device=mel_logits.device)
            
            outputs["loss"] = (loss_text, loss_mel)
            
            # Calculate mel prediction accuracy
            with torch.no_grad():
                # Apply mask to get valid positions for accuracy calculation
                mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
                mel_targets_flat = mel_targets.view(-1)
                mel_mask_flat = mel_mask.view(-1)
                
                # Only calculate accuracy on valid positions
                if mel_mask_flat.sum() > 0:
                    valid_mel_logits = mel_logits_flat[mel_mask_flat]
                    valid_mel_targets = mel_targets_flat[mel_mask_flat]
                    mel_acc_1, mel_acc_10, mel_acc_20 = top_k_accuracy(valid_mel_logits, valid_mel_targets, k=(1, 10, 20))
                    outputs["mel_accuracy"] = {"acc_1": mel_acc_1, "acc_10": mel_acc_10, "acc_20": mel_acc_20}
                else:
                    outputs["mel_accuracy"] = {"acc_1": 0.0, "acc_10": 0.0, "acc_20": 0.0}
                    
        if output_logits:
            outputs["logits"] = (text_logits, mel_logits)
            outputs["targets"] = (text_targets, mel_targets)
        

    if output_latent:
        outputs["latent"] = gpt2_outputs["latent"]
        
    clear_torch_cache()
    return outputs

def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: Tuple[int, ...] = (1, 10, 20)) -> List[float]:
    """计算 top-k 准确率。"""
    max_k = max(k)
    _, topk_preds = torch.topk(logits, max_k, dim=1)  # (B*L, max_k)
    
    # Reshape for comparison
    targets_reshaped = targets.view(-1, 1) # (B*L, 1)
    topk_preds_reshaped = topk_preds.view(-1, max_k) # (B*L, max_k)

    # Remove unused variable and optimize calculation
    # correct = (topk_preds_reshaped == targets_reshaped).any(dim=-1)
    
    res = []
    for ki in k:
        # Check if the target is in the top-ki predictions
        correct_k = (topk_preds_reshaped[:, :ki] == targets_reshaped).any(dim=-1)
        acc = correct_k.float().mean().item() * 100
        res.append(acc)
    return res

class Trainer:
    """封装 UnifiedVoice 微调过程的训练器。"""
    def __init__(self, config: DictConfig):
        """
        初始化训练器。

        Args:
            config (DictConfig): 从 YAML 文件加载的 OmegaConf 配置对象。
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        self._set_seed(self.config.train.seed)

        # 准备目录和日志
        self.finetune_dir = self.config.train.finetune_model_dir
        self.checkpoint_dir = os.path.join(self.finetune_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._setup_logging()

        # 加载说话人条件向量
        self.speaker_conditions = load_speaker_conditions(config)
        self.speaker_list = list(self.speaker_conditions.keys())
        logger.info(f"Loaded conditions for {len(self.speaker_list)} speakers: {self.speaker_list}")

        # 加载模型和分词器
        self._load_models()

        # 设置优化器和调度器
        self._setup_optimizer_and_scheduler()

        # 初始化训练状态
        self.best_val_loss = (0, float('inf'), float('inf'))  # (epoch, text_loss, mel_loss)
        self.update_steps = 0

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Set random seed to {seed}")

    def _setup_logging(self):
        """配置 loguru 日志记录器。"""
        log_path = os.path.join(self.checkpoint_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        logger.add(log_path, level="INFO", encoding="utf-8")
        logger.info("Logging configured. Logs will be saved to console and file.")
        logger.info("Full configuration:\n" + OmegaConf.to_yaml(self.config))

    def _load_models(self):
        """加载 BPE、BigVGAN 和 UnifiedVoice 模型。"""
        logger.info("Loading models...")
        # BPE
        bpe_model_path = os.path.join(self.finetune_dir, self.config.dataset.bpe_model)
        self.bpe_model = spm.SentencePieceProcessor(bpe_model_path)
        logger.info("BPE model loaded.")
        
        # BigVGAN (用于可能的推理测试)
        # bigvgan_checkpoint_path = os.path.join(self.finetune_dir, self.config.bigvgan_checkpoint)
        # self.bigvgan = BigVGAN(self.config.bigvgan)
        # bigvgan_state_dict = torch.load(bigvgan_checkpoint_path, map_location="cpu", weights_only=True)["generator"]
        # self.bigvgan.load_state_dict(bigvgan_state_dict, strict=True)
        # self.bigvgan.remove_weight_norm()
        # self.bigvgan.eval().to(self.device)
        # logger.info("BigVGAN model loaded.")
        # del bigvgan_state_dict
        
        # UnifiedVoice
        gpt_checkpoint_path = os.path.join(self.finetune_dir, self.config.gpt_checkpoint)
        self.model = load_UnifiedVoice(self.config.gpt, gpt_checkpoint_path, self.device)
        logger.info("UnifiedVoice base model loaded.")
    
        # 应用 LoRA（这会冻结所有参数）
        self.model = self._apply_lora(self.model)
        logger.info("LoRA applied to the model.")
        
        # 注册多说话人的mean_condition作为可学习参数
        self.speaker_mean_conditions = {}
        for speaker_id, condition in self.speaker_conditions.items():
            # 验证和调整形状
            if condition.ndim == 2:  # (32, dim) -> (1, 32, dim)
                condition = condition.unsqueeze(0)
            elif condition.ndim != 3:
                raise ValueError(f"Expected condition tensor to have 2 or 3 dimensions, got {condition.ndim} for speaker {speaker_id}")
            
            # 将每个说话人的condition注册为可学习参数
            param_name = f"mean_condition_{speaker_id}"
            param = torch.nn.Parameter(condition.to(self.device), requires_grad=True)
            
            # 使用正确的方式注册参数
            self.model.register_parameter(param_name, param)
            self.speaker_mean_conditions[speaker_id] = param
            
            logger.debug(f"Registered parameter {param_name} with shape {param.shape}")
        
        logger.info(f"Loaded and registered {len(self.speaker_mean_conditions)} speaker conditions.")

    def _apply_lora(self, model: UnifiedVoice) -> UnifiedVoice:
        """为模型配置并应用 LoRA。"""
        lora_cfg = self.config.train.lora
        gpt_lora_config = LoraConfig(
            r=lora_cfg.r,
            target_modules=lora_cfg.target_modules,
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            bias="none",
        )
        model.requires_grad_(False)
        model.inference_model = get_peft_model(model.inference_model, gpt_lora_config)
        return model

    def _setup_optimizer_and_scheduler(self, num_training_steps: int = 1000):
        """创建优化器和学习率调度器。"""
        opt_cfg = self.config.train.optimizer
        optimizer = create_loraplus_optimizer(
            model=self.model,
            optimizer_cls=AdamW,
            lr=opt_cfg.learning_rate,
            loraplus_lr_ratio=opt_cfg.loraplus_lr_ratio,
            loraplus_weight_decay=opt_cfg.weight_decay,
        )
        self.optimizer = optimizer

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * opt_cfg.warmup_ratio),
            num_training_steps=num_training_steps,
        )
        self.scheduler = scheduler
        logger.info("Optimizer (LoRA+) and Scheduler (CosineAnnealingWithWarmup) created.")

    def _train_step(self, data_batch: tuple) -> Tuple[torch.Tensor, torch.Tensor, dict]:  # 修正：应该返回三个值
        """执行单个训练步骤（前向和后向传播）。"""
        self.model.train()
        self.model.inference_model.kv_cache = False  # 训练时禁用 KV 缓存
    
        # Unpack data_batch: mel_spec, mel_codes, text_ids, conditions, speaker_ids, mel_lengths, codes_lengths, text_lengths
        mel_spec, mel_codes, text_ids, conditions, speaker_ids, mel_lengths, codes_lengths, text_lengths = data_batch
        outputs = forward_UnifiedVoice(
            self.model,
            mel_spec,
            mel_codes,
            text_ids,
            mel_lengths,
            codes_lengths,
            text_lengths,
            speaker_ids=speaker_ids,  # 确保这一行存在
            output_loss=True,
            output_logits=True,
        )
        loss_text, loss_mel = outputs["loss"]
        mel_accuracy = outputs.get("mel_accuracy", {"acc_1": 0.0, "acc_10": 0.0, "acc_20": 0.0})
        return loss_text, loss_mel, mel_accuracy

    @torch.no_grad()
    def _validate_epoch(self, valid_ds: Dataset, epoch: int):
        """在验证集上评估模型。"""
        self.model.eval()
        logger.info(f"Validating at epoch {epoch + 1}...")
        
        total_text_loss, total_mel_loss = 0.0, 0.0
        total_text_tokens, total_mel_tokens = 0, 0
        all_mel_logits, all_mel_targets = [], []
        num_batches = 0

        for batch in tqdm(valid_ds, desc="Validation", dynamic_ncols=True):
            # 正确处理新的 data_batch 格式，区分 tensor 和非 tensor 数据
            data_batch = []
            for item in batch:
                if torch.is_tensor(item):
                    data_batch.append(item.to(self.device))
                else:
                    data_batch.append(item)  # speaker_ids 是字符串列表，不需要 .to(device)

            # 正确解包新格式：mel_spec, mel_codes, text_ids, conditions, speaker_ids, mel_lengths, codes_lengths, text_lengths
            mel_spec, mel_codes, text_ids, conditions, speaker_ids, mel_lengths, codes_lengths, text_lengths = data_batch
            
            outputs = forward_UnifiedVoice(
                self.model,
                mel_spec,
                mel_codes,
                text_ids,
                mel_lengths,
                codes_lengths,
                text_lengths,
                speaker_ids=speaker_ids,  # 添加这一行
                output_loss=True,
                output_logits=True,
            )
            
            loss_text, loss_mel = outputs["loss"]
            batch_text_tokens = text_lengths.sum().item()
            batch_mel_tokens = (codes_lengths + 1).sum().item()  # +1 for stop token

            total_text_loss += loss_text.item() * batch_text_tokens
            total_mel_loss += loss_mel.item() * batch_mel_tokens
            total_text_tokens += batch_text_tokens
            total_mel_tokens += batch_mel_tokens
            num_batches += 1

            # Collect logits and targets for accuracy calculation
            # 仅使用 MEL 部分的 logits 和 targets
            current_mel_logits = outputs["logits"][1]  # mel logits [B, V, L]
            current_mel_targets = outputs["targets"][1]  # mel targets [B, L]
            if current_mel_logits.numel() > 0 and current_mel_targets.numel() > 0:
                # Create mask based on actual sequence lengths instead of assuming 0 is padding
                batch_size = current_mel_targets.size(0)
                mel_mask = torch.zeros_like(current_mel_targets, dtype=torch.bool)
                for i in range(batch_size):
                    actual_mel_len = codes_lengths[i].item() + 1  # +1 for stop token if add_mel_stop_token is True
                    mel_mask[i, :actual_mel_len] = True
                
                valid_mask = mel_mask.view(-1)
                if valid_mask.sum() > 0:
                    mel_logits_filtered = current_mel_logits.permute(0, 2, 1).reshape(-1, current_mel_logits.size(1))[valid_mask]
                    mel_targets_filtered = current_mel_targets.view(-1)[valid_mask]
                    all_mel_logits.append(mel_logits_filtered)
                    all_mel_targets.append(mel_targets_filtered)
            
            clear_torch_cache()

        avg_text_loss = total_text_loss / total_text_tokens
        avg_mel_loss = total_mel_loss / total_mel_tokens
        
        # 计算整体准确率
        all_mel_logits = torch.cat(all_mel_logits, dim=0)
        all_mel_targets = torch.cat(all_mel_targets, dim=0)
        acc_1, acc_10, acc_20 = top_k_accuracy(all_mel_logits, all_mel_targets, k=(1, 10, 20))

        logger.info(f"**Validation results at epoch {epoch + 1}**")
        logger.info(f"Text Loss: {avg_text_loss:.4f}, Mel Loss: {avg_mel_loss:.4f}")
        logger.info(f"Accuracy@1: {acc_1:.2f}%, Accuracy@10: {acc_10:.2f}%, Accuracy@20: {acc_20:.2f}%")
        
        return avg_text_loss, avg_mel_loss, acc_1, acc_10, acc_20

    def _save_checkpoint(self, file_name: str, merge_lora: bool, unload_after_merge: bool):
        """保存模型检查点，包含说话人信息。"""
        checkpoint_path = os.path.join(self.checkpoint_dir, file_name)
        
        self.model.eval()
        
        model_to_save = self.model
        
        if merge_lora:
            logger.info("Merging LoRA weights into the model for saving...")
            if unload_after_merge:
                # 为了在不影响继续训练的情况下保存完全融合的模型，我们创建一个深拷贝
                # 注意：这可能会消耗较多内存和时间
                logger.info("Creating a deep copy of the model for a clean merge. This may take a moment...")
                model_to_save = copy.deepcopy(self.model)
                
                # 在深拷贝上进行融合与卸载
                fused_inference_model = model_to_save.inference_model.merge_and_unload()
                model_to_save.inference_model = fused_inference_model
                logger.info("LoRA weights merged and unloaded in the copied model.")
            else:
                # 如果只是临时融合，直接在原模型上操作，后续再unmerge
                self.model.inference_model.merge_adapter()
    
        # 保存模型状态和说话人信息
        state_dict = model_to_save.state_dict()
        checkpoint_data = {
            'model': state_dict,
            'speakers': self.speaker_list,  # 保存说话人列表
            'speaker_conditions': {speaker_id: param.detach().cpu().numpy() 
                                 for speaker_id, param in self.speaker_mean_conditions.items()}
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        logger.info(f"Saved conditions for speakers: {self.speaker_list}")
    
        # 如果创建了深拷贝，清理它
        if merge_lora and unload_after_merge:
            del model_to_save
            clear_torch_cache()
            logger.info("Cleaned up the temporary merged model.")
        
        # 如果是临时融合，恢复模型状态以便继续训练
        if merge_lora and not unload_after_merge:
            logger.info("Unmerging LoRA weights to continue training...")
            self.model.inference_model.unmerge_adapter()
                
        self.model.train()

    def train(self, train_ds: Dataset, valid_ds: Dataset):
        """
        启动完整的训练流程。

        Args:
            train_ds (Dataset): 训练数据集。
            valid_ds (Dataset): 验证数据集。
        """
        train_cfg = self.config.train
        total_ds_count = len(train_ds)
        
        samples_per_epoch = total_ds_count
        total_update_steps = samples_per_epoch * train_cfg.epochs
        self._setup_optimizer_and_scheduler(num_training_steps=total_update_steps)
        
        logger.info(f"Starting training for {train_cfg.epochs} epochs.")
        logger.info(f"Total samples per epoch: {samples_per_epoch}")
        logger.info(f"Total update steps: {total_update_steps}")

        text_weight = train_cfg.text_weight

        for epoch in range(train_cfg.epochs):
            logger.info(f"EPOCH {epoch + 1}/{train_cfg.epochs} started" + "=" * 30)

            for batch_idx, batch in enumerate(train_ds):
                # 正确处理新的 data_batch 格式，区分 tensor 和非 tensor 数据
                data_batch = []
                for item in batch:
                    if torch.is_tensor(item):
                        data_batch.append(item.to(self.device))
                    else:
                        data_batch.append(item)  # speaker_ids 是字符串列表，不需要 .to(device)

                loss_text, loss_mel, mel_accuracy = self._train_step(tuple(data_batch))
                acc_1, acc_10, acc_20 = mel_accuracy["acc_1"], mel_accuracy["acc_10"], mel_accuracy["acc_20"]

                weighted_loss = text_weight * loss_text + (1.0 - text_weight) * loss_mel
                if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                    logger.warning(f"NaN or Inf loss at epoch {epoch}, batch {batch_idx}. Skipping.")
                    continue

                # ------------------ Optimisation Step ------------------
                self.optimizer.zero_grad()
                weighted_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.update_steps += 1

                # Logging
                logger.info(
                    f"Epoch {epoch + 1}/{train_cfg.epochs} | Batch {batch_idx + 1}/{len(train_ds)} | "
                    f"text_loss={loss_text.item():.4f}, mel_loss={loss_mel.item():.4f}, "
                    f"acc@1={acc_1:.2f}%, acc@10={acc_10:.2f}%, acc@20={acc_20:.2f}%, "
                    f"grad_norm={grad_norm.item():.2f}"
                )

            # --- Epoch End ---
            val_text_loss, val_mel_loss, _, _, _ = self._validate_epoch(valid_ds, epoch)
            
            # 每个epoch结束后都保存当前模型
            epoch_checkpoint_name = f"gpt_epoch_{epoch + 1}.pth"
            logger.info(f"Saving model for epoch {epoch + 1}: {epoch_checkpoint_name}")
            self._save_checkpoint(epoch_checkpoint_name, merge_lora=True, unload_after_merge=True)
            
            # 检查早停
            #patience = train_cfg.early_stopping_patience
            #if epoch > 0 and val_mel_loss >= self.best_val_loss[2] and (epoch - self.best_val_loss[0]) >= patience:
            #    logger.info(
            #        f"Early stopping at epoch {epoch + 1} as validation mel_loss has not improved for {patience} epochs."
            #    )
            #    break

            if val_mel_loss < self.best_val_loss[2]:
                logger.info(f"New best validation mel_loss: {val_mel_loss:.4f}. Saving best model.")
                self.best_val_loss = (epoch, val_text_loss, val_mel_loss)
                self._save_checkpoint("gpt_best.pth", merge_lora=True, unload_after_merge=True)

            clear_torch_cache()

        # --- Training End ---
        logger.info("Training finished. Saving final model.")
        self._save_checkpoint("gpt_finetuned.pth", merge_lora=True, unload_after_merge=True)
        
        # 保存最终配置
        final_config_path = os.path.join(self.finetune_dir, "config_finetuned.yaml")
        final_config = self.config.copy()
        final_config.gpt_checkpoint = "checkpoints/gpt_finetuned.pth"
        OmegaConf.save(final_config, final_config_path)
        logger.info(f"Final config saved to {final_config_path}")
        
        logger.info(f"Best validation loss at epoch {self.best_val_loss[0] + 1}: "
                    f"text_loss: {self.best_val_loss[1]:.4f}, mel_loss: {self.best_val_loss[2]:.4f}")

def main():
    config_path = "finetune_models/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}.")
    
    config = OmegaConf.load(config_path)
    bpe_model_path = os.path.join(config.train.finetune_model_dir, config.dataset.bpe_model)

    # 使用新的多说话人数据加载函数，传递BPE路径而不是对象
    train_ds, valid_ds = load_finetune_datasets(config, bpe_model_path) 
    train_ds = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune_fn, num_workers=4)
    valid_ds = DataLoader(valid_ds, batch_size=8, shuffle=False, collate_fn=collate_finetune_fn, num_workers=2)

    trainer = Trainer(config)
    trainer.train(train_ds, valid_ds)
    logger.info("Finetuning UnifiedVoice completed.")


if __name__ == "__main__":
    main()