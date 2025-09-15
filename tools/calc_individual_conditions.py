#!/usr/bin/env python3
"""
读取 jsonl 清单中的 mel 谱文件，通过 UnifiedVoice 的 `get_conditioning` 计算每条样本的
condition embedding，并将每个样本的 condition 保存为单独的 .npy 文件，
同时更新 jsonl 文件添加 condition 字段。

用法示例：
    python tools/calc_individual_conditions.py \
        --manifest finetune_data/M00100_train.jsonl \
        --config finetune_models/config.yaml \
        --output_dir finetune_data/conditions \
        --output_jsonl finetune_data/M00100_train_with_conditions.jsonl
"""
import argparse
import json
import os
import sys
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

# --- 解析命令行 ---
parser = argparse.ArgumentParser()
parser.add_argument('--manifest', "-m", required=True, help='jsonl 数据清单')
parser.add_argument('--config', "-c", required=True, help='finetune_models/config.yaml')
parser.add_argument('--output_dir', "-o", default='conditions', help='condition 文件输出目录')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# --- 加载配置并构建模型 ---
cfg = OmegaConf.load(args.config)
ckpt_path = "checkpoints/gpt.pth.open_source"

sys.path.append('.')  # 方便脚本在项目根目录外执行
from indextts.gpt.model import UnifiedVoice  # noqa: E402

uv = UnifiedVoice(**cfg.gpt)

print(f"loading checkpoint from {ckpt_path}")
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
uv.load_state_dict(state['model'] if 'model' in state else state, strict=True)
uv.eval().to(args.device)

# --- 遍历样本，计算并保存每个 condition embedding ---
count = 0
updated_items = []

with open(args.manifest, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        item = json.loads(line)
        
        # 读取 mel 文件
        mel_np: np.ndarray = np.load(item['mels'])  # (1, 100, T)
        if mel_np.ndim == 3 and mel_np.shape[0] == 1:  # (1, 100, T)
            mel_np = mel_np.squeeze(0)
        else:
            raise ValueError(f"未知的 mel 形状: {mel_np.shape}")

        mel = torch.from_numpy(mel_np).unsqueeze(0).to(args.device)  # (1, 100, T)
        length = torch.tensor([mel_np.shape[0]], device=args.device)
        
        # 计算 condition embedding
        with torch.no_grad():
            cond = uv.get_conditioning(mel, length)  # (1, 32, dim)
            cond = cond.squeeze(0).cpu().float().numpy()  # (32, dim)
        
        # 生成 condition 文件名
        # 使用原始音频文件名作为基础，或者使用行号
        if 'audio' in item:
            base_name = os.path.splitext(os.path.basename(item['audio']))[0]
        else:
            base_name = f"sample_{line_idx:06d}"
        
        condition_filename = f"{base_name}_condition.npy"
        condition_path = os.path.join(args.output_dir, condition_filename)
        
        # 保存 condition embedding
        np.save(condition_path, cond)
        
        # 更新 item，添加 condition 字段
        updated_item = item.copy()
        updated_item['condition'] = os.path.abspath(condition_path)
        updated_items.append(updated_item)
        
        count += 1
        if count % 100 == 0:
            print(f"processed {count} samples", file=sys.stderr)

if count == 0:
    raise RuntimeError('manifest 为空或读取失败')

# 保存更新后的 jsonl 文件
with open(os.path.join(args.output_dir, "output.jsonl"), 'w', encoding='utf-8') as f:
    for item in updated_items:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"处理完成！共处理 {count} 个样本")
print(f"condition 文件保存到: {args.output_dir}")