# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

import torch
import yaml
from loguru import logger


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    checkpoint = torch.load(model_pth, map_location='cpu', weights_only=False)
    
    # 获取模型设备
    device = next(model.parameters()).device
    
    # 处理多说话人的 speaker_conditions（在checkpoint顶层）
    if 'speaker_conditions' in checkpoint:
        logger.info(f"Loading multi-speaker conditions from checkpoint")
        speaker_conditions = checkpoint['speaker_conditions']
        
        for speaker_id, condition_array in speaker_conditions.items():
            # 转换为tensor并移动到正确设备
            condition_tensor = torch.from_numpy(condition_array).float().to(device)
            
            # 确保维度正确 (1, 32, dim)
            if condition_tensor.dim() == 2:
                condition_tensor = condition_tensor.unsqueeze(0)
            
            # 注册为模型参数
            param_name = f"mean_condition_{speaker_id}"
            setattr(model, param_name, torch.nn.Parameter(condition_tensor))
            logger.info(f"Loaded speaker {speaker_id} condition: {condition_tensor.shape}")
    
    # 获取model部分进行加载
    model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 处理单一的 mean_condition（向后兼容，在model_state_dict中）
    if 'mean_condition' in model_state_dict and hasattr(model, 'mean_condition'):
        logger.info(f"Loading single mean_condition: checkpoint device={model_state_dict['mean_condition'].device}, target device={device}")
        model.mean_condition = model_state_dict['mean_condition'].to(device)
        logger.info(f"After loading: mean_condition device={model.mean_condition.device}")
        # 从model_state_dict中移除，避免在load_state_dict时冲突
        model_state_dict = {k: v for k, v in model_state_dict.items() if k != 'mean_condition'}
    
    model.load_state_dict(model_state_dict, strict=False)
    
    info_path = re.sub('.pth$', '.yaml', model_pth)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    
    # 返回说话人列表信息（如果存在，在checkpoint顶层）
    if 'speakers' in checkpoint:
        configs['speakers'] = checkpoint['speakers']
    
    return configs
