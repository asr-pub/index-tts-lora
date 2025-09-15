#!/usr/bin/env python3
"""
音频特征提取工具

该脚本用于处理音频数据，提取梅尔频谱特征、离散代码本索引和condition latent，
并计算medoid样本用于语音合成模型的训练。

主要功能：
1. 音频数据预处理和特征提取
2. 离散变分自编码器(DVAE)编码
3. Condition latent提取
4. Medoid样本计算
5. 训练/验证集分割
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.vqvae.xtts_dvae import DiscreteVAE

# 常量定义
DEFAULT_OUTPUT_DIR = "finetune_data/processed_data/"
DEFAULT_MODEL_PATH = "checkpoints/gpt.pth.open_source"
FINETUNE_MODEL_DIR = "finetune_models"
CONFIG_FILENAME = "config.yaml"
METADATA_FILENAME = "metadata.jsonl"
TRAIN_SPLIT_RATIO = 0.9
CONDITION_LATENT_DIM = 32


class AudioProcessor:
    """音频处理器类，封装音频特征提取相关功能"""
    
    def __init__(self, dvae: DiscreteVAE, mel_config: Dict, device: str = 'cuda'):
        self.dvae = dvae
        self.mel_config = mel_config
        self.device = device
        self.mel_feature = MelSpectrogramFeatures(**mel_config)
    
    @torch.no_grad()
    def process_audio_data(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sr: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理音频数据，包括提取梅尔频谱特征、获取离散代码本索引等。

        Args:
            audio: 输入的音频数据
            sr: 音频的采样率

        Returns:
            处理后的音频数据、梅尔频谱特征和离散代码本索引
        """
        # 数据类型转换
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        
        # 重采样
        if sr != self.mel_config['sample_rate']:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, 
                new_freq=self.mel_config['sample_rate']
            )
            audio = resampler(audio)
        
        # 提取梅尔频谱特征
        mel = self.mel_feature(audio)
        
        # 获取离散代码本索引
        codes = self.dvae.get_codebook_indices(mel)
        
        # 处理音频维度
        if audio.ndim > 1 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        
        return audio, mel, codes


class ConditionExtractor:
    """Condition latent提取器类"""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def extract_condition_latent(
        self, 
        mel: torch.Tensor
    ) -> np.ndarray:
        """
        从梅尔频谱中提取condition latent
        
        Args:
            mel: 梅尔频谱特征 (1, mel_dim, T) 或 (mel_dim, T)
        
        Returns:
            condition latent (32, dim)
        """
        if self.model is None:
            raise ValueError("模型不能为None，需要传入已加载的UnifiedVoice模型")
        
        # 确保mel数据格式正确
        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        
        # 确保mel是3维的 (1, mel_dim, T)
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        elif mel.ndim == 3 and mel.shape[0] == 1:
            pass
        else:
            raise ValueError(f"不支持的mel形状: {mel.shape}")
        
        mel = mel.to(self.device)
        mel_length = torch.tensor([mel.shape[-1]], device=self.device)
        
        # 使用UnifiedVoice模型提取condition latent
        with torch.no_grad():
            condition = self.model.get_conditioning(mel, mel_length)
            condition = condition.squeeze(0).cpu().float().numpy()
        
        return condition


class MedoidCalculator:
    """Medoid计算器类"""
    
    @staticmethod
    def compute_distance_matrix(
        data: np.ndarray, 
        metric: str = 'euclidean'
    ) -> np.ndarray:
        """
        计算距离矩阵
        
        Args:
            data: 数据矩阵 (N, features)
            metric: 距离度量方式 ('euclidean' 或 'cosine')
        
        Returns:
            距离矩阵 (N, N)
        """
        n = data.shape[0]
        dist_matrix = np.zeros((n, n))
        
        for i in tqdm(range(n), desc="计算距离矩阵"):
            for j in range(i + 1, n):
                if metric == 'euclidean':
                    dist = np.linalg.norm(data[i] - data[j])
                elif metric == 'cosine':
                    dist = 1 - np.dot(data[i], data[j]) / (
                        np.linalg.norm(data[i]) * np.linalg.norm(data[j])
                    )
                else:
                    raise ValueError(f"不支持的距离度量: {metric}")
                
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    @classmethod
    def find_medoid_condition(
        cls, 
        condition_files: List[str], 
        distance_metric: str = 'euclidean'
    ) -> Dict:
        """
        找出condition latent分布中心的样本（medoid）
        
        Args:
            condition_files: condition文件路径列表
            distance_metric: 距离度量方式
        
        Returns:
            medoid信息字典
        """
        logger.info(f"开始计算medoid，共{len(condition_files)}个样本")
        
        # 读取所有condition latent
        conditions = []
        condition_infos = []
        
        for condition_path in condition_files:
            if not os.path.exists(condition_path):
                logger.warning(f"Condition文件不存在: {condition_path}，跳过")
                continue
            
            try:
                cond = np.load(condition_path)
                conditions.append(cond)
                condition_infos.append({
                    'index': len(conditions) - 1,
                    'condition_path': condition_path
                })
            except Exception as e:
                logger.error(f"加载condition文件失败: {condition_path}, 错误: {e}")
                continue
        
        if len(conditions) == 0:
            raise RuntimeError('没有找到有效的condition文件')
        
        logger.info(f"成功读取{len(conditions)}个condition latent")
        
        # 将所有condition堆叠成一个数组
        conditions_array = np.stack(conditions, axis=0)
        logger.info(f"Conditions数组形状: {conditions_array.shape}")
        
        # 将数据展平以便计算距离
        N, H, W = conditions_array.shape
        data_flat = conditions_array.reshape(N, H * W)
        
        # 计算距离矩阵
        logger.info("计算距离矩阵...")
        dist_matrix = cls.compute_distance_matrix(data_flat, distance_metric)
        
        # 找出medoid
        logger.info("寻找medoid...")
        dist_sums = np.sum(dist_matrix, axis=1)
        medoid_idx = np.argmin(dist_sums)
        medoid_info = condition_infos[medoid_idx]
        medoid_condition = conditions[medoid_idx]
        
        logger.info(f"找到medoid: 索引{medoid_idx}")
        logger.info(f"Medoid到所有样本的距离之和: {dist_sums[medoid_idx]:.6f}")
        
        # 计算统计信息
        stats = {
            'total_samples': len(conditions),
            'medoid_index': int(medoid_idx),
            'medoid_distance_sum': float(dist_sums[medoid_idx]),
            'distance_metric': distance_metric,
            'distance_stats': {
                'mean': float(np.mean(dist_sums)),
                'std': float(np.std(dist_sums)),
                'min': float(np.min(dist_sums)),
                'max': float(np.max(dist_sums))
            }
        }
        
        return {
            'medoid_info': medoid_info,
            'medoid_condition': medoid_condition,
            'statistics': stats,
            'distance_matrix': dist_matrix
        }


def load_unified_voice_model(
    config_path: str, 
    model_path: str, 
    device: str = 'cuda'
):
    """
    加载UnifiedVoice模型
    
    Args:
        config_path: 配置文件路径
        model_path: 模型检查点路径
        device: 计算设备
    
    Returns:
        加载好的UnifiedVoice模型
    """
    sys.path.append('.')
    from indextts.gpt.model import UnifiedVoice
    
    # 加载配置
    cfg = OmegaConf.load(config_path)
    
    # 创建模型
    model = UnifiedVoice(**cfg.gpt)
    
    # 加载预训练权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型检查点文件未找到: {model_path}")
    
    logger.info(f"正在加载UnifiedVoice模型检查点: {model_path}")
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(
        state['model'] if 'model' in state else state, 
        strict=True
    )
    model.eval().to(device)
    
    logger.info("UnifiedVoice模型加载完成")
    return model


def parse_audio_line(line: str) -> Tuple[str, str]:
    """解析音频列表文件中的一行"""
    line = line.strip()
    if not line:
        raise ValueError("空行")
    
    if "\t" in line:
        return line.split("\t", 1)
    elif "|" in line:
        return line.split("|", 1)
    else:
        raise ValueError(f"不支持的格式: {line}")


def save_medoid_results(
    medoid_result: Dict, 
    output_dir: str
) -> None:
    """保存medoid计算结果"""
    # 保存medoid信息
    medoid_info_path = os.path.join(output_dir, "medoid_info.json")
    with open(medoid_info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'medoid_info': medoid_result['medoid_info'],
            'statistics': medoid_result['statistics'],
            'medoid_condition_shape': list(medoid_result['medoid_condition'].shape)
        }, f, ensure_ascii=False, indent=2)
    
    # 保存medoid condition
    medoid_condition_path = os.path.join(output_dir, "medoid_condition.npy")
    np.save(medoid_condition_path, medoid_result['medoid_condition'])
    
    # 保存距离矩阵
    distance_matrix_path = os.path.join(output_dir, "distance_matrix.npy")
    np.save(distance_matrix_path, medoid_result['distance_matrix'])
    
    logger.info("Medoid计算完成:")
    logger.info(f"  - Medoid信息: {medoid_info_path}")
    logger.info(f"  - Medoid condition: {medoid_condition_path}")
    logger.info(f"  - 距离矩阵: {distance_matrix_path}")
    logger.info(f"  - Medoid样本: {medoid_result['medoid_info']['condition_path']}")


def split_dataset(
    metadata_file: str, 
    output_dir: str
) -> Tuple[str, str]:
    """分割数据集为训练集和验证集"""
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 随机打乱数据
    np.random.shuffle(lines)

    # 计算分割点
    valid_size = int(len(lines) * (1 - TRAIN_SPLIT_RATIO))

    
    # 分割数据
    valid_lines = lines[-valid_size:]
    train_lines = lines[:-valid_size]
    
    # 保存训练集
    train_file = os.path.join(output_dir, 'metadata_train.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # 保存验证集
    valid_file = os.path.join(output_dir, 'metadata_valid.jsonl')
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    
    logger.info("数据集分割完成:")
    logger.info(f"  - 训练集: {train_file} ({len(train_lines)}条数据)")
    logger.info(f"  - 验证集: {valid_file} ({len(valid_lines)}条数据)")
    
    return train_file, valid_file


def save_speaker_info(
    audio_list_path: str,
    output_dir: str,
    lines: List[str]
) -> None:
    """保存说话人信息"""
    total_duration = sum(json.loads(line)["duration"] for line in lines)
    speaker_info = {
        "speaker": Path(audio_list_path).stem,
        "avg_duration": total_duration / len(lines),
        "sample_num": len(lines),
        "total_duration_in_seconds": total_duration,
        "total_duration_in_minutes": total_duration / 60,
        "total_duration_in_hours": total_duration / 3600,
        "train_jsonl": os.path.abspath(os.path.join(output_dir, "metadata_train.jsonl")),
        "valid_jsonl": os.path.abspath(os.path.join(output_dir, "metadata_valid.jsonl")),
        "medoid_condition": os.path.abspath(os.path.join(output_dir, "medoid_condition.npy")),
    }
    
    speaker_info_file = os.path.join(output_dir, "..", 'speaker_info.json')
    
    # 读取现有信息或创建新列表
    if os.path.exists(speaker_info_file):
        with open(speaker_info_file, 'r', encoding='utf-8') as f:
            speaker_info_list = json.load(f)
    else:
        speaker_info_list = []
    
    speaker_info_list.append(speaker_info)
    
    # 保存更新后的信息
    with open(speaker_info_file, 'w', encoding='utf-8') as f:
        json.dump(speaker_info_list, f, ensure_ascii=False, indent=4)


def setup_models(
    config: OmegaConf, 
    args: argparse.Namespace
) -> Tuple[DiscreteVAE, Optional[object]]:
    """设置和加载模型"""
    # 加载DiscreteVAE模型
    logger.info("正在加载 DiscreteVAE 模型...")
    dvae = DiscreteVAE(**config.vqvae)
    
    dvae_checkpoint_path = os.path.join(FINETUNE_MODEL_DIR, config.dvae_checkpoint)
    pre_trained_dvae = torch.load(
        dvae_checkpoint_path, 
        map_location=args.device, 
        weights_only=True
    )
    dvae.load_state_dict(
        pre_trained_dvae["model"] if "model" in pre_trained_dvae else pre_trained_dvae,
        strict=True
    )
    dvae.eval()
    del pre_trained_dvae
    
    # 加载UnifiedVoice模型（如果需要）
    unified_voice_model = None
    if args.extract_condition:
        config_path = os.path.join(FINETUNE_MODEL_DIR, CONFIG_FILENAME)
        unified_voice_model = load_unified_voice_model(
            config_path, args.model_path, args.device
        )
    
    return dvae, unified_voice_model


def process_audio_files(
    args: argparse.Namespace,
    config: OmegaConf,
    audio_processor: AudioProcessor,
    condition_extractor: Optional[ConditionExtractor],
    output_dir: str
) -> Tuple[str, List[str]]:
    """处理音频文件列表"""
    metadata_file = os.path.join(output_dir, METADATA_FILENAME)
    condition_files = []
    
    with open(args.audio_list, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="处理音频文件"):
            try:
                wav_path, txt = parse_audio_line(line)
            except ValueError as e:
                logger.warning(f"跳过无效行: {line.strip()}, 错误: {e}")
                continue
            
            if not os.path.exists(wav_path):
                logger.warning(f"音频文件未找到: {wav_path}")
                continue
            
            # 读取音频文件
            try:
                audio, sr = torchaudio.load(wav_path)
            except Exception as e:
                logger.error(f"读取音频文件时出错: {wav_path}, 错误信息: {e}")
                continue
            
            # 处理音频数据
            try:
                processed_audio, mel, codes = audio_processor.process_audio_data(audio, sr)
            except Exception as e:
                logger.error(f"处理音频数据时出错: {wav_path}, 错误信息: {e}")
                continue
            
            # 计算音频时长
            duration = processed_audio.shape[-1] / config.dataset.mel.sample_rate
            
            # 保存特征文件
            base_name = os.path.basename(wav_path)
            out_codebook = os.path.join(output_dir, f"{base_name}_codes.npy")
            out_mel = os.path.join(output_dir, f"{base_name}_mel.npy")
            
            try:
                np.save(out_codebook, codes.cpu().numpy())
                np.save(out_mel, mel.cpu().numpy())
            except Exception as e:
                logger.error(f"保存特征文件时出错: {out_codebook}, 错误信息: {e}")
                continue
            
            # 提取condition latent（如果需要）
            condition_path = None
            if condition_extractor is not None:
                try:
                    condition = condition_extractor.extract_condition_latent(mel)
                    condition_path = os.path.join(output_dir, f"{base_name}_condition.npy")
                    np.save(condition_path, condition)
                    condition_files.append(condition_path)
                except Exception as e:
                    logger.error(f"提取condition时出错: {wav_path}, 错误信息: {e}")
                    continue
            
            # 写入元数据
            try:
                data_entry = {
                    "audio": wav_path,
                    "text": txt,
                    "codes": os.path.abspath(out_codebook),
                    "mels": os.path.abspath(out_mel),
                    "duration": round(duration, 4)
                }
                
                if condition_path:
                    data_entry["condition"] = os.path.abspath(condition_path)
                
                with open(metadata_file, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"写入元数据时出错: {metadata_file}, 错误信息: {e}")
                continue
    
    return metadata_file, condition_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Process audio data using DiscreteVAE and find medoid condition."
    )
    parser.add_argument(
        "--audio_list", 
        type=str, 
        required=True, 
        help="Path to the input audio list file."
    )
    parser.add_argument(
        "--extract_condition", 
        action="store_true", 
        help="Extract condition latents and find medoid."
    )
    parser.add_argument(
        "--distance_metric", 
        default='euclidean', 
        choices=['euclidean', 'cosine'], 
        help="Distance metric for medoid calculation."
    )
    parser.add_argument(
        "--output_dir", 
        default=DEFAULT_OUTPUT_DIR, 
        help="Output directory for processed data."
    )
    parser.add_argument(
        "--model_path", 
        default=DEFAULT_MODEL_PATH, 
        help="Path to the UnifiedVoice model checkpoint."
    )
    parser.add_argument(
        "--device", 
        default='cuda' if torch.cuda.is_available() else 'cpu', 
        help="Device to use for computation."
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    config_path = os.path.join(FINETUNE_MODEL_DIR, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    
    # 加载配置
    config = OmegaConf.load(config_path)
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_list_name = Path(args.audio_list).stem
    output_dir = os.path.join(args.output_dir, f"{audio_list_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置模型
    dvae, unified_voice_model = setup_models(config, args)
    
    # 创建处理器
    audio_processor = AudioProcessor(dvae, config.dataset.mel, args.device)
    condition_extractor = None
    if unified_voice_model is not None:
        condition_extractor = ConditionExtractor(unified_voice_model, args.device)
    
    # 处理音频文件
    metadata_file, condition_files = process_audio_files(
        args, config, audio_processor, condition_extractor, output_dir
    )
    
    logger.info(f"特征提取完成，结果保存在: {output_dir}")
    
    # 计算medoid（如果需要）
    if args.extract_condition and condition_files:
        logger.info("开始计算medoid condition...")
        try:
            medoid_result = MedoidCalculator.find_medoid_condition(
                condition_files, args.distance_metric
            )
            save_medoid_results(medoid_result, output_dir)
        except Exception as e:
            logger.error(f"计算medoid时出错: {e}")
    
    # 分割数据集
    split_dataset(metadata_file, output_dir)
    
    # 保存说话人信息
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    save_speaker_info(args.audio_list, output_dir, lines)


if __name__ == "__main__":
    main()