import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# 导入与infer.py相同的文本处理类
from indextts.utils.front import TextNormalizer, TextTokenizer


class FinetuneDataset(Dataset):
    """
    Custom dataset used for UnifiedVoice fine-tuning, supporting multi-speaker data.
    """
    def __init__(self, manifest_files: List[str], bpe_path: str, speaker_ids: List[str], config: DictConfig):
        """
        Args:
            manifest_files (List[str]): List of paths to manifest files for different speakers.
            bpe_path (str): Path to the BPE model file.
            speaker_ids (List[str]): List of speaker IDs corresponding to manifest files.
            config (DictConfig): Extra preprocessing configuration.
        """
        super().__init__()
        self.config = config
        self.data = []
        
        # 使用与infer.py完全相同的文本处理器
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.tokenizer = TextTokenizer(bpe_path, self.normalizer)
        
        print(">> TextNormalizer loaded for training")
        print(f">> BPE model loaded from: {bpe_path}")
        
        # 加载所有说话人的数据
        for manifest_file, speaker_id in zip(manifest_files, speaker_ids):
            logger.info(f"Loading data from manifest: {manifest_file} for speaker: {speaker_id}")
            with open(manifest_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        # 根据音频时长过滤数据
                        item = json.loads(line.strip())

                        duration = item.get("duration", 0)
                        if duration > 20 or duration < 1:
                            continue

                        item['speaker_id'] = speaker_id  # 添加说话人ID
                        self.data.append(item)
        
        logger.info(f"Loaded {len(self.data)} samples from {len(manifest_files)} speakers.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        
        text = item["text"]
        codes_path = item.get("codes", None)
        mels_path = item.get("mels", None)
        condition_path = item.get("condition", None)
        speaker_id = item["speaker_id"]

        # 使用与infer.py完全相同的文本处理流程
        # 这确保训练和推理时的文本处理完全一致
        text_tokens_list = self.tokenizer.tokenize(text)  # 这里会调用normalize + tokenize_by_CJK_char
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        text_ids = torch.LongTensor(text_ids).unsqueeze(0)

        # 加载音频特征
        codes_npy = np.load(codes_path)
        mels_npy = np.load(mels_path)
        condition_npy = np.load(condition_path) if condition_path else None

        mel_spec = torch.FloatTensor(mels_npy)  # [B, D, T]
        mel_codes = torch.LongTensor(codes_npy)  # [B, T]
        condition = torch.FloatTensor(condition_npy) if condition_npy is not None else None

        return (mel_spec, mel_codes, text_ids, condition, speaker_id)


# ------------------------------------------------------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------------------------------------------------------


def _pad_sequence(seqs, pad_value=0, dim=-1):
    """Pad a list of tensors on the specified dimension to the max length.

    Args:
        seqs (List[torch.Tensor]): list of tensors with shape (..., L_i)
        pad_value (int|float): value to use for padding
        dim (int): dimension to pad on (default: last dimension)

    Returns:
        Tuple[torch.Tensor, torch.LongTensor]:
            - padded tensor of shape (*batch, max_len)
            - lengths tensor (B,)
    """

    assert dim == -1, "Padding dimension must be the last dimension"

    lengths = torch.tensor([s.shape[dim] for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()

    # Determine output shape
    out_shape = list(seqs[0].shape)
    out_shape[dim] = max_len
    out_shape = [len(seqs)] + out_shape  # prepend batch dim

    padded = seqs[0].new_full(out_shape, pad_value)

    for i, s in enumerate(seqs):
        # 在最后一维进行padding，复制原始数据到对应位置
        if s.dim() == 1:  # 1D tensor: [L] -> [B, L]
            padded[i, :s.shape[0]] = s
        elif s.dim() == 2:  # 2D tensor: [D, L] -> [B, D, L]
            padded[i, :, :s.shape[1]] = s
        elif s.dim() == 3:  # 3D tensor: [C, D, L] -> [B, C, D, L]
            padded[i, :, :, :s.shape[2]] = s
        else:  # 4D及以上维度
            padded[i, ..., :s.shape[-1]] = s

    return padded, lengths


def collate_finetune_fn(batch):
    """Collate function for :class:`FinetuneDataset`, supporting multi-speaker data.

    Steps performed:
    1. Right-pad ``mel_spec``, ``mel_codes``, ``text_ids``, and ``condition`` along the time dimension so they share a common length
       within the batch.
    2. Return the padded tensors **and** the original sequence lengths so the model or loss function can apply masking.

    Args:
        batch (List[Tuple[Tensor, Tensor, Tensor, Tensor, str]]): A list of samples yielded by
            :meth:`FinetuneDataset.__getitem__`.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, List[str], LongTensor, LongTensor, LongTensor]:
            ``mel_specs`` (B, D, T_max), ``mel_codes`` (B, T_max_codes), ``text_ids`` (B, T_max_text),
            ``conditions`` (B, C, T_max_condition), ``speaker_ids`` (List[str]),
            followed by their respective length tensors ``mel_lengths`` / ``codes_lengths`` / ``text_lengths``.
    """

    mel_specs, mel_codes, text_ids, conditions, speaker_ids = zip(*batch)

    # Remove the extra batch dimension left by data preprocessing for mel_specs and mel_codes (1, ...) -> (...)
    mel_specs = [spec.squeeze(0) if spec.dim() >= 3 and spec.size(0) == 1 else spec for spec in mel_specs]
    mel_codes = [codes.squeeze(0) if codes.dim() >= 2 and codes.size(0) == 1 else codes for codes in mel_codes]

    # Remove the extra batch dimension added to text_ids inside the dataset (1, L) -> (L)
    text_ids = [ids.squeeze(0) if ids.dim() == 2 and ids.size(0) == 1 else ids for ids in text_ids]
    
    # conditions mast not None
    assert all(cond is not None for cond in conditions), "conditions must not be None"
    conditions = [cond.squeeze(0) if cond.dim() >= 3 and cond.size(0) == 1 else cond for cond in conditions]


    # Pad
    mel_specs_padded, mel_lengths = _pad_sequence(list(mel_specs), pad_value=0.0, dim=-1)
    mel_codes_padded, codes_lengths = _pad_sequence(list(mel_codes), pad_value=0, dim=-1)
    text_ids_padded, text_lengths = _pad_sequence(list(text_ids), pad_value=0, dim=-1)
    
    # Stack conditions directly since they all have the same shape [32, 1280]
    conditions_padded = torch.stack(conditions, dim=0)  # [B, 32, 1280]

    return (
        mel_specs_padded, # [B, 100, T_max]
        mel_codes_padded, # [B, T_max_codes]
        text_ids_padded, # [B, T_max]
        conditions_padded, # [B, 32, 1280]
        list(speaker_ids), # [B]
        mel_lengths,
        codes_lengths,
        text_lengths,
    )


def load_finetune_datasets(config: DictConfig, bpe_path: str) -> Tuple[Dataset, Dataset]:
    """Utility helper to load the train/validation datasets for multi-speaker training.

    Args:
        config (DictConfig): Global configuration.
        bpe_path (str): Path to the BPE model file.

    Returns:
        Tuple[Dataset, Dataset]: ``(train_dataset, validation_dataset)``.
    """
    # 读取说话人信息
    speaker_info_path = os.path.join(config.train.data_path, "speaker_info.json")
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        speaker_info_list = json.load(f)
    
    train_manifest_files = []
    valid_manifest_files = []
    speaker_ids = []
    
    # 收集所有说话人的训练和验证数据文件
    for speaker_info in speaker_info_list:
        speaker_id = speaker_info['speaker']

        train_file = speaker_info['train_jsonl']
        valid_file = speaker_info['valid_jsonl']
        
        if os.path.exists(train_file) and os.path.exists(valid_file):
            train_manifest_files.append(train_file)
            valid_manifest_files.append(valid_file)
            speaker_ids.append(speaker_id)
            logger.info(f"Added speaker {speaker_id} with data from {train_file} and {valid_file}")
        else:
            logger.warning(f"Missing metadata files for speaker {speaker_id}")
    
    # 创建数据集
    train_dataset = FinetuneDataset(train_manifest_files, bpe_path, speaker_ids, config)
    valid_dataset = FinetuneDataset(valid_manifest_files, bpe_path, speaker_ids, config)
    
    return train_dataset, valid_dataset


def load_speaker_conditions(config: DictConfig) -> dict:
    """加载所有说话人的mean_condition。
    
    Args:
        config (DictConfig): Global configuration.
        
    Returns:
        dict: Dictionary mapping speaker_id to mean_condition tensor.
    """
    speaker_info_path = os.path.join(config.train.data_path, "speaker_info.json")
    with open(speaker_info_path, 'r', encoding='utf-8') as f:
        speaker_info_list = json.load(f)
    
    speaker_conditions = {}
    for speaker_info in speaker_info_list:
        speaker_id = speaker_info['speaker']
        medoid_path = speaker_info['medoid_condition']
        
        if os.path.exists(medoid_path):
            condition = np.load(medoid_path)
            speaker_conditions[speaker_id] = torch.from_numpy(condition).float()
            logger.info(f"Loaded mean condition for speaker {speaker_id}: shape {condition.shape}")
        else:
            raise ValueError(f"Missing medoid_condition.npy for speaker {speaker_id}")
    
    return speaker_conditions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load finetune datasets.")
    parser.add_argument("--config", type=str, default="finetune_models/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--bpe_model", type=str, default="finetune_models/bpe.model", help="Path to the SentencePiece model.")

    args = parser.parse_args()

    # Load config file
    config = OmegaConf.load(args.config)

    # Load datasets (现在直接传递BPE路径)
    train_dataset, valid_dataset = load_finetune_datasets(config, args.bpe_model)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")

    loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_finetune_fn)
    sample_batch = next(iter(loader))

    mel_specs, mel_codes, text_ids, conditions, speaker_ids, mel_lens, code_lens, text_lens = sample_batch
    logger.info(f"Sample batch shapes -- mel_specs: {tuple(mel_specs.shape)}, mel_codes: {tuple(mel_codes.shape)}, text_ids: {tuple(text_ids.shape)}")
    if conditions is not None:
        logger.info(f"Conditions shape: {tuple(conditions.shape)}")
    logger.info(f"Speaker IDs: {speaker_ids}")
    logger.info(f"Lengths -- mel: {mel_lens.tolist()}, codes: {code_lens.tolist()}, text: {text_lens.tolist()}")