#!/usr/bin/env python3
"""
简单的 SentencePiece tokenizer 测试脚本
"""

import argparse
import os

import sentencepiece as smp


def main():
    parser = argparse.ArgumentParser(description="测试 SentencePiece tokenizer")
    parser.add_argument(
        "--bpe-model",
        type=str,
        default="finetune_models/bpe.model",
        help="BPE 模型文件路径"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    if not os.path.exists(args.bpe_model):
        print(f"❌ BPE 模型文件不存在: {args.bpe_model}")
        return
    
    bpe_model = smp.SentencePieceProcessor()
    bpe_model.Load(args.bpe_model)
    
    print(f"✅ 已加载 BPE 模型: {args.bpe_model}")
    print(f"📊 词汇表大小: {bpe_model.GetPieceSize()}")

    # 输出词汇表
    #for i in range(bpe_model.GetPieceSize()):
    #    print(f"{i}: {bpe_model.IdToPiece(i)}")

    text = "HELLO"
    tokens = bpe_model.Encode(text, out_type=int)
    print(f"📝 输入文本: '{text}'")
    print(f"🔢 Token IDs: {tokens}")
    for i in tokens:
        print(f"{i}: {bpe_model.IdToPiece(i)}")


if __name__ == "__main__":
    main()
