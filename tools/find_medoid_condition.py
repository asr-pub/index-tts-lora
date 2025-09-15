#!/usr/bin/env python3
"""
找出 condition latent 分布中心的样本（medoid）
即距离所有其他样本距离之和最小的样本

用法示例：
    python tools/find_medoid_condition.py \
        --input M00100_condition/output.jsonl \
        --output medoid_info.json
"""
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# --- 解析命令行 ---
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='输入的 jsonl 文件，包含 condition 字段')
parser.add_argument('--output', default='medoid_info.json', help='输出 medoid 信息文件')
parser.add_argument('--distance_metric', default='euclidean', choices=['euclidean', 'cosine'], help='距离度量方式')
args = parser.parse_args()

# 转换为绝对路径
args.input = os.path.abspath(args.input)
args.output = os.path.abspath(args.output)

print(f"读取输入文件: {args.input}")
print(f"距离度量: {args.distance_metric}")

# --- 读取所有 condition latent ---
conditions = []
condition_infos = []  # 保存每个样本的信息
count = 0

with open(args.input, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        item = json.loads(line)
        if 'condition' not in item:
            print(f"警告: 第 {line_idx+1} 行缺少 condition 字段，跳过")
            continue
        
        condition_path = item['condition']
        if not os.path.exists(condition_path):
            print(f"警告: condition 文件不存在: {condition_path}，跳过")
            continue
        
        # 加载 condition latent
        cond = np.load(condition_path)  # (32, dim)
        conditions.append(cond)
        condition_infos.append({
            'index': count,
            'line_number': line_idx + 1,
            'condition_path': condition_path,
            'original_item': item
        })
        count += 1
        
        if count % 100 == 0:
            print(f"已读取 {count} 个样本")

if len(conditions) == 0:
    raise RuntimeError('没有找到有效的 condition 文件')

print(f"总共读取 {len(conditions)} 个 condition latent")

# 将所有 condition 堆叠成一个数组
conditions_array = np.stack(conditions, axis=0)  # (N, 32, dim)
print(f"Conditions 数组形状: {conditions_array.shape}")

# 将数据展平以便计算距离
N, H, W = conditions_array.shape
data_flat = conditions_array.reshape(N, H * W)  # (N, 32*dim)

# --- 计算距离矩阵 ---
print("计算距离矩阵...")

def compute_distance_matrix(data, metric='euclidean'):
    """计算距离矩阵"""
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    
    for i in tqdm(range(n), desc="计算距离"):
        for j in range(i+1, n):
            if metric == 'euclidean':
                dist = np.linalg.norm(data[i] - data[j])
            elif metric == 'cosine':
                # 余弦距离 = 1 - 余弦相似度
                cos_sim = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
                dist = 1 - cos_sim
            else:
                raise ValueError(f"不支持的距离度量: {metric}")
            
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # 对称矩阵
    
    return dist_matrix

dist_matrix = compute_distance_matrix(data_flat, args.distance_metric)

# --- 找出 medoid ---
print("寻找 medoid...")

# 计算每个样本到所有其他样本的距离之和
dist_sums = np.sum(dist_matrix, axis=1)

# 找出距离之和最小的样本索引
medoid_idx = np.argmin(dist_sums)
medoid_info = condition_infos[medoid_idx]
medoid_condition = conditions[medoid_idx]

print(f"找到 medoid: 索引 {medoid_idx}")
print(f"Medoid 信息: {medoid_info}")
print(f"Medoid 到所有样本的距离之和: {dist_sums[medoid_idx]:.6f}")

# --- 计算一些统计信息 ---
stats = {
    'total_samples': len(conditions),
    'medoid_index': int(medoid_idx),
    'medoid_distance_sum': float(dist_sums[medoid_idx]),
    'distance_metric': args.distance_metric,
    'distance_stats': {
        'mean': float(np.mean(dist_sums)),
        'std': float(np.std(dist_sums)),
        'min': float(np.min(dist_sums)),
        'max': float(np.max(dist_sums))
    }
}

# --- 保存结果 ---
result = {
    'medoid_info': medoid_info,
    'statistics': stats,
    'medoid_condition_shape': medoid_condition.shape
}

# 保存 medoid 信息
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 保存 medoid condition
medoid_condition_path = os.path.splitext(args.output)[0] + '_medoid_condition.npy'
np.save(medoid_condition_path, medoid_condition)

# 保存距离矩阵（可选，用于进一步分析）
dist_matrix_path = os.path.splitext(args.output)[0] + '_distance_matrix.npy'
np.save(dist_matrix_path, dist_matrix)

print(f"\n结果保存完成:")
print(f"Medoid 信息: {args.output}")
print(f"Medoid condition: {medoid_condition_path}")
print(f"距离矩阵: {dist_matrix_path}")
print(f"\nMedoid 样本详情:")
print(f"  - 原始行号: {medoid_info['line_number']}")
print(f"  - Condition 文件: {medoid_info['condition_path']}")
if 'audio' in medoid_info['original_item']:
    print(f"  - 音频文件: {medoid_info['original_item']['audio']}")
if 'text' in medoid_info['original_item']:
    print(f"  - 文本: {medoid_info['original_item']['text']}")