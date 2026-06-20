# -*- coding: utf-8 -*-
"""
构建基因级别 Alu 特征矩阵
输入：alu_exonic_mapping.csv, intronic_alu_per_transcript.csv, gencode.v49 GTF
输出：feature_matrix.csv

修改日期：2026-05-31
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict

# ==================== 参数 ====================
# 修改日期：2026-05-31
EXONIC_FILE = "alu_exonic_mapping.csv"
INTRONIC_FILE = "intronic_alu_per_transcript.csv"
GTF_FILE = "gencode.v49.primary_assembly.annotation.gtf"
OUTPUT_FILE = "feature_matrix.csv"

# ==================== 1. 读取外显子 Alu 数据 ====================
# 修改日期：2026-05-31
exonic_df = pd.read_csv(EXONIC_FILE)
print(f"外显子 Alu 记录: {len(exonic_df)}")

# ==================== 2. 按 gene_id 聚合外显子 Alu 特征 ====================
# 修改日期：2026-05-31
print("正在聚合外显子 Alu 特征...")

# 按 region 计数
region_counts = exonic_df.groupby(['gene_id', 'region']).size().unstack(fill_value=0)
for col in ['5UTR', 'CDS', '3UTR']:
    if col not in region_counts:
        region_counts[col] = 0

# 按 orientation 计数
ori_counts = exonic_df.groupby(['gene_id', 'orientation']).size().unstack(fill_value=0)
for col in ['sense', 'antisense']:
    if col not in ori_counts:
        ori_counts[col] = 0

# 提取 gene_name 映射
gene_name_map = exonic_df[['gene_id', 'gene_name']].drop_duplicates().set_index('gene_id')['gene_name'].to_dict()

# 合并特征
gene_features = pd.DataFrame(index=region_counts.index)
gene_features['gene_name'] = gene_features.index.map(gene_name_map)
gene_features['alu_exonic_total'] = exonic_df.groupby('gene_id').size()
gene_features['alu_5utr'] = region_counts.get('5UTR', 0).values
gene_features['alu_cds'] = region_counts.get('CDS', 0).values
gene_features['alu_3utr'] = region_counts.get('3UTR', 0).values

# 方向
gene_features['alu_sense'] = ori_counts.get('sense', 0).values
gene_features['alu_antisense'] = ori_counts.get('antisense', 0).values

# ==================== 3. Alu 亚家族分类 ====================
# 修改日期：2026-05-31
print("正在分类 Alu 亚家族...")

def classify_alu_subfamily(alu_id):
    """根据 Alu ID 名称判断亚家族"""
    if alu_id.startswith('AluJ'):
        return 'aluJ'
    elif alu_id.startswith('AluS'):
        return 'aluS'
    elif alu_id.startswith('AluY'):
        return 'aluY'
    elif alu_id == 'Alu' or alu_id.startswith('Alu'):
        return 'alu_other'
    return 'alu_other'

exonic_df['subfamily'] = exonic_df['alu_id'].apply(classify_alu_subfamily)
subfamily_counts = exonic_df.groupby(['gene_id', 'subfamily']).size().unstack(fill_value=0)
for col in ['aluJ', 'aluS', 'aluY', 'alu_other']:
    gene_features[col + '_count'] = subfamily_counts.get(col, 0).values

# ==================== 4. 读取内含子 Alu 数据 ====================
# 修改日期：2026-05-31
print("正在聚合内含子 Alu 特征...")
intronic_df = pd.read_csv(INTRONIC_FILE)

# 按 gene_id 汇总内含子 Alu 数（一个 gene 可能有多个 transcript）
intronic_gene = intronic_df.groupby('gene_id')['intronic_repeat_count'].sum()
gene_features['intronic_alu_count'] = gene_features.index.map(
    lambda g: intronic_gene.get(g, 0)
)

# ==================== 5. 从 GTF 获取基因长度和外显子长度 ====================
# 修改日期：2026-05-31
print("正在从 GTF 提取基因长度和外显子长度...")

gene_lengths = {}   # gene_id -> (gene_start, gene_end)
# 外显子长度：按 gene 汇总外显子总长度（仅 canonical transcript）
canonical_tx_ids = set()
tx_to_gene = {}     # transcript_id -> gene_id

# 第一遍：收集 canonical transcript
with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        feature = parts[2]
        attr_str = parts[8]
        attrs = dict(re.findall(r'(\w+)\s+"([^"]*)"', attr_str))

        if feature == 'gene':
            gid = attrs.get('gene_id', '').split('.')[0]
            gene_lengths[gid] = (int(parts[3]), int(parts[4]))

        elif feature == 'transcript':
            tx_id = attrs.get('transcript_id', '').split('.')[0]
            is_canonical = any(tag in ('MANE_Select', 'basic') for tag in re.findall(r'tag\s+"([^"]*)"', attr_str))
            if is_canonical:
                canonical_tx_ids.add(tx_id)
                gid = attrs.get('gene_id', '').split('.')[0]
                tx_to_gene[tx_id] = gid

# 第二遍：计算每个 canonical transcript 的外显子总长，再聚合到 gene
gene_exonic_len = defaultdict(int)
with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'exon':
            continue
        attr_str = parts[8]
        m = re.search(r'transcript_id "([^"]+)"', attr_str)
        if not m:
            continue
        tx_id = m.group(1).split('.')[0]
        if tx_id not in canonical_tx_ids:
            continue
        gid = tx_to_gene.get(tx_id, '')
        if gid:
            exon_len = int(parts[4]) - int(parts[3]) + 1
            gene_exonic_len[gid] += exon_len

# 计算 gene_length 和外显子长度（对有多 transcript 的 gene 取最大 transcript 的外显子长度）
# 实际上对于 exonic_length，应该用非冗余的 exon 坐标，但这里简化为所有 canonical transcript 外显子总长
# 因为选择 canonical transcript 是每个 gene_id 取一个，所以用 tx_to_gene 里的值
# 但实际可能有多个 canonical transcript 对应同一个 gene。最好用外显子合并去重

# 更精确：读取所有 canonical transcript 的外显子坐标，合并去重
gene_exon_intervals = defaultdict(set)
with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'exon':
            continue
        attr_str = parts[8]
        m = re.search(r'transcript_id "([^"]+)"', attr_str)
        if not m:
            continue
        tx_id = m.group(1).split('.')[0]
        if tx_id not in canonical_tx_ids:
            continue
        gid = tx_to_gene.get(tx_id, '')
        if gid:
            gene_exon_intervals[gid].add((int(parts[3]), int(parts[4])))

# 合并重叠的外显子区间并计算总长
gene_exonic_length = {}
for gid, intervals in gene_exon_intervals.items():
    sorted_iv = sorted(intervals)
    merged = []
    for s, e in sorted_iv:
        if not merged:
            merged.append([s, e])
        else:
            if s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
    total = sum(e - s + 1 for s, e in merged)
    gene_exonic_length[gid] = total

gene_features['gene_length'] = gene_features.index.map(
    lambda g: gene_lengths[g][1] - gene_lengths[g][0] + 1 if g in gene_lengths else 0
)
gene_features['exonic_length'] = gene_features.index.map(
    lambda g: gene_exonic_length.get(g, 0)
)

# ==================== 6. 衍生特征 ====================
# 修改日期：2026-05-31
print("正在计算衍生特征...")

gene_features['alu_density_per_kb'] = (
    gene_features['alu_exonic_total'] / (gene_features['exonic_length'] / 1000)
).replace([np.inf, -np.inf], 0).fillna(0)

gene_features['alu_3utr_ratio'] = (
    gene_features['alu_3utr'] / gene_features['alu_exonic_total']
).replace([np.inf, -np.inf], 0).fillna(0)

gene_features['alu_antisense_ratio'] = (
    gene_features['alu_antisense'] / gene_features['alu_exonic_total']
).replace([np.inf, -np.inf], 0).fillna(0)

gene_features['has_antisense_alu'] = (gene_features['alu_antisense'] > 0).astype(int)
gene_features['has_3utr_alu'] = (gene_features['alu_3utr'] > 0).astype(int)

# ==================== 7. 排序并输出 ====================
# 修改日期：2026-05-31
# 按 alu_exonic_total 降序排列
final_cols = [
    'gene_name',
    'alu_exonic_total', 'alu_5utr', 'alu_cds', 'alu_3utr',
    'alu_sense', 'alu_antisense',
    'alu_density_per_kb',
    'alu_3utr_ratio', 'alu_antisense_ratio',
    'has_antisense_alu', 'has_3utr_alu',
    'aluJ_count', 'aluS_count', 'aluY_count',
    'intronic_alu_count',
    'gene_length', 'exonic_length',
]

# 补齐缺失列
for col in ['aluJ_count', 'aluS_count', 'aluY_count', 'alu_other_count']:
    if col not in gene_features.columns:
        gene_features[col] = 0

output_df = gene_features[final_cols].copy()
output_df = output_df.sort_values('alu_exonic_total', ascending=False).reset_index()
output_df = output_df.rename(columns={'index': 'gene_id'})

output_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n处理完成！")
print(f"输出: {OUTPUT_FILE}")
print(f"基因数: {len(output_df)}")
print(f"特征数: {len(output_df.columns)}")
print(f"列: {list(output_df.columns)}")
print(f"\n前 5 行:")
print(output_df.head(5).to_string(index=False))