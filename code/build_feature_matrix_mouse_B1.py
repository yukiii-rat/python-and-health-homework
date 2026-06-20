# -*- coding: utf-8 -*-
"""
构建 B1-only 小鼠特征矩阵
从 sine_exonic_mapping.csv 过滤 B1 亚族，重建全量特征
输出 feature_matrix_mouse_B1.csv
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict

EXONIC_FILE = "output1/sine_exonic_mapping.csv"
INTRONIC_FILE = "output1/intronic_sine_per_transcript.csv"
GTF_FILE = "data/gencode.vM38.primary_assembly.annotation.gtf"
OUTPUT_FILE = "output1/feature_matrix_mouse_B1.csv"

# B1 亚族 ID 列表
B1_IDS = ['B1F', 'B1F1', 'B1F2', 'B1_Mm', 'B1_Mur1', 'B1_Mur2', 'B1_Mur3',
          'B1_Mur4', 'B1_Mus1', 'B1_Mus2', 'ID_B1']

# 1. 读取并过滤 B1
exonic_df = pd.read_csv(EXONIC_FILE)
exonic_df = exonic_df[exonic_df['sine_id'].isin(B1_IDS)].copy()
print(f"B1 exonic 记录: {len(exonic_df)}")

# 2. 按 gene_id 聚合
region_counts = exonic_df.groupby(['gene_id', 'region']).size().unstack(fill_value=0)
for col in ['5UTR', 'CDS', '3UTR']:
    if col not in region_counts: region_counts[col] = 0

ori_counts = exonic_df.groupby(['gene_id', 'orientation']).size().unstack(fill_value=0)
for col in ['sense', 'antisense']:
    if col not in ori_counts: ori_counts[col] = 0

gene_name_map = exonic_df[['gene_id', 'gene_name']].drop_duplicates().set_index('gene_id')['gene_name'].to_dict()

gene_features = pd.DataFrame(index=region_counts.index)
gene_features['gene_name'] = gene_features.index.map(gene_name_map)
gene_features['B1_exonic_total'] = exonic_df.groupby('gene_id').size()
gene_features['B1_5utr'] = region_counts['5UTR'].values
gene_features['B1_cds'] = region_counts['CDS'].values
gene_features['B1_3utr'] = region_counts['3UTR'].values
gene_features['B1_sense'] = ori_counts['sense'].values
gene_features['B1_antisense'] = ori_counts['antisense'].values

print(f"有 B1 外显子插入的基因数: {len(gene_features)}")

# 3. 内含子数据
intronic_df = pd.read_csv(INTRONIC_FILE)
intronic_gene = intronic_df.groupby('gene_id')['intronic_repeat_count'].sum()
gene_features['intronic_B1_count'] = gene_features.index.map(lambda g: intronic_gene.get(g, 0))

# 4. 基因 / 外显子长度
gene_lengths = {}
canonical_tx_ids = set()
tx_to_gene = {}

with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split('\t')
        if len(parts) < 9: continue
        feature = parts[2]; attr_str = parts[8]
        attrs = dict(re.findall(r'(\w+)\s+"([^"]*)"', attr_str))
        if feature == 'gene':
            gid = attrs.get('gene_id', '').split('.')[0]
            gene_lengths[gid] = (int(parts[3]), int(parts[4]))
        elif feature == 'transcript':
            tx_id = attrs.get('transcript_id', '').split('.')[0]
            if any(tag == 'basic' for tag in re.findall(r'tag\s+"([^"]*)"', attr_str)):
                canonical_tx_ids.add(tx_id)
                tx_to_gene[tx_id] = attrs.get('gene_id', '').split('.')[0]

gene_exon_intervals = defaultdict(set)
with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'exon': continue
        m = re.search(r'transcript_id "([^"]+)"', parts[8])
        if not m: continue
        tx_id = m.group(1).split('.')[0]
        if tx_id not in canonical_tx_ids: continue
        gid = tx_to_gene.get(tx_id, '')
        if gid: gene_exon_intervals[gid].add((int(parts[3]), int(parts[4])))

gene_exonic_length = {}
for gid, ivs in gene_exon_intervals.items():
    merged = []
    for s, e in sorted(ivs):
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    gene_exonic_length[gid] = sum(e - s + 1 for s, e in merged)

gene_features['gene_length'] = gene_features.index.map(
    lambda g: gene_lengths[g][1] - gene_lengths[g][0] + 1 if g in gene_lengths else 0)
gene_features['exonic_length'] = gene_features.index.map(lambda g: gene_exonic_length.get(g, 0))

# 5. 衍生特征
gene_features['B1_density_per_kb'] = (
    gene_features['B1_exonic_total'] / (gene_features['exonic_length'] / 1000)
).replace([np.inf, -np.inf], 0).fillna(0)
gene_features['B1_3utr_ratio'] = (
    gene_features['B1_3utr'] / gene_features['B1_exonic_total']
).replace([np.inf, -np.inf], 0).fillna(0)
gene_features['B1_antisense_ratio'] = (
    gene_features['B1_antisense'] / gene_features['B1_exonic_total']
).replace([np.inf, -np.inf], 0).fillna(0)
gene_features['has_antisense_B1'] = (gene_features['B1_antisense'] > 0).astype(int)
gene_features['has_3utr_B1'] = (gene_features['B1_3utr'] > 0).astype(int)

# 6. 输出
final_cols = [
    'gene_name', 'B1_exonic_total', 'B1_5utr', 'B1_cds', 'B1_3utr',
    'B1_sense', 'B1_antisense', 'B1_density_per_kb',
    'B1_3utr_ratio', 'B1_antisense_ratio',
    'has_antisense_B1', 'has_3utr_B1',
    'intronic_B1_count', 'gene_length', 'exonic_length'
]
output_df = gene_features[final_cols].copy()
output_df = output_df.sort_values('B1_exonic_total', ascending=False).reset_index().rename(columns={'index': 'gene_id'})
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n输出: {OUTPUT_FILE}")
print(f"基因数: {len(output_df)}, 特征数: {len(output_df.columns)}")
print(f"列: {list(output_df.columns)}")