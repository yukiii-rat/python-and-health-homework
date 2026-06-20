# -*- coding: utf-8 -*-
"""
构建基因级别 SINE 特征矩阵（小鼠 mm39）
输入：sine_exonic_mapping.csv, intronic_sine_per_transcript.csv, gencode.vM38 GTF
输出：feature_matrix_mouse.csv

修改日期：2026-05-31
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict

# ==================== 参数 ====================
# 修改日期：2026-05-31
EXONIC_FILE = "sine_exonic_mapping.csv"
INTRONIC_FILE = "intronic_sine_per_transcript.csv"
GTF_FILE = "gencode.vM38.primary_assembly.annotation.gtf"
OUTPUT_FILE = "feature_matrix_mouse.csv"
ID_COL = "sine_id"  # 外显子文件中重复元件 ID 的列名

# ==================== 1. 读取外显子 SINE 数据 ====================
# 修改日期：2026-05-31
exonic_df = pd.read_csv(EXONIC_FILE)
print(f"外显子 SINE 记录: {len(exonic_df)}")

# ==================== 2. 按 gene_id 聚合外显子 SINE 特征 ====================
# 修改日期：2026-05-31
print("正在聚合外显子 SINE 特征...")

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

# gene_name 映射
gene_name_map = exonic_df[['gene_id', 'gene_name']].drop_duplicates().set_index('gene_id')['gene_name'].to_dict()

gene_features = pd.DataFrame(index=region_counts.index)
gene_features['gene_name'] = gene_features.index.map(gene_name_map)
gene_features['sine_exonic_total'] = exonic_df.groupby('gene_id').size()
gene_features['sine_5utr'] = region_counts.get('5UTR', 0).values
gene_features['sine_cds'] = region_counts.get('CDS', 0).values
gene_features['sine_3utr'] = region_counts.get('3UTR', 0).values

# 方向
gene_features['sine_sense'] = ori_counts.get('sense', 0).values
gene_features['sine_antisense'] = ori_counts.get('antisense', 0).values

# ==================== 3. SINE 亚家族分类 ====================
# 修改日期：2026-05-31
print("正在分类 SINE 亚家族...")

def classify_sine_subfamily(sine_id):
    """根据 SINE ID 判断亚家族"""
    if sine_id.startswith('B1') or sine_id in ('ID_B1',):
        return 'B1'
    elif sine_id.startswith('B2'):
        return 'B2'
    elif sine_id.startswith('B3') or sine_id.startswith('B4'):
        return 'B3_B4'
    elif sine_id == 'ID' or sine_id.startswith('ID') and sine_id not in ('ID_B1',):
        return 'ID'
    elif sine_id.startswith('MIR') or sine_id.startswith('Mam') or sine_id.startswith('Amn'):
        return 'MIR_related'
    elif sine_id.startswith('PB1'):
        return 'PB1'
    elif sine_id.startswith('LFSINE') or sine_id.startswith('RSINE'):
        return 'Other'
    else:
        return 'Other'

exonic_df['subfamily'] = exonic_df[ID_COL].apply(classify_sine_subfamily)
subfamily_counts = exonic_df.groupby(['gene_id', 'subfamily']).size().unstack(fill_value=0)
for sf in ['B1', 'B2', 'B3_B4', 'ID', 'MIR_related', 'PB1', 'Other']:
    gene_features[f'sine_{sf}_count'] = subfamily_counts.get(sf, 0).values

# ==================== 4. 读取内含子 SINE 数据 ====================
# 修改日期：2026-05-31
print("正在聚合内含子 SINE 特征...")
intronic_df = pd.read_csv(INTRONIC_FILE)

intronic_gene = intronic_df.groupby('gene_id')['intronic_repeat_count'].sum()
gene_features['intronic_sine_count'] = gene_features.index.map(
    lambda g: intronic_gene.get(g, 0)
)

# ==================== 5. 从 GTF 获取基因长度和外显子长度 ====================
# 修改日期：2026-05-31
print("正在从 GTF 提取基因长度和外显子长度...")

gene_lengths = {}
canonical_tx_ids = set()
tx_to_gene = {}

# 第一遍：收集 canonical transcript（小鼠仅 basic tag）
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
            is_canonical = any(tag == 'basic' for tag in re.findall(r'tag\s+"([^"]*)"', attr_str))
            if is_canonical:
                canonical_tx_ids.add(tx_id)
                gid = attrs.get('gene_id', '').split('.')[0]
                tx_to_gene[tx_id] = gid

# 第二遍：合并外显子坐标，去重计算外显子总长
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

gene_features['sine_density_per_kb'] = (
    gene_features['sine_exonic_total'] / (gene_features['exonic_length'] / 1000)
).replace([np.inf, -np.inf], 0).fillna(0)

gene_features['sine_3utr_ratio'] = (
    gene_features['sine_3utr'] / gene_features['sine_exonic_total']
).replace([np.inf, -np.inf], 0).fillna(0)

gene_features['sine_antisense_ratio'] = (
    gene_features['sine_antisense'] / gene_features['sine_exonic_total']
).replace([np.inf, -np.inf], 0).fillna(0)

gene_features['has_antisense_sine'] = (gene_features['sine_antisense'] > 0).astype(int)
gene_features['has_3utr_sine'] = (gene_features['sine_3utr'] > 0).astype(int)

# ==================== 7. 排序并输出 ====================
# 修改日期：2026-05-31
subfamily_cols = sorted([c for c in gene_features.columns if c.startswith('sine_') and c.endswith('_count') and c not in (
    'sine_exonic_total', 'sine_5utr', 'sine_cds', 'sine_3utr', 'sine_sense', 'sine_antisense',
    'sine_density_per_kb', 'sine_3utr_ratio', 'sine_antisense_ratio'
)])

final_cols = [
    'gene_name',
    'sine_exonic_total', 'sine_5utr', 'sine_cds', 'sine_3utr',
    'sine_sense', 'sine_antisense',
    'sine_density_per_kb',
    'sine_3utr_ratio', 'sine_antisense_ratio',
    'has_antisense_sine', 'has_3utr_sine',
] + subfamily_cols + [
    'intronic_sine_count',
    'gene_length', 'exonic_length',
]

output_df = gene_features[final_cols].copy()
output_df = output_df.sort_values('sine_exonic_total', ascending=False).reset_index()
output_df = output_df.rename(columns={'index': 'gene_id'})

output_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n处理完成！")
print(f"输出: {OUTPUT_FILE}")
print(f"基因数: {len(output_df)}")
print(f"特征数: {len(output_df.columns)}")
print(f"列: {list(output_df.columns)}")
print(f"\n前 5 行:")
print(output_df.head(5).to_string(index=False))