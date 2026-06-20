# -*- coding: utf-8 -*-
"""
提取 canonical transcript 的内含子坐标，与重复元件取交集，汇总到 transcript 级别。
支持人类 Alu（hg38）和小鼠 SINE（mm39）分析。

用法：
  python intronic_repeats_per_transcript.py human
  python intronic_repeats_per_transcript.py mouse

输出：intronic_alu_per_transcript.csv / intronic_sine_per_transcript.csv
列：transcript_id, gene_id, intronic_repeat_count

修改日期：2026-05-31
"""

import pandas as pd
import numpy as np
import re
import sys

# ==================== 参数设置 ====================
# 修改日期：2026-05-31
if len(sys.argv) < 2:
    print("用法: python intronic_repeats_per_transcript.py [human|mouse]")
    sys.exit(1)

mode = sys.argv[1].lower()

if mode == 'human':
    BED_FILE = "alu_hg38.bed"
    GTF_FILE = "gencode.v49.primary_assembly.annotation.gtf"
    OUTPUT_CSV = "intronic_alu_per_transcript.csv"
    REPEAT_LABEL = "Alu"
elif mode == 'mouse':
    BED_FILE = "sine_mm39.bed"
    GTF_FILE = "gencode.vM38.primary_assembly.annotation.gtf"
    OUTPUT_CSV = "intronic_sine_per_transcript.csv"
    REPEAT_LABEL = "SINE"
else:
    print(f"未知模式: {mode}，请用 human 或 mouse")
    sys.exit(1)

print(f"模式: {mode}")
print(f"BED: {BED_FILE}")
print(f"GTF: {GTF_FILE}")

# ==================== 1. 读取重复元件 BED ====================
# 修改日期：2026-05-31
bed_df = pd.read_csv(
    BED_FILE,
    sep='\t',
    header=None,
    names=['chrom', 'start', 'end', 'repeat_id', 'score', 'strand']
)
bed_df['start'] = bed_df['start'] + 1  # BED 0-based → 1-based
print(f"读取 {REPEAT_LABEL} 数量: {len(bed_df)}")

# ==================== 2. 解析 GTF，提取 canonical transcript 的外显子 ====================
# 修改日期：2026-05-31
print("正在解析 GTF...")

canonical_tx_ids = set()
exon_data = []  # [(tx_id, chrom, start, end)]

with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue

        feature = parts[2]
        attr_str = parts[8]

        # 解析属性
        attrs = {}
        for m in re.finditer(r'(\w+)\s+"([^"]*)"', attr_str):
            attrs[m.group(1)] = m.group(2)

        tx_id = attrs.get('transcript_id', '').split('.')[0]

        if feature == 'transcript':
            if mode == 'human':
                is_canonical = any(tag in ('MANE_Select', 'basic') for tag in re.findall(r'tag\s+"([^"]*)"', attr_str))
            else:
                is_canonical = any(tag == 'basic' for tag in re.findall(r'tag\s+"([^"]*)"', attr_str))
            if is_canonical:
                canonical_tx_ids.add(tx_id)

        elif feature == 'exon' and tx_id in canonical_tx_ids:
            exon_data.append((tx_id, parts[0], int(parts[3]), int(parts[4])))

print(f"Canonical transcript: {len(canonical_tx_ids)}")
print(f"外显子记录: {len(exon_data)}")

# ==================== 3. 从外显子推导内含子 ====================
# 修改日期：2026-05-31
print("正在推导内含子坐标...")

# 按 transcript 分组外显子，排序，取间隙
tx_exons = {}
for tx_id, chrom, s, e in exon_data:
    tx_exons.setdefault(tx_id, []).append((chrom, s, e))

intron_rows = []
for tx_id, exons in tx_exons.items():
    exons.sort(key=lambda x: x[1])  # 按 start 排序
    chrom = exons[0][0]
    for i in range(len(exons) - 1):
        intron_start = exons[i][2] + 1
        intron_end = exons[i + 1][1] - 1
        if intron_start <= intron_end:
            intron_rows.append({
                'transcript_id': tx_id,
                'chrom': chrom,
                'start': intron_start,
                'end': intron_end,
            })

intron_df = pd.DataFrame(intron_rows)
print(f"内含子总数: {len(intron_df)}")
print(f"有内含子的 transcript: {intron_df['transcript_id'].nunique()}")

# ==================== 4. 内含子与重复元件取交集 ====================
# 修改日期：2026-05-31
print(f"正在计算 {REPEAT_LABEL} 与内含子的交集...")

intronic_repeat_counts = {}  # transcript_id -> count

# 按染色体分组加速
for chrom, intron_chrom in intron_df.groupby('chrom'):
    bed_chrom = bed_df[bed_df['chrom'] == chrom]
    if bed_chrom.empty:
        continue

    for _, intron_row in intron_chrom.iterrows():
        tx_id = intron_row['transcript_id']
        intron_start = intron_row['start']
        intron_end = intron_row['end']

        # 找到与该内含子重叠的重复元件
        overlaps = bed_chrom[
            (bed_chrom['start'] <= intron_end) &
            (bed_chrom['end'] >= intron_start)
        ]

        # 计入该 transcript 的计数（去重，防止一个 Alu 落在多个内含子中被重复计数）
        repeat_ids = set(overlaps['repeat_id'])
        if repeat_ids:
            intronic_repeat_counts[tx_id] = intronic_repeat_counts.get(tx_id, 0) + len(repeat_ids)

# 同时收集 gene_id
tx_to_gene = {}
with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'transcript':
            continue
        attr_str = parts[8]
        m = re.search(r'transcript_id \"([^\"]+)\"', attr_str)
        m2 = re.search(r'gene_id \"([^\"]+)\"', attr_str)
        if m and m2:
            tx = m.group(1).split('.')[0]
            if tx in intronic_repeat_counts:
                tx_to_gene[tx] = m2.group(1).split('.')[0]

# ==================== 5. 输出 ====================
# 修改日期：2026-05-31
output_rows = []
for tx_id, count in intronic_repeat_counts.items():
    output_rows.append({
        'transcript_id': tx_id,
        'gene_id': tx_to_gene.get(tx_id, ''),
        'intronic_repeat_count': count,
    })

output_df = pd.DataFrame(output_rows)
output_df = output_df.sort_values('intronic_repeat_count', ascending=False).reset_index(drop=True)
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n处理完成！输出: {OUTPUT_CSV}")
print(f"总记录数: {len(output_df)}")
print(f"内含子含 {REPEAT_LABEL} 的 transcript 数: {len(output_df)}")
print(f"平均每 transcript {REPEAT_LABEL} 数: {output_df['intronic_repeat_count'].mean():.2f}")
print(f"最大 {REPEAT_LABEL} 数: {output_df['intronic_repeat_count'].max()}")