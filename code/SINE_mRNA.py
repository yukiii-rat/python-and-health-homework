# -*- coding: utf-8 -*-
"""
SINE 元件与外显子区域交集分析（小鼠 mm39）
输入：sine_mm39.bed, gencode.vM38.primary_assembly.annotation.gtf
输出：sine_exonic_mapping.csv

处理流程：
1. 从 GTF 筛选 canonical transcript（tag 为 basic，每个 gene_id 只取一个）
2. 提取每个 canonical transcript 的 5'UTR、CDS、3'UTR 坐标
3. 将 SINE BED 与外显子坐标取交集
4. 边界过滤：外显子总重叠比例 ≥ EXONIC_OVERLAP_THRESHOLD
5. SINE 跨越多区域时按 CDS > 3UTR > 5UTR 优先级分配
6. 判断方向：与 gene strand 相同 = sense，相反 = antisense

修改日期：2026-05-31
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict

# ==================== 参数设置 ====================
# 修改日期：2026-05-31
SINE_BED = "sine_mm39.bed"
GTF_FILE = "gencode.vM38.primary_assembly.annotation.gtf"
OUTPUT_CSV = "sine_exonic_mapping.csv"
EXONIC_OVERLAP_THRESHOLD = 0.8  # 外显子重叠比例阈值

# ==================== 1. 读取 SINE BED 文件 ====================
# 修改日期：2026-05-31
sine_df = pd.read_csv(
    SINE_BED,
    sep='\t',
    header=None,
    names=['chrom', 'start', 'end', 'sine_id', 'score', 'strand']
)
# BED 是 0-based 半开区间 [start, end)，转为 1-based 全闭区间 [start, end]
sine_df['start'] = sine_df['start'] + 1
sine_df['length'] = sine_df['end'] - sine_df['start'] + 1
print(f"读取 SINE 数量: {len(sine_df)}")
print(f"SINE 列: {list(sine_df.columns)}")

# ==================== 2. 读取 GTF，筛选 canonical transcript ====================
# 修改日期：2026-05-31
print("正在解析 GTF 文件...")

# 存储 canonical transcript 及其 gene 信息
canonical_transcripts = {}  # transcript_id -> {gene_id, gene_name, transcript_id, chrom, strand}
# 存储每个 transcript 的各区域坐标
transcript_regions = defaultdict(lambda: defaultdict(list))  # transcript_id -> region_type -> [(start, end), ...]

# 第一次遍历：找出 canonical transcript
canonical_tx_ids = set()
# 临时存储 gene 信息
gene_info = {}  # gene_id -> gene_name

with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue

        chrom = parts[0]
        source = parts[1]
        feature = parts[2]
        start = int(parts[3])
        end = int(parts[4])
        strand = parts[6]
        attr_str = parts[8]

        # 解析属性
        attrs = {}
        for m in re.finditer(r'(\w+)\s+"([^"]*)"', attr_str):
            attrs[m.group(1)] = m.group(2)

        if feature == 'gene':
            gid = attrs.get('gene_id', '').split('.')[0]  # 去掉版本号
            gname = attrs.get('gene_name', '')
            gene_info[gid] = gname

        elif feature == 'transcript':
            tx_id = attrs.get('transcript_id', '').split('.')[0]
            gid = attrs.get('gene_id', '').split('.')[0]
            gname = attrs.get('gene_name', '')

            # 检查是否为 canonical（小鼠无 MANE_Select，仅用 basic tag）
            is_canonical = False
            tag_match = re.findall(r'tag\s+"([^"]*)"', attr_str)
            for tag in tag_match:
                if tag == 'basic':
                    is_canonical = True
                    break

            if is_canonical:
                canonical_tx_ids.add(tx_id)
                canonical_transcripts[tx_id] = {
                    'gene_id': gid,
                    'gene_name': gname,
                    'transcript_id': tx_id,
                    'chrom': chrom,
                    'strand': strand,
                }

print(f"找到 canonical transcript 数量: {len(canonical_tx_ids)}")

# ==================== 3. 提取每个 canonical transcript 的区域坐标 ====================
# 修改日期：2026-05-31
# 第二次遍历：收集 canonical transcript 的 exon / CDS / UTR 坐标
print("正在提取 exon / CDS / UTR 坐标...")

with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue

        feature = parts[2]
        if feature not in ('exon', 'CDS', 'UTR'):
            continue

        start = int(parts[3])
        end = int(parts[4])
        attr_str = parts[8]

        attrs = {}
        for m in re.finditer(r'(\w+)\s+"([^"]*)"', attr_str):
            attrs[m.group(1)] = m.group(2)

        tx_id = attrs.get('transcript_id', '').split('.')[0]
        if tx_id not in canonical_tx_ids:
            continue

        transcript_regions[tx_id][feature].append((start, end))

print(f"有区域坐标的 transcript 数量: {len(transcript_regions)}")

# ==================== 4. 将 UTR 分类为 5'UTR / 3'UTR ====================
# 修改日期：2026-05-31
# 对于每个 transcript，根据 CDS 位置和 strand 区分 5'UTR 和 3'UTR
transcript_final_regions = defaultdict(list)  # tx_id -> [(chrom, start, end, region, strand, gene_id, gene_name)]

for tx_id, regions in transcript_regions.items():
    tx_info = canonical_transcripts[tx_id]
    strand = tx_info['strand']
    chrom = tx_info['chrom']
    gene_id = tx_info['gene_id']
    gene_name = tx_info['gene_name']

    cds_coords = regions.get('CDS', [])
    utr_coords = regions.get('UTR', [])

    # 获取 CDS 的最早和最晚位置
    if cds_coords:
        cds_min = min(s for s, e in cds_coords)
        cds_max = max(e for s, e in cds_coords)
    else:
        # 没有 CDS（非编码转录本），跳过
        continue

    # 分类 UTR
    for s, e in utr_coords:
        if strand == '+':
            if e <= cds_max:
                region = '5UTR'
            else:
                region = '3UTR'
        else:  # 负链：基因组上靠右的是 5'，靠左的是 3'
            if s >= cds_min:
                region = '5UTR'
            else:
                region = '3UTR'

        transcript_final_regions[tx_id].append((chrom, s, e, region, strand, gene_id, gene_name))

    # 添加 CDS 区域
    for s, e in cds_coords:
        transcript_final_regions[tx_id].append((chrom, s, e, 'CDS', strand, gene_id, gene_name))

print(f"构建了 {len(transcript_final_regions)} 个 transcript 的区域数据")

# ==================== 5. SINE 与区域坐标取交集 ====================
# 修改日期：2026-05-31
# 将每个 transcript 的所有区域展开为一个 DataFrame
region_rows = []
for tx_id, regions_list in transcript_final_regions.items():
    for chrom, s, e, region, strand, gid, gname in regions_list:
        region_rows.append({
            'chrom': chrom,
            'start': s,
            'end': e,
            'region': region,
            'strand': strand,
            'gene_id': gid,
            'gene_name': gname,
            'transcript_id': tx_id,
        })

region_df = pd.DataFrame(region_rows)
# 合并重叠的相邻区域（同一个 transcript 的相同 region 类型）
region_df = region_df.sort_values(['transcript_id', 'region', 'start']).reset_index(drop=True)
print(f"总区域数（合并前）: {len(region_df)}")

# 合并同一 transcript 内相邻/重叠的同类区域
def merge_intervals(group):
    sorted_coords = sorted(group)
    merged = []
    for s, e in sorted_coords:
        if not merged:
            merged.append([s, e])
        else:
            if s <= merged[-1][1] + 1:  # 相邻或重叠
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
    return [(s, e) for s, e in merged]

merged_rows = []
for (tx_id, region), grp in region_df.groupby(['transcript_id', 'region']):
    chrom = grp['chrom'].iloc[0]
    strand = grp['strand'].iloc[0]
    gene_id = grp['gene_id'].iloc[0]
    gene_name = grp['gene_name'].iloc[0]
    intervals = list(zip(grp['start'], grp['end']))
    for s, e in merge_intervals(intervals):
        merged_rows.append({
            'chrom': chrom,
            'start': s,
            'end': e,
            'region': region,
            'strand': strand,
            'gene_id': gene_id,
            'gene_name': gene_name,
            'transcript_id': tx_id,
        })

region_df = pd.DataFrame(merged_rows)
print(f"总区域数（合并后）: {len(region_df)}")

# ==================== 6. 计算 SINE 与每个区域的重叠 ====================
# 修改日期：2026-05-31
# 对每对 SINE x 区域，检查染色体是否匹配，计算重叠长度
print("正在计算 SINE 与区域的重叠...")

results = []
# 按染色体分组加速
for chrom, sine_chrom in sine_df.groupby('chrom'):
    region_chrom = region_df[region_df['chrom'] == chrom]
    if region_chrom.empty:
        continue

    # 遍历该染色体上的每个 SINE
    for _, sine_row in sine_chrom.iterrows():
        sine_id = sine_row['sine_id']
        sine_start = sine_row['start']
        sine_end = sine_row['end']
        sine_len = sine_row['length']
        sine_strand = sine_row['strand']

        # 找到与该 SINE 重叠的区域
        overlaps = region_chrom[
            (region_chrom['start'] <= sine_end) &
            (region_chrom['end'] >= sine_start)
        ]

        if overlaps.empty:
            continue

        # 计算每个重叠的长度和比例
        for _, reg_row in overlaps.iterrows():
            overlap_start = max(sine_start, reg_row['start'])
            overlap_end = min(sine_end, reg_row['end'])
            overlap_len = overlap_end - overlap_start + 1
            overlap_frac = overlap_len / sine_len

            results.append({
                'sine_id': sine_id,
                'gene_id': reg_row['gene_id'],
                'gene_name': reg_row['gene_name'],
                'transcript_id': reg_row['transcript_id'],
                'region': reg_row['region'],
                'strand': reg_row['strand'],
                'overlap_len': overlap_len,
                'overlap_fraction': round(overlap_frac, 4),
            })

if not results:
    print("没有找到任何重叠！")
    # 输出空文件（带列名）
    empty_df = pd.DataFrame(columns=[
        'sine_id', 'gene_id', 'gene_name', 'transcript_id',
        'region', 'orientation', 'overlap_fraction'
    ])
    empty_df.to_csv(OUTPUT_CSV, index=False)
    print(f"已输出空文件: {OUTPUT_CSV}")
    exit()

overlap_df = pd.DataFrame(results)
print(f"初始重叠记录数: {len(overlap_df)}")

# ==================== 7. 边界过滤 ≥ 80% ====================
# 修改日期：2026-05-31
filtered_df = overlap_df[overlap_df['overlap_fraction'] >= EXONIC_OVERLAP_THRESHOLD].copy()
print(f"边界过滤后 (≥{EXONIC_OVERLAP_THRESHOLD:.0%}): {len(filtered_df)} 条")

# ==================== 8. 跨区域优先级分配 ====================
# 修改日期：2026-05-31
# SINE 跨越多区域时按 CDS > 3UTR > 5UTR 优先级分配
region_priority = {'CDS': 0, '3UTR': 1, '5UTR': 2}
filtered_df['priority'] = filtered_df['region'].map(region_priority)

# 对每个 (sine_id, gene_id, transcript_id) 取优先级最高的区域
# 如有多个相同优先级，保留 overlap_fraction 最大的
filtered_df = filtered_df.sort_values(
    ['sine_id', 'gene_id', 'transcript_id', 'priority', 'overlap_fraction'],
    ascending=[True, True, True, True, False]
).drop_duplicates(
    subset=['sine_id', 'gene_id', 'transcript_id'],
    keep='first'
)
print(f"优先级去重后: {len(filtered_df)} 条")

# ==================== 9. 判断方向 ====================
# 修改日期：2026-05-31
# 将 SINE BED 的 strand 合并进来
sine_strand_map = sine_df.set_index('sine_id')['strand'].to_dict()
filtered_df['sine_strand'] = filtered_df['sine_id'].map(sine_strand_map)
filtered_df['orientation'] = np.where(
    filtered_df['sine_strand'] == filtered_df['strand'],
    'sense',
    'antisense'
)

# ==================== 10. 输出结果 ====================
# 修改日期：2026-05-31
output_df = filtered_df[[
    'sine_id', 'gene_id', 'gene_name', 'transcript_id',
    'region', 'orientation', 'overlap_fraction'
]].copy()

output_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n处理完成！输出文件: {OUTPUT_CSV}")
print(f"最终记录数: {len(output_df)}")
print(f"列: {list(output_df.columns)}")
print(f"\n各区域类型分布:")
print(output_df['region'].value_counts().to_string())
print(f"\n方向分布:")
print(output_df['orientation'].value_counts().to_string())