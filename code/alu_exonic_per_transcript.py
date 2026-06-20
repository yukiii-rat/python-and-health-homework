# -*- coding: utf-8 -*-
"""
Alu 元件与外显子区域交集分析 — 全转录本版本（单次遍历 GTF 优化版）
输入：alu_hg38.bed, gencode.v49.primary_assembly.annotation.gtf
输出：alu_exonic_per_transcript.csv

与 Alu_mRNA.py 的区别：
1. 处理 GTF 中所有转录本（不限于 canonical）
2. 对每个 ENST 转录本输出 Alu 在各区域的存在情况
3. 不进行跨区域优先级分配，保留所有区域信息

适用场景：与 TEDD transcript-level TE 值合并做基因内配对检验

修改日期：2026-06-07
"""

import pandas as pd
import numpy as np
import re
import sys
from collections import defaultdict
import os

print = lambda x: (__import__('builtins').print(x), __import__('sys').stdout.flush())[0] if False else __import__('builtins').print(x, flush=True)

# ==================== 参数设置 ====================
_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.path.dirname(os.path.abspath(sys.argv[0]))
BASE_DIR = os.path.dirname(_script_dir)
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output1')

ALU_BED = os.path.join(DATA_DIR, "alu_hg38.bed")
GTF_FILE = os.path.join(DATA_DIR, "gencode.v49.primary_assembly.annotation.gtf")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "alu_exonic_per_transcript.csv")
EXONIC_OVERLAP_THRESHOLD = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 1. 读取 Alu BED ====================
print("=" * 60)
print("Step 1: 读取 Alu BED 文件")
print("=" * 60)

alu_df = pd.read_csv(
    ALU_BED, sep='\t', header=None,
    names=['chrom', 'start', 'end', 'alu_id', 'score', 'strand']
)
alu_df['start'] = alu_df['start'] + 1
alu_df['length'] = alu_df['end'] - alu_df['start'] + 1
print(f"Alu 数量: {len(alu_df)}")

# ==================== 2. 单次遍历 GTF ====================
print("=" * 60)
print("Step 2: 单次遍历 GTF（3.1GB），收集所有转录本和区域坐标")
print("=" * 60)

all_transcripts = {}
transcript_regions = defaultdict(lambda: defaultdict(list))

import time
t0 = time.time()
line_count = 0

# 正则预编译加速
ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')

with open(GTF_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line_count += 1
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue

        chrom = parts[0]
        feature = parts[2]
        start = int(parts[3])
        end = int(parts[4])
        strand = parts[6]
        attr_str = parts[8]

        # 解析属性 — 只取需要的字段
        tx_id = gid = gname = ''
        for m in ATTR_RE.finditer(attr_str):
            k = m.group(1)
            v = m.group(2)
            if k == 'transcript_id':
                tx_id = v.split('.')[0]
            elif k == 'gene_id':
                gid = v.split('.')[0]
            elif k == 'gene_name':
                gname = v
            if tx_id and gid and gname:
                break

        if feature == 'transcript':
            if tx_id and tx_id not in all_transcripts:
                all_transcripts[tx_id] = {
                    'gene_id': gid, 'gene_name': gname,
                    'chrom': chrom, 'strand': strand,
                }
        elif feature in ('exon', 'CDS', 'UTR') and tx_id:
            transcript_regions[tx_id][feature].append((start, end))

        if line_count % 5000000 == 0:
            print(f"  ... 已处理 {line_count/1e6:.0f}M 行，耗时 {time.time()-t0:.0f}s")

elapsed = time.time() - t0
print(f"GTF 解析完成: {line_count/1e6:.1f}M 行, {elapsed:.0f}s")
print(f"总转录本数: {len(all_transcripts)}")
print(f"有区域坐标的转录本: {len(transcript_regions)}")

# ==================== 3. UTR 分类 ====================
print("=" * 60)
print("Step 3: 分类 UTR 为 5'UTR / 3'UTR")
print("=" * 60)

transcript_final_regions = defaultdict(list)
tx_with_cds = 0

for tx_id, regions in transcript_regions.items():
    info = all_transcripts[tx_id]
    strand, chrom = info['strand'], info['chrom']
    gid, gname = info['gene_id'], info['gene_name']

    cds = regions.get('CDS', [])
    utr = regions.get('UTR', [])

    if cds:
        cds_min = min(s for s, e in cds)
        cds_max = max(e for s, e in cds)
        tx_with_cds += 1

    for s, e in utr:
        if cds:
            region = '5UTR' if (strand == '+' and e <= cds_max) or (strand == '-' and s >= cds_min) else '3UTR'
        else:
            region = 'UTR_nocds'
        transcript_final_regions[tx_id].append((chrom, s, e, region, strand, gid, gname))

    for s, e in cds:
        transcript_final_regions[tx_id].append((chrom, s, e, 'CDS', strand, gid, gname))

print(f"有 CDS: {tx_with_cds}, 构建区域: {len(transcript_final_regions)}")

# ==================== 4. 合并区域 ====================
print("=" * 60)
print("Step 4: 合并相邻/重叠区域")
print("=" * 60)

def merge_intervals(group):
    merged = []
    for s, e in sorted(group):
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

region_rows = []
for tx_id, regions_list in transcript_final_regions.items():
    for chrom, s, e, region, strand, gid, gname in regions_list:
        region_rows.append(dict(zip(
            ['chrom','start','end','region','strand','gene_id','gene_name','transcript_id'],
            [chrom, s, e, region, strand, gid, gname, tx_id]
        )))

region_df = pd.DataFrame(region_rows)
print(f"区域数（合并前）: {len(region_df)}")

merged_rows = []
for (tx_id, region), grp in region_df.groupby(['transcript_id', 'region'], sort=False):
    s0 = grp['start'].iloc[0]
    for s, e in merge_intervals(list(zip(grp['start'], grp['end']))):
        r = grp.iloc[0]
        merged_rows.append({k: r[k] for k in ['chrom','strand','gene_id','gene_name']})
        merged_rows[-1].update({'transcript_id': tx_id, 'region': region, 'start': s, 'end': e})

region_df = pd.DataFrame(merged_rows)
print(f"区域数（合并后）: {len(region_df)}")

# ==================== 5. Alu 重叠计算 ====================
print("=" * 60)
print("Step 5: 计算 Alu 与区域重叠")
print("=" * 60)

results = []
for chrom, ac in alu_df.groupby('chrom'):
    rc = region_df[region_df['chrom'] == chrom]
    if rc.empty:
        continue
    for _, alu in ac.iterrows():
        ol = rc[(rc['start'] <= alu['end']) & (rc['end'] >= alu['start'])]
        if ol.empty:
            continue
        for _, reg in ol.iterrows():
            olen = min(alu['end'], reg['end']) - max(alu['start'], reg['start']) + 1
            ofrac = olen / alu['length']
            results.append({
                'alu_id': alu['alu_id'], 'gene_id': reg['gene_id'],
                'gene_name': reg['gene_name'], 'transcript_id': reg['transcript_id'],
                'region': reg['region'], 'tx_strand': reg['strand'],
                'alu_strand': alu['strand'],
                'overlap_len': olen, 'overlap_fraction': round(ofrac, 4),
            })

if not results:
    print("无重叠！")
    pd.DataFrame(columns=['transcript_id','gene_id','gene_name','alu_5utr','alu_cds','alu_3utr','has_alu_exonic']).to_csv(OUTPUT_CSV, index=False)
    exit()

overlap_df = pd.DataFrame(results)
print(f"初始重叠: {len(overlap_df)}")

# ==================== 6. 过滤 ≥80% ====================
print("=" * 60)
print(f"Step 6: 过滤 (≥{EXONIC_OVERLAP_THRESHOLD:.0%})")
print("=" * 60)

filtered = overlap_df[overlap_df['overlap_fraction'] >= EXONIC_OVERLAP_THRESHOLD].copy()
print(f"过滤后: {len(filtered)}")

if filtered.empty:
    pd.DataFrame(columns=['transcript_id','gene_id','gene_name','alu_5utr','alu_cds','alu_3utr','has_alu_exonic']).to_csv(OUTPUT_CSV, index=False)
    exit()

# ==================== 7. 方向 ====================
filtered['orientation'] = np.where(filtered['alu_strand'] == filtered['tx_strand'], 'sense', 'antisense')

# ==================== 8. Per-transcript 摘要 ====================
print("=" * 60)
print("Step 7: 构建 per-transcript 摘要")
print("=" * 60)

tx_alu_set = set(filtered['transcript_id'])
tx_region_info = filtered.groupby(['transcript_id', 'region']).size().unstack(fill_value=0)

summary = []
for tx_id, info in all_transcripts.items():
    has = tx_id in tx_alu_set
    if has:
        r = tx_region_info.loc[tx_id] if tx_id in tx_region_info.index else pd.Series(dtype=int)
        utr5 = int(r.get('5UTR', 0) > 0)
        cds = int(r.get('CDS', 0) > 0)
        utr3 = int(r.get('3UTR', 0) > 0)
        nocds = int(r.get('UTR_nocds', 0) > 0)
        n_alu = len(filtered[filtered['transcript_id'] == tx_id]['alu_id'].unique())
        orient = filtered[filtered['transcript_id'] == tx_id]['orientation'].value_counts()
        main_orient = orient.index[0] if len(orient) > 0 else ''
    else:
        utr5 = cds = utr3 = nocds = 0
        n_alu = 0
        main_orient = ''

    summary.append({
        'transcript_id': tx_id, 'gene_id': info['gene_id'],
        'gene_name': info['gene_name'], 'chrom': info['chrom'],
        'strand': info['strand'],
        'alu_5utr': utr5, 'alu_cds': cds, 'alu_3utr': utr3,
        'alu_utr_nocds': nocds,
        'has_alu_exonic': 1 if (utr5 or cds or utr3) else 0,
        'alu_count': n_alu, 'alu_orientation': main_orient,
    })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(['gene_name', 'transcript_id']).reset_index(drop=True)

# ==================== 9. 输出 ====================
print("=" * 60)
print("Step 8: 输出结果")
print("=" * 60)

summary_df.to_csv(OUTPUT_CSV, index=False)
print(f"输出: {OUTPUT_CSV}")
print(f"总转录本: {len(summary_df)}")
print(f"含 Alu（外显子）: {summary_df['has_alu_exonic'].sum()}")
print(f"含 Alu 的基因: {summary_df[summary_df['has_alu_exonic'] == 1]['gene_id'].nunique()}")
print(f"5'UTR: {summary_df['alu_5utr'].sum()}, CDS: {summary_df['alu_cds'].sum()}, 3'UTR: {summary_df['alu_3utr'].sum()}")

detail_csv = os.path.join(OUTPUT_DIR, "alu_exonic_overlap_detail.csv")
filtered[['transcript_id','gene_id','gene_name','region','alu_id','orientation','overlap_fraction']].to_csv(detail_csv, index=False)
print(f"详情: {detail_csv} ({len(filtered)} 条)")