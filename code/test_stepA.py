"""
=============================================================================
Step A test - Verify exon-based GTF parsing + feature extraction
=============================================================================
Checkpoints:
  1. GTF correctly reads exon features
  2. split_exons_by_cds correctly splits at CDS boundaries
  3. spliced_utr_dist sums exon blocks (no introns)
  4. l2fc formula is correct
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import re
import os
import time
from collections import defaultdict
from intervaltree import IntervalTree

# ========================== 测试路径 ==========================
TEST_DIR = 'D:/pycharm/data/health and medicine of basic python coding/python and health homework/test_data'
DATA_DIR = 'D:/pycharm/data/health and medicine of basic python coding/python and health homework/data'

GTF_PATH = TEST_DIR + '/test_gencode.gtf'
ALU_BED_PATH = TEST_DIR + '/test_alu.bed'
TEDD_PATH = TEST_DIR + '/test_tedd.csv'
TWOBIT_PATH = DATA_DIR + '/hg38.2bit'
OUTPUT_CSV = TEST_DIR + '/test_features_output.csv'

# ========================== 常量 ==========================
ALU_OVERLAP_THRESHOLD = 50
REGION_PRIORITY = ['3UTR', "5'UTR", 'CDS']

# ========================== 辅助函数（从主脚本复制） ==========================

def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_iv[0])]
    for s, e in sorted_iv[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def split_exons_by_cds(exons, cds_min, cds_max, strand):
    """
    将 exon 区间按照 CDS 边界切割，分配为 5'UTR / CDS / 3'UTR exon blocks。
    所有坐标均为 1-based closed。
    """
    utr5, cds_part, utr3 = [], [], []
    for s, e in sorted(exons):
        if e < cds_min:
            if strand == '+':
                utr5.append((s, e))
            else:
                utr3.append((s, e))
        elif s > cds_max:
            if strand == '+':
                utr3.append((s, e))
            else:
                utr5.append((s, e))
        elif s >= cds_min and e <= cds_max:
            cds_part.append((s, e))
        else:
            if s < cds_min:
                left = (s, cds_min - 1)
                if strand == '+':
                    utr5.append(left)
                else:
                    utr3.append(left)
            inner_s = max(s, cds_min)
            inner_e = min(e, cds_max)
            if inner_s <= inner_e:
                cds_part.append((inner_s, inner_e))
            if e > cds_max:
                right = (cds_max + 1, e)
                if strand == '+':
                    utr3.append(right)
                else:
                    utr5.append(right)
    return utr5, cds_part, utr3


def calc_spliced_dist(tx_reg, assigned_region, alu_start, alu_end):
    """计算 Alu 到 CDS 边界在剪接后 mRNA 上的外显子水平距离。"""
    strand = tx_reg['strand']
    if assigned_region == "5'UTR":
        alu_edge = alu_end if strand == '+' else (alu_start + 1)
        exons = tx_reg['utr5_raw_exons']
        is_rev = (strand == '-')
        forward = True
    elif assigned_region == '3UTR':
        alu_edge = (alu_start + 1) if strand == '+' else alu_end
        exons = tx_reg['utr3_raw_exons']
        is_rev = (strand == '-')
        forward = False
    else:
        return np.nan
    if not exons:
        return np.nan
    exons_sorted = sorted(exons, key=lambda x: x[0], reverse=is_rev)
    target = -1
    for idx, (s, e) in enumerate(exons_sorted):
        if s <= alu_edge <= e:
            target = idx
            break
    if target == -1:
        return 0
    s_ex, e_ex = exons_sorted[target]
    if forward:
        remain = (e_ex - alu_edge) if not is_rev else (alu_edge - s_ex)
        remain = max(0, remain)
        downstream = sum(e - s for s, e in exons_sorted[target + 1:])
        dist = remain + downstream
    else:
        remain = (alu_edge - s_ex) if not is_rev else (e_ex - alu_edge)
        remain = max(0, remain)
        upstream = sum(e - s for s, e in exons_sorted[:target])
        dist = remain + upstream
    assert dist < 50000, f"spliced_utr_dist={dist} 异常大"
    return dist


def reverse_complement(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
            'N': 'N', 'n': 'n'}
    return ''.join(comp.get(b, b) for b in reversed(seq))


# ========================== 1. GTF 解析 ==========================
print("=" * 70)
print("[1] GTF 解析（exon feature）")
print("=" * 70)

GTF_ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')
tx_info = {}
tx_exons = defaultdict(list)
tx_cds = defaultdict(list)
tx_has_cds = set()

with open(GTF_PATH, 'r', encoding='utf-8') as f:
    for line in f:
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

        if feature not in ('transcript', 'CDS', 'exon'):
            continue

        tx_id = gid = gname = ''
        for m in GTF_ATTR_RE.finditer(attr_str):
            k = m.group(1); v = m.group(2)
            if k == 'transcript_id': tx_id = v.split('.')[0]
            elif k == 'gene_id': gid = v.split('.')[0]
            elif k == 'gene_name': gname = v
            if tx_id and gid and gname: break
        if not tx_id: continue

        if feature == 'transcript' and tx_id not in tx_info:
            tx_info[tx_id] = {'gene_id': gid, 'gene_name': gname,
                              'chrom': chrom, 'strand': strand}
        elif feature == 'CDS':
            tx_cds[tx_id].append((start, end))
            tx_has_cds.add(tx_id)
        elif feature == 'exon':
            tx_exons[tx_id].append((start, end))

print(f"  GTF 读取完成")
print(f"  总转录本: {len(tx_info)}")

# 筛选编码转录本 + exon 切割分配区域
cds_tx_ids = set(tx_info.keys()) & tx_has_cds
print(f"  编码转录本: {len(cds_tx_ids)}")

tx_regions = {}
for tx_id in cds_tx_ids:
    info = tx_info[tx_id]
    strand = info['strand']
    cds_merged = merge_intervals(tx_cds.get(tx_id, []))
    if not cds_merged:
        continue
    cds_min = min(s for s, e in cds_merged)
    cds_max = max(e for s, e in cds_merged)
    raw_exons = tx_exons.get(tx_id, [])
    if not raw_exons:
        continue
    utr5_list, cds_list, utr3_list = split_exons_by_cds(
        raw_exons, cds_min, cds_max, strand
    )
    tx_regions[tx_id] = {
        'utr5': utr5_list, 'cds': cds_list, 'utr3': utr3_list,
        'cds_min': cds_min, 'cds_max': cds_max,
        'strand': strand, 'chrom': info['chrom'],
        'gene_id': info['gene_id'], 'gene_name': info['gene_name'],
        'utr5_raw_exons': sorted(utr5_list, key=lambda x: x[0]),
        'utr3_raw_exons': sorted(utr3_list, key=lambda x: x[0]),
    }

print(f"  成功构建区域: {len(tx_regions)} 个转录本")

# 打印每个转录本的区域统计
for tx_id, reg in sorted(tx_regions.items()):
    info = tx_info[tx_id]
    print(f"\n  --- {tx_id} ({reg['gene_name']}, {reg['strand']} strand) ---")
    print(f"      CDS: {reg['cds_min']}-{reg['cds_max']} ({reg['cds_min']}-{reg['cds_max']})")
    print(f"      5'UTR exon blocks: {reg['utr5']}")
    print(f"      3'UTR exon blocks: {reg['utr3']}")
    print(f"      CDS exon blocks: {reg['cds']}")
    # 验证每个 exon block 不超过各自区域（考虑链特异性）
    strand = reg['strand']
    for s, e in reg['utr5']:
        if strand == '+':
            assert e < reg['cds_min'], f"  Error: 5'UTR exon ({s},{e}) crosses CDS start {reg['cds_min']}"
        else:
            assert s > reg['cds_max'], f"  Error: 5'UTR exon ({s},{e}) crosses CDS end {reg['cds_max']}"
    for s, e in reg['utr3']:
        if strand == '+':
            assert s > reg['cds_max'], f"  Error: 3'UTR exon ({s},{e}) crosses CDS end {reg['cds_max']}"
        else:
            assert e < reg['cds_min'], f"  Error: 3'UTR exon ({s},{e}) crosses CDS start {reg['cds_min']}"
    for s, e in reg['cds']:
        assert s >= reg['cds_min'] and e <= reg['cds_max'], \
            f"  Error: CDS exon ({s},{e}) outside CDS [{reg['cds_min']},{reg['cds_max']}]"
    print(f"      [OK] All exon blocks within correct regions")


# ========================== 2. 加载 TEDD TE ==========================
print(f"\n{'=' * 70}")
print("[2] 加载 TEDD TE")
print('=' * 70)

tedd_raw = pd.read_csv(TEDD_PATH)
if 'CELL_LINE' in tedd_raw.columns:
    hek = tedd_raw[tedd_raw['CELL_LINE'].str.upper().str.contains('HEK', na=False)]
    if len(hek) == 0:
        hek = tedd_raw
else:
    hek = tedd_raw

tedd_map = hek.groupby('TRANSCRIPT_ID').first()['TE'].to_dict()
tedd_gene_map = hek.groupby('TRANSCRIPT_ID').first()['GENE_ID'].to_dict()
print(f"  TEDD 转录本: {len(tedd_map)}")

tedd_common = set(tx_regions.keys()) & set(tedd_map.keys())
print(f"  在 GTF + TEDD 中的: {len(tedd_common)}")
tx_regions = {tx_id: tx_regions[tx_id] for tx_id in tedd_common}

for tx_id in tx_regions:
    print(f"    {tx_id}: TE={tedd_map[tx_id]:.4f}")


# ========================== 3. 构建 Alu IntervalTree ==========================
print(f"\n{'=' * 70}")
print("[3] 加载 Alu BED")
print('=' * 70)

alu_df = pd.read_csv(ALU_BED_PATH, sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'alu_id', 'score', 'strand'])
alu_trees = {}
alu_details = {}

for _, row in alu_df.iterrows():
    chrom = row['chrom']
    if chrom not in alu_trees:
        alu_trees[chrom] = IntervalTree()
    alu_unique_id = f"{chrom}_{row['start']}_{row['end']}_{row['strand']}"
    subfamily = row['alu_id'].split('-')[0] if '-' in str(row['alu_id']) else str(row['alu_id'])
    alu_trees[chrom].addi(row['start'] + 1, row['end'] + 1, alu_unique_id)
    alu_details[alu_unique_id] = {
        'chrom': chrom, 'start': row['start'], 'end': row['end'],
        'strand': row['strand'], 'subfamily': subfamily,
    }

print(f"  Alu 总数: {len(alu_df)}")


# ========================== 4. 打开 hg38.2bit ==========================
print(f"\n{'=' * 70}")
print("[4] 打开 hg38.2bit")
print('=' * 70)
import twobitreader
genome = twobitreader.TwoBitFile(TWOBIT_PATH)
print("  ✅ 已打开")


# ========================== 5. 碰撞检测 + l2fc ==========================
print(f"\n{'=' * 70}")
print("[5] 碰撞检测 + l2fc")
print('=' * 70)

from Bio.Align import PairwiseAligner
aligner = PairwiseAligner()
aligner.mode = 'local'
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.gap_score = -2

# 碰撞检测
tx_alu_status = {}
tx_alu_overlaps = {}
for tx_id in tx_regions:
    reg = tx_regions[tx_id]
    alu_tree = alu_trees.get(reg['chrom'])
    if alu_tree is None:
        tx_alu_status[tx_id] = False
        continue
    all_overlaps = {}
    for region_key, region_label in [('utr5', "5'UTR"), ('cds', 'CDS'), ('utr3', '3UTR')]:
        intervals = reg[region_key]
        if not intervals:
            continue
        for qs, qe in intervals:
            for alu_iv in alu_tree.overlap(qs, qe + 1):
                ol = min(qe, alu_iv.end - 1) - max(qs, alu_iv.begin) + 1
                if ol >= ALU_OVERLAP_THRESHOLD:
                    aid = alu_iv.data
                    if aid not in all_overlaps:
                        all_overlaps[aid] = {}
                    all_overlaps[aid][region_label] = ol
    if all_overlaps:
        tx_alu_status[tx_id] = True
        tx_alu_overlaps[tx_id] = all_overlaps
    else:
        tx_alu_status[tx_id] = False

print(f"  Alu+ 转录本: {sum(1 for v in tx_alu_status.values() if v)}")
print(f"  Alu- 转录本: {sum(1 for v in tx_alu_status.values() if not v)}")
for tx_id, is_plus in tx_alu_status.items():
    print(f"    {tx_id} ({tx_regions[tx_id]['gene_name']}): {'Alu+' if is_plus else 'Alu-'}")
    if is_plus:
        n_alu = len(tx_alu_overlaps[tx_id])
        regions = set()
        for aid, ro in tx_alu_overlaps[tx_id].items():
            regions.update(ro.keys())
        print(f"      Alu 数: {n_alu}, 覆盖区域: {regions}")

# 计算 l2fc
gene_te_alu_minus = defaultdict(list)
for tx_id, is_plus in tx_alu_status.items():
    if not is_plus:
        gid = tx_regions[tx_id]['gene_id']
        te = tedd_map.get(tx_id)
        if te is not None and te > 0:
            gene_te_alu_minus[gid].append(te)

gene_baseline = {}
for gid, te_list in gene_te_alu_minus.items():
    gene_baseline[gid] = np.median(te_list)

print(f"\n  基因基线:")
for gid, b in gene_baseline.items():
    print(f"    {gid}: Alu- 中位数 TE = {b:.4f}")

tx_l2fc = {}
for tx_id in tx_alu_overlaps.keys():
    gid = tx_regions[tx_id]['gene_id']
    baseline = gene_baseline.get(gid, 0)
    if baseline <= 0:
        continue
    te = tedd_map.get(tx_id)
    if te is None:
        continue
    l2fc = np.log2((te + 0.001) / (baseline + 0.001))
    tx_l2fc[tx_id] = l2fc

for tx_id, l2fc in tx_l2fc.items():
    te = tedd_map[tx_id]
    gid = tx_regions[tx_id]['gene_id']
    baseline = gene_baseline.get(gid, 'N/A')
    print(f"    {tx_id}: TE={te:.4f}, baseline={baseline}, l2fc={l2fc:.4f}")
    print(f"      公式验证: log2(({te:.4f} + 0.001) / ({baseline} + 0.001)) = {l2fc:.4f}")

tx_alu_overlaps = {tx_id: ov for tx_id, ov in tx_alu_overlaps.items() if tx_id in tx_l2fc}


# ========================== 6. 特征提取 ==========================
print(f"\n{'=' * 70}")
print("[6] 特征提取（逐 Alu）")
print('=' * 70)

features_rows = []
HUMAN_18S_HELIX34_TARGET = "CGATGTCGTGTTGGATTGGA"

for tx_id in sorted(tx_alu_overlaps.keys()):
    reg = tx_regions[tx_id]
    strand = reg['strand']
    all_overlaps = tx_alu_overlaps[tx_id]
    tx_te_l2fc = tx_l2fc[tx_id]

    print(f"\n  --- {tx_id} ({reg['gene_name']}) ---")

    for alu_id, region_overlaps in sorted(all_overlaps.items()):
        detail = alu_details.get(alu_id)
        if detail is None:
            continue

        alu_start, alu_end = detail['start'], detail['end']
        alu_len = alu_end - alu_start
        alu_center = (alu_start + alu_end) / 2
        alu_chrom = detail['chrom']
        alu_strand = detail['strand']

        # 区域分配
        assigned_region = None
        for r in REGION_PRIORITY:
            if r in region_overlaps:
                assigned_region = r
                break
        if assigned_region is None:
            continue

        # spliced_utr_dist
        spliced_utr_dist = calc_spliced_dist(reg, assigned_region, alu_start, alu_end)

        # is_antisense
        is_antisense = 0 if alu_strand == strand else 1

        # subfamily
        sf = detail['subfamily']

        # 序列
        raw_seq = genome[alu_chrom][alu_start:alu_end].upper()
        if alu_strand == '-':
            alu_seq = reverse_complement(raw_seq)
        else:
            alu_seq = raw_seq

        # Alu 坐标（BED 0-based）
        print(f"    Alu {alu_id}: {assigned_region} (chr{alu_chrom}:{alu_start}-{alu_end}, {alu_strand}, overlap={region_overlaps.get(assigned_region, 0)}bp)")
        print(f"      spliced_utr_dist={spliced_utr_dist}, is_antisense={is_antisense}, subfamily={sf}")
        print(f"      Alu 序列长度: {len(alu_seq)}bp")

        features_rows.append({
            'transcript_id': tx_id,
            'gene_id': reg['gene_id'],
            'gene_name': reg['gene_name'],
            'chrom': alu_chrom,
            'strand': strand,
            'alu_id': sf,
            'alu_unique_id': alu_id,
            'alu_chr_start': alu_start,
            'alu_chr_end': alu_end,
            'alu_length': alu_len,
            'alu_strand': alu_strand,
            'region_inserted': assigned_region,
            'overlap_bp': region_overlaps.get(assigned_region, 0),
            'spliced_utr_dist': round(spliced_utr_dist, 0) if not np.isnan(spliced_utr_dist) else np.nan,
            'is_antisense': is_antisense,
            'subfamily': sf,
            'l2fc': round(tx_te_l2fc, 4),
        })

print(f"\n{'=' * 70}")
print(f"特征提取完成: {len(features_rows)} 条记录")
print(f'{"=" * 70}')

# ========================== 7. 输出验证 ==========================
feat_df = pd.DataFrame(features_rows)
print(f"\n{'=' * 70}")
print("[7] 输出验证")
print('=' * 70)
print(f"  总行数: {len(feat_df)}")
print(f"  唯一转录本: {feat_df['transcript_id'].nunique()}")
print(f"  唯一基因: {feat_df['gene_id'].nunique()}")
print(f"\n  l2fc 分布:")
print(f"    {feat_df[['transcript_id', 'l2fc']].drop_duplicates().to_string(index=False)}")
print(f"\n  spliced_utr_dist:")
for region in feat_df['region_inserted'].unique():
    sub = feat_df[feat_df['region_inserted'] == region]['spliced_utr_dist'].dropna()
    if len(sub) > 0:
        print(f"    {region}: min={sub.min():.0f}, max={sub.max():.0f}, median={sub.median():.0f}")
print(f"\n  区域分布:")
print(f"    {feat_df['region_inserted'].value_counts().to_string()}")

# 保存
feat_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ 测试完成，输出已保存: {OUTPUT_CSV}")
