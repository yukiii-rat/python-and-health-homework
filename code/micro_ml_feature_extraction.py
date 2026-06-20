# -*- coding: utf-8 -*-
"""
=============================================================================
转录本级别 Alu 微观特征提取脚本  — 因果分析特征矩阵构建
=============================================================================

功能：
  结合 gencode.v49.gtf + alu_hg38.bed + hg38.2bit + TEDD TE 数据，
  对每个有 TE 值的编码转录本提取 Alu 微观特征。

目标变量：
  per-transcript l2fc = log2((TE + 0.001) / (基因内 Alu- 转录本中位数 TE + 0.001))

输出：
  micro_ml_features_integrated.csv  — 特征矩阵（transcript_id × Alu 特征 + l2fc）

依赖：
  pip install twobitreader biopython tqdm

作者备注：
  - pyranges 的 ncls 依赖在 Windows 上编译失败，使用 intervaltree + pandas 替代
  - GTF 解析使用流式处理（单次遍历），内存峰值 < 2GB
  - 2025-06-16: 重构为 per-transcript l2fc + 正确 Alu 坐标 + spliced_utr_dist
  - 2026-06-16: GTF 解析改用 exon feature + CDS 边界切割，spliced_utr_dist 天然不含内含子
=============================================================================
"""

import pandas as pd
import numpy as np
import re
import os
import sys
import time
from collections import defaultdict
from intervaltree import IntervalTree
import twobitreader
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# ========================== 0. 常量定义 ==========================

# 2024 Nature Communications: 人类 18S rRNA Helix 34 核心保守结合靶向序列
HUMAN_18S_HELIX34_TARGET = "CGATGTCGTGTTGGATTGGA"

ALU_OVERLAP_THRESHOLD = 50  # bp，判定 Alu 属于某区域的最小重叠
MIN_PAIRED_GENES = 20       # 与 v4.0 保持一致（仅用于全局基线质量过滤）

# 区域优先级：当 Alu 同时≥50bp 于多个区域，按此分配
REGION_PRIORITY = ['3UTR', "5'UTR", 'CDS']

# 路径设置
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else \
    os.path.dirname(os.path.abspath(sys.argv[0]))
BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output1')

GTF_PATH = os.path.join(DATA_DIR, "gencode.v49.primary_assembly.annotation.gtf")
ALU_BED_PATH = os.path.join(DATA_DIR, "alu_hg38.bed")
TWOBIT_PATH = os.path.join(DATA_DIR, "hg38.2bit")
TEDD_PATH = os.path.join(DATA_DIR, "TEDD00137.distribution_transcript.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "micro_ml_features_integrated.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("转录本级别 Alu 微观特征提取 — 机器学习特征矩阵构建")
print("=" * 80)

t_start = time.time()

# ========================== 1. 辅助函数 ==========================

def merge_intervals(intervals):
    """合并重叠/相邻区间，返回 (start, end) 列表（1-based closed）"""
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


def count_alu_overlaps_detail(query_intervals, alu_tree, threshold=50):
    """
    返回与 query_intervals 重叠 ≥ threshold bp 的 (alu_id, overlap_bp) 列表。
    用于特征提取时需要逐 Alu 处理，不能只计数。
    """
    if not query_intervals or not alu_tree:
        return []
    results = {}  # alu_id → max_overlap_bp
    for qs, qe in query_intervals:
        for alu_iv in alu_tree.overlap(qs, qe + 1):
            ol = min(qe, alu_iv.end - 1) - max(qs, alu_iv.begin) + 1
            if ol >= threshold:
                aid = alu_iv.data
                if aid not in results or ol > results[aid]:
                    results[aid] = ol
    return [(aid, ol) for aid, ol in results.items()]


def reverse_complement(seq):
    """DNA 反向互补"""
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
            'N': 'N', 'n': 'n'}
    return ''.join(comp.get(b, b) for b in reversed(seq))


def calc_spliced_dist(tx_reg, assigned_region, alu_start, alu_end):
    """
    计算 Alu 到 CDS 边界在剪接后 mRNA 上的外显子水平距离。
    完全基于 GTF exon-block 坐标，不含任何内含子长度。
    """
    strand = tx_reg['strand']

    # Alu 的 CDS 侧边缘（在剪接后 mRNA 上靠 CDS 的那一侧）
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

    # 按 mRNA 5'→3' 排序
    exons_sorted = sorted(exons, key=lambda x: x[0], reverse=is_rev)

    # 找到 Alu 边缘落在哪个外显子
    target = -1
    for idx, (s, e) in enumerate(exons_sorted):
        if s <= alu_edge <= e:
            target = idx
            break

    if target == -1:
        # 不在任何 UTR 外显子内 → 紧邻 CDS 边界，距离为 0
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

    assert dist < 50000, f"spliced_utr_dist={dist} 异常大 (tx={tx_reg.get('gene_name','?')}, region={assigned_region})"
    return dist


def split_exons_by_cds(exons, cds_min, cds_max, strand):
    """
    将 exon 区间按照 CDS 边界切割，分配为 5'UTR / CDS / 3'UTR exon blocks。
    所有坐标均为 1-based closed。

    参数:
        exons: [(start, end)] — 该转录本所有 exon 区间
        cds_min, cds_max: CDS 区域的基因组最小/最大边界（来自合并的 CDS 区间）
        strand: '+' 或 '-'

    返回:
        (utr5_blocks, cds_blocks, utr3_blocks) 每个都是 [(start, end)]
    """
    utr5, cds_part, utr3 = [], [], []

    for s, e in sorted(exons):
        if e < cds_min:
            # Exon 完全在 CDS 左侧基因组位置
            if strand == '+':
                utr5.append((s, e))
            else:
                utr3.append((s, e))
        elif s > cds_max:
            # Exon 完全在 CDS 右侧基因组位置
            if strand == '+':
                utr3.append((s, e))
            else:
                utr5.append((s, e))
        elif s >= cds_min and e <= cds_max:
            # Exon 完全在 CDS 内部
            cds_part.append((s, e))
        else:
            # Exon 跨越 CDS 边界 — 切割
            if s < cds_min:
                left = (s, cds_min - 1)
                if strand == '+':
                    utr5.append(left)
                else:
                    utr3.append(left)
            # CDS 内部部分
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


# ========================== 2. GTF 解析 ==========================

print("\n[1/5] 解析 GTF 注释...")
t1 = time.time()

GTF_ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')

# 存储结构
tx_info = {}        # tx_id → {gene_id, gene_name, chrom, strand}
tx_exons = defaultdict(list)   # tx_id → [(start, end)]
tx_cds = defaultdict(list)       # tx_id → [(start, end)]

line_count = 0
tx_has_cds = set()

with open(GTF_PATH, 'r', encoding='utf-8') as f:
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

        if feature not in ('transcript', 'CDS', 'exon'):
            continue

        tx_id = gid = gname = ''
        for m in GTF_ATTR_RE.finditer(attr_str):
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
        if not tx_id:
            continue

        if feature == 'transcript' and tx_id not in tx_info:
            tx_info[tx_id] = {'gene_id': gid, 'gene_name': gname,
                              'chrom': chrom, 'strand': strand}
        elif feature == 'CDS':
            tx_cds[tx_id].append((start, end))
            tx_has_cds.add(tx_id)
        elif feature == 'exon':
            tx_exons[tx_id].append((start, end))

        if line_count % 500000 == 0:
            print(f"  已处理 {line_count/1e6:.1f}M 行...", flush=True)

print(f"  GTF 行数: {line_count/1e6:.1f}M")
print(f"  总转录本: {len(tx_info)}")
print(f"  有 CDS: {len(tx_has_cds)}")

# 筛选编码转录本 + 用 exon 边界分配 5'UTR/CDS/3'UTR
cds_tx_ids = set(tx_info.keys()) & tx_has_cds
print(f"  编码转录本: {len(cds_tx_ids)}")

# 预计算每个编码转录本的三个区域坐标（基于 exon 切割）
tx_regions = {}  # tx_id → {utr5, cds, utr3} 每个都是 [(start,end)]

for tx_id in tqdm(cds_tx_ids, desc="  分配区域", unit="tx"):
    info = tx_info[tx_id]
    strand = info['strand']

    cds_merged = merge_intervals(tx_cds.get(tx_id, []))
    if not cds_merged:
        continue
    cds_min = min(s for s, e in cds_merged)
    cds_max = max(e for s, e in cds_merged)

    # 获取该转录本的所有 exon 区间
    raw_exons = tx_exons.get(tx_id, [])
    if not raw_exons:
        continue

    # 按 CDS 边界切割 exon 区间，分配到 5'UTR / CDS / 3'UTR
    utr5_list, cds_list, utr3_list = split_exons_by_cds(
        raw_exons, cds_min, cds_max, strand
    )

    tx_regions[tx_id] = {
        'utr5': utr5_list,
        'cds': cds_list,
        'utr3': utr3_list,
        'cds_min': cds_min,
        'cds_max': cds_max,
        'strand': strand,
        'chrom': info['chrom'],
        'gene_id': info['gene_id'],
        'gene_name': info['gene_name'],
        # 存储 exon-level UTR 外显子列表（天然不含 intron）用于 spliced_utr_dist
        'utr5_raw_exons': sorted(utr5_list, key=lambda x: x[0]),
        'utr3_raw_exons': sorted(utr3_list, key=lambda x: x[0]),
    }

print(f"  有效编码转录本: {len(tx_regions)}")
print(f"  GTF 解析耗时: {time.time()-t1:.0f}s")


# ========================== 3. 加载 TEDD TE 数据 ==========================

print(f"\n{'=' * 60}")
print('Step 3: 加载 TEDD TE 数据')
print('=' * 60)
tedd_t3 = time.time()

tedd_raw = pd.read_csv(TEDD_PATH)
# TEDD 可能有多组织/多条件，取 HEK293T 或首个 TE 值
if 'CELL_LINE' in tedd_raw.columns:
    # 优先 HEK293T
    hek = tedd_raw[tedd_raw['CELL_LINE'].str.upper().str.contains('HEK', na=False)]
    if len(hek) == 0:
        hek = tedd_raw  # fallback
else:
    hek = tedd_raw

# 按转录本去重（取首个 TE）
tedd_map = hek.groupby('TRANSCRIPT_ID').first()['TE'].to_dict()
tedd_gene_map = hek.groupby('TRANSCRIPT_ID').first()['GENE_ID'].to_dict()

print(f'  TEDD 总行数: {len(tedd_raw):,}')
print(f'  唯一转录本 TE 值: {len(tedd_map):,}')
print(f'  TEDD 加载耗时: {time.time()-tedd_t3:.0f}s')

# 过滤 tx_regions 为只在 TEDD 中的转录本
tedd_common = set(tx_regions.keys()) & set(tedd_map.keys())
print(f'  在 GTF 和 TEDD 中的编码转录本: {len(tedd_common):,}')
tx_regions = {tx_id: tx_regions[tx_id] for tx_id in tedd_common}


# ========================== 4. 构建 Alu IntervalTree ==========================

print(f"\n{'=' * 60}")
print('Step 4: 加载 Alu BED 并构建 IntervalTree')
print('=' * 60)
t2 = time.time()

alu_df = pd.read_csv(ALU_BED_PATH, sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'alu_id', 'score', 'strand'])
alu_trees = {}    # chrom → IntervalTree
alu_details = {}  # unique_id → {chrom, start, end, strand, subfamily}

for _, row in tqdm(alu_df.iterrows(), total=len(alu_df), desc="  构建索引", unit="alu"):
    chrom = row['chrom']
    if chrom not in alu_trees:
        alu_trees[chrom] = IntervalTree()
    # BED 0-based half-open → 1-based closed
    # 使用唯一标识符避免 BED 中非唯一 alu_id 导致的坐标覆盖 Bug
    alu_unique_id = f"{chrom}_{row['start']}_{row['end']}_{row['strand']}"
    subfamily = row['alu_id'].split('-')[0] if '-' in str(row['alu_id']) else str(row['alu_id'])
    alu_trees[chrom].addi(row['start'] + 1, row['end'] + 1, alu_unique_id)
    alu_details[alu_unique_id] = {
        'chrom': chrom,
        'start': row['start'],
        'end': row['end'],
        'strand': row['strand'],
        'subfamily': subfamily,
    }

print(f"  Alu 总数: {len(alu_df):,}")
print(f"  染色体/Scaffold: {len(alu_trees)}")
print(f"  加载耗时: {time.time()-t2:.0f}s")


# ========================== 5. 打开 hg38.2bit ==========================

print(f"\n{'=' * 60}")
print('Step 5: 打开 hg38.2bit 基因组')
print('=' * 60)
genome = twobitreader.TwoBitFile(TWOBIT_PATH)
print(f"  染色体数: {len(list(genome.keys()))}")


# ========================== 6. 预计算：Alu 碰撞 → 分类 Alu+/Alu- → 基因基线 l2fc ==========================

print(f"\n{'=' * 60}")
print('Step 6: 预计算 Alu+ / Alu- 分类 + 基因基线 l2fc')
print('=' * 60)
t_pre = time.time()

# 初始化 PairwiseAligner（用于后续特征提取中的 rrna_18s_score）
aligner = PairwiseAligner()
aligner.mode = 'local'
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.gap_score = -2

# 第一遍：对所有 TEDD 编码转录本做碰撞检测，区分 Alu+/Alu-
tx_alu_status = {}      # tx_id → True/False
tx_alu_overlaps = {}    # tx_id → {alu_id: {region: overlap_bp}} (仅 Alu+)

for tx_id in tqdm(list(tx_regions.keys()), desc="  碰撞检测", unit="tx"):
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

n_alu_plus = sum(1 for v in tx_alu_status.values() if v)
n_alu_minus = sum(1 for v in tx_alu_status.values() if not v)
print(f'  Alu+ 转录本: {n_alu_plus:,}')
print(f'  Alu- 转录本: {n_alu_minus:,}')

# 计算每基因的 Alu- 中位数 TE 基线
gene_te_alu_minus = defaultdict(list)
for tx_id, is_plus in tx_alu_status.items():
    if not is_plus:
        gid = tx_regions[tx_id]['gene_id']
        te = tedd_map.get(tx_id)
        if te is not None and te > 0:
            gene_te_alu_minus[gid].append(te)

gene_baseline = {}  # gene_id → median TE of Alu- transcripts
for gid, te_list in gene_te_alu_minus.items():
    gene_baseline[gid] = np.median(te_list)

n_baseline_genes = sum(1 for v in gene_baseline.values() if v > 0)
print(f'  有 Alu- 基线（中位数 > 0）的基因: {n_baseline_genes:,}')

# 计算每个 Alu+ 转录本的 l2fc
tx_l2fc = {}  # tx_id → l2fc
n_no_baseline = 0
for tx_id in tqdm(tx_alu_overlaps.keys(), desc="  计算 l2fc", unit="tx"):
    gid = tx_regions[tx_id]['gene_id']
    baseline = gene_baseline.get(gid, 0)
    if baseline <= 0:
        n_no_baseline += 1
        continue
    te = tedd_map.get(tx_id)
    if te is None:
        continue
    l2fc = np.log2((te + 0.001) / (baseline + 0.001))
    tx_l2fc[tx_id] = l2fc

print(f'  有 l2fc 的 Alu+ 转录本: {len(tx_l2fc):,}')
print(f'  无基线被跳过的: {n_no_baseline:,}')
print(f'  预计算耗时: {time.time()-t_pre:.0f}s')

# 过滤：只保留有 l2fc 的 Alu+ 转录本
tx_alu_overlaps = {tx_id: ov for tx_id, ov in tx_alu_overlaps.items()
                   if tx_id in tx_l2fc}


# ========================== 7. 特征提取主循环 ==========================

print(f"\n{'=' * 60}")
print('Step 7: 特征提取（逐 Alu）')
print('=' * 60)
t3 = time.time()

features_rows = []  # 每行 = 一个 (transcript_id, alu_id, l2fc) 组合

for tx_id in tqdm(list(tx_alu_overlaps.keys()), desc="  提取特征", unit="tx"):
    reg = tx_regions[tx_id]
    strand = reg['strand']
    all_overlaps = tx_alu_overlaps[tx_id]
    tx_te_l2fc = tx_l2fc[tx_id]

    for alu_id, region_overlaps in all_overlaps.items():
        detail = alu_details.get(alu_id)
        if detail is None:
            continue

        alu_start, alu_end = detail['start'], detail['end']
        alu_len = alu_end - alu_start
        alu_center = (alu_start + alu_end) / 2
        alu_chrom = detail['chrom']
        alu_strand = detail['strand']

        # ---- 特征 1: region_inserted (按优先级分配) ----
        assigned_region = None
        for r in REGION_PRIORITY:
            if r in region_overlaps:
                assigned_region = r
                break
        if assigned_region is None:
            continue

        # ---- 特征 2: region_relative_pos (Alu 中心在区域内的相对位置 0~1) ----
        if assigned_region == "5'UTR":
            region_intervals = reg['utr5']
        elif assigned_region == 'CDS':
            region_intervals = reg['cds']
        else:
            region_intervals = reg['utr3']

        if region_intervals:
            r_min = min(s for s, e in region_intervals)
            r_max = max(e for s, e in region_intervals)
            region_len = r_max - r_min
            if region_len > 0:
                relative_pos = (alu_center - r_min) / region_len
                relative_pos = max(0.0, min(1.0, relative_pos))
            else:
                relative_pos = 0.5
        else:
            relative_pos = np.nan

        # ---- 特征 3: log2_dist_to_aug / log2_dist_to_stop (保留但不再主要使用) ----
        cds_min = reg['cds_min']
        cds_max = reg['cds_max']
        if strand == '+':
            dist_to_aug = alu_center - cds_min
            dist_to_stop = alu_center - cds_max
        else:
            dist_to_aug = cds_max - alu_center
            dist_to_stop = cds_min - alu_center
        log2_dist_to_aug = np.log2(abs(dist_to_aug) + 1) if not np.isnan(dist_to_aug) else np.nan
        log2_dist_to_stop = np.log2(abs(dist_to_stop) + 1) if not np.isnan(dist_to_stop) else np.nan

        # ---- 特征 4: spliced_utr_dist (剪接后外显子水平距离) ----
        spliced_utr_dist = calc_spliced_dist(reg, assigned_region, alu_start, alu_end)

        # ---- 特征 5: is_antisense ----
        is_antisense = 0 if alu_strand == strand else 1

        # ---- 特征 6: subfamily_age ----
        sf = detail['subfamily']
        if 'AluJ' in sf or 'aluJ' in sf:
            subfamily_age = 1
        elif 'AluS' in sf or 'aluS' in sf:
            subfamily_age = 2
        elif 'AluY' in sf or 'aluY' in sf:
            subfamily_age = 3
        else:
            subfamily_age = 0

        # ---- 序列特征: 需要提取 Alu 碱基序列 ----
        try:
            raw_seq = genome[alu_chrom][alu_start:alu_end].upper()
        except Exception:
            continue
        if not raw_seq:
            continue
        if alu_strand == '-':
            alu_seq = reverse_complement(raw_seq)
        else:
            alu_seq = raw_seq

        seq_len = len(alu_seq)
        if seq_len == 0:
            continue

        # DNA → RNA 转换（UUG 等 RNA 基序需要 U 而非 T）
        alu_rna = alu_seq.replace('T', 'U')

        # ---- 特征 7: agg_motif_density ----
        agg_motif_density = alu_rna.count('AGG') / seq_len

        # ---- 特征 8: ugg_motif_density ----
        ugg_motif_density = alu_rna.count('UGG') / seq_len

        # ---- 特征 9: sl1_ugg_count ----
        sl1_seq = alu_rna[-80:] if seq_len >= 80 else alu_rna
        sl1_ugg_count = sl1_seq.count('UGG')

        # ---- 特征 10: gc_rich_stem_density ----
        gc_stem_count = 0
        for i in range(len(alu_seq) - 3):
            if all(b in 'GCgc' for b in alu_seq[i:i+4]):
                gc_stem_count += 1
        gc_rich_stem_density = gc_stem_count / seq_len

        # ---- 特征 11: rrna_18s_score ----
        try:
            rrna_18s_score = aligner.align(alu_seq, HUMAN_18S_HELIX34_TARGET).score
        except Exception:
            rrna_18s_score = np.nan

        # ---- 组装特征行 ----
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
            'region_relative_pos': round(relative_pos, 4),
            'log2_dist_to_aug': round(log2_dist_to_aug, 4),
            'log2_dist_to_stop': round(log2_dist_to_stop, 4),
            'spliced_utr_dist': round(spliced_utr_dist, 0) if not np.isnan(spliced_utr_dist) else np.nan,
            'is_antisense': is_antisense,
            'subfamily_age': subfamily_age,
            'subfamily': sf,
            'agg_motif_density': round(agg_motif_density, 6),
            'ugg_motif_density': round(ugg_motif_density, 6),
            'sl1_ugg_count': sl1_ugg_count,
            'gc_rich_stem_density': round(gc_rich_stem_density, 6),
            'rrna_18s_score': round(rrna_18s_score, 4) if not np.isnan(rrna_18s_score) else np.nan,
            'l2fc': round(tx_te_l2fc, 4),
        })

print(f"  特征提取完成: {len(features_rows)} 条 (transcript×Alu) 记录")
print(f"  涉及转录本: {len(set(r['transcript_id'] for r in features_rows))}")
print(f"  特征提取耗时: {time.time()-t3:.0f}s")


# ========================== 8. 输出 ==========================

print(f"\n{'=' * 60}")
print('Step 8: 保存特征矩阵')
print('=' * 60)

feat_df = pd.DataFrame(features_rows)

feat_df.to_csv(OUTPUT_CSV, index=False, float_format='%.6g')

print(f'  已保存: {OUTPUT_CSV}')
print(f'  总行数 (Alu×transcript): {len(feat_df):,}')
print(f'  唯一转录本: {feat_df["transcript_id"].nunique():,}')
print(f'  唯一基因: {feat_df["gene_id"].nunique():,}')
print(f'  有 l2fc 值: {feat_df["l2fc"].notna().sum():,}')
print(f'\n  l2fc 分布:')
print(f'    均值: {feat_df["l2fc"].mean():.4f}')
print(f'    中位数: {feat_df["l2fc"].median():.4f}')
print(f'    SD: {feat_df["l2fc"].std():.4f}')
print(f'    最小值: {feat_df["l2fc"].min():.4f}')
print(f'    最大值: {feat_df["l2fc"].max():.4f}')
print(f'\n  spliced_utr_dist 分布:')
valid_dist = feat_df['spliced_utr_dist'].dropna()
if len(valid_dist) > 0:
    print(f'    均值: {valid_dist.mean():.0f}')
    print(f'    中位数: {valid_dist.median():.0f}')
    print(f'    最小值: {valid_dist.min():.0f}')
    print(f'    最大值: {valid_dist.max():.0f}')
    for region in ["5'UTR", '3UTR']:
        sub = feat_df[(feat_df['region_inserted'] == region) & feat_df['spliced_utr_dist'].notna()]['spliced_utr_dist']
        if len(sub) > 0:
            print(f'    {region}: n={len(sub):,}, median={sub.median():.0f}, max={sub.max():.0f}')
print(f'\n区域分布:')
print(feat_df['region_inserted'].value_counts().to_string())
print(f'\n总运行耗时: {time.time()-t_start:.0f}s ({((time.time()-t_start)/60):.1f} 分钟)')
print('完成！')
