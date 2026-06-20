# -*- coding: utf-8 -*-
"""
=============================================================================
同基因内转录本配对的 Alu 与翻译效率 (TE) 差异分析  — v4.0 组合式剂量与位置
=============================================================================

v4.0 相对于 v3.2 的关键改进：
  1. [计数检测] Step 2 从布尔碰撞 → 计数碰撞（统计每个区域的 Alu 数量）
  2. [布局指纹] 生成 layout = f"u{n_utr5}_c{n_cds}_t{n_utr3}" 组合编码
  3. [组合分析] Step 6 完全重写为多布局 + 剂量 + 协同分析
  4. [新可视化] L2FC 柱状图、组合热图、剂量效应图、协同效应图

分析流程：
  1. GTF 解析 → 提取编码转录本的 5'UTR / CDS / 3'UTR 坐标
  2. Alu 计数检测（≥50 bp 重叠阈值，统计各区域 Alu 数量）
  3. TE 数据对齐 + 极值清洗（TE > 0 且非空）
  4. 基因内中位数聚合 → 配对数据
  5. Wilcoxon 符号秩检验（全区域总览）
  6. 组合布局分析（单布局配对 + 剂量效应 + 协同效应）
  7. 可视化（Barplot / Heatmap / Dose / Synergy）

输入（请确认 BASE_DIR 指向你的数据目录）：
  - data/gencode.v49.primary_assembly.annotation.gtf
  - data/alu_hg38.bed
  - data/TEDD00137.distribution_transcript.csv

输出（v4 后缀以区分版本）：
  - output1/alu_te_within_gene_v4_results.csv
  - output1/alu_te_within_gene_v4_paired_genes.csv
  - output1/alu_te_within_gene_v4_paired_violin.png
  - output1/alu_te_within_gene_v4_layout_analysis.csv
  - output1/alu_te_within_gene_v4_dose_analysis.csv
  - output1/alu_te_within_gene_v4_synergy_analysis.csv
  - output1/alu_te_within_gene_v4_layout_l2fc_barplot.png
  - output1/alu_te_within_gene_v4_layout_heatmap.png
  - output1/alu_te_within_gene_v4_layout_dose.png
  - output1/alu_te_within_gene_v4_layout_synergy.png

修改日期：2026-06-12
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, kruskal, mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import re
import os
import sys
from collections import defaultdict
import time
from intervaltree import IntervalTree
from itertools import product

warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ========================== 0. 路径设置 ==========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.path.dirname(os.path.abspath(sys.argv[0]))
# 脚本在 output1/gene_pair_analize/ 下，项目根目录需上跳两级
BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output1')

# 输入文件路径
GTF_PATH = os.path.join(DATA_DIR, "gencode.v49.primary_assembly.annotation.gtf")
ALU_BED_PATH = os.path.join(DATA_DIR, "alu_hg38.bed")
TEDD_PATH = os.path.join(DATA_DIR, "TEDD00137.distribution_transcript.csv")

# 输出路径（v4 后缀）
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_results.csv")
OUTPUT_PAIRED_GENES = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_paired_genes.csv")
OUTPUT_VIOLIN = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_paired_violin.png")
OUTPUT_LAYOUT_CSV = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_layout_analysis.csv")
OUTPUT_DOSE_CSV = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_dose_analysis.csv")
OUTPUT_SYNERGY_CSV = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_synergy_analysis.csv")
OUTPUT_L2FC_BAR = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_layout_l2fc_barplot.png")
OUTPUT_HEATMAP = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_layout_heatmap.png")
OUTPUT_DOSE_PLOT = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_layout_dose.png")
OUTPUT_SYNERGY_PLOT = os.path.join(OUTPUT_DIR, "alu_te_within_gene_v4_layout_synergy.png")

ALU_OVERLAP_THRESHOLD = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================== 辅助函数 ==========================

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


def count_alu_overlaps(query_intervals, alu_tree, threshold=50):
    """
    v4.0: 计算 query_intervals 中与 Alu 重叠长度 >= threshold 的
    不同 Alu 元素个数。

    返回: int — 不同 Alu 的数量
    """
    if not query_intervals or not alu_tree:
        return 0
    alu_ids = set()
    for qs, qe in query_intervals:
        for alu_iv in alu_tree.overlap(qs, qe + 1):
            ol = min(qe, alu_iv.end - 1) - max(qs, alu_iv.begin) + 1
            if ol >= threshold:
                alu_ids.add(alu_iv.data)
    return len(alu_ids)


def plot_paired_te(paired_df, output_path, region_label, p_value, n_genes):
    """绘制配对箱线图 + 小提琴图"""
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 150

    if not np.isnan(p_value):
        if p_value < 0.001:
            star = "***"
            sig_text = f"Wilcoxon p = {p_value:.2e}"
        elif p_value < 0.01:
            star = "**"
            sig_text = f"Wilcoxon p = {p_value:.4f}"
        elif p_value < 0.05:
            star = "*"
            sig_text = f"Wilcoxon p = {p_value:.4f}"
        else:
            star = "ns"
            sig_text = f"Wilcoxon p = {p_value:.4f} ns"
    else:
        star = ""
        sig_text = "N/A"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1 = axes[0]
    plot_df = paired_df.copy()

    plot_df_sorted = plot_df.copy()
    plot_df_sorted['abs_diff'] = plot_df_sorted['diff'].abs()
    n_show = min(30, len(plot_df_sorted))
    plot_df_sorted = plot_df_sorted.nlargest(n_show, 'abs_diff')
    plot_df_sorted = plot_df_sorted.sort_values('te_median_alu_minus')

    for _, row in plot_df_sorted.iterrows():
        ax1.plot([0, 1], [row['te_median_alu_minus'], row['te_median_alu_plus']],
                 color='grey', alpha=0.35, linewidth=0.7)

    bp_data = [plot_df['te_median_alu_minus'].values, plot_df['te_median_alu_plus'].values]
    bp = ax1.boxplot(bp_data, positions=[0, 1], widths=0.5, patch_artist=True,
                     medianprops={'color': 'black', 'linewidth': 2},
                     flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.4})
    colors = ['#6baed6', '#fd8d3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    np.random.seed(42)
    for i, col in enumerate(['te_median_alu_minus', 'te_median_alu_plus']):
        jitter = np.random.normal(0, 0.04, len(plot_df))
        ax1.scatter(np.full(len(plot_df), i) + jitter, plot_df[col].values,
                    alpha=0.5, s=20, color=colors[i], edgecolors='white', linewidth=0.5, zorder=5)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Alu- (u0_c0_t0)', 'Alu+'])
    ax1.set_ylabel('Median TE per gene')
    ax1.set_title(f'{region_label} — Paired Boxplot (N={n_genes} genes)')
    ax1.text(0.5, 0.95, f'{sig_text} {star}', transform=ax1.transAxes,
             ha='center', fontsize=11, fontweight='bold', va='top')

    ax2 = axes[1]
    long_data = pd.concat([
        pd.DataFrame({'group': 'Alu-', 'TE': plot_df['te_median_alu_minus']}),
        pd.DataFrame({'group': 'Alu+', 'TE': plot_df['te_median_alu_plus']}),
    ])
    sns.violinplot(data=long_data, x='group', y='TE', order=['Alu-', 'Alu+'],
                   palette={'Alu-': '#6baed6', 'Alu+': '#fd8d3c'},
                   ax=ax2, inner='quartile', cut=0)

    top30 = plot_df.nlargest(min(30, len(plot_df)), 'abs_diff') if 'abs_diff' in plot_df else plot_df
    np.random.seed(42)
    for _, row in top30.iterrows():
        jit = np.random.uniform(-0.15, 0.15)
        ax2.plot([0 + jit, 1 + jit], [row['te_median_alu_minus'], row['te_median_alu_plus']],
                 color='grey', alpha=0.3, linewidth=0.6)

    ax2.set_xticklabels(['Alu-', 'Alu+'])
    ax2.set_ylabel('Median TE per gene')
    ax2.set_title(f'{region_label} — Paired Violin (N={n_genes} genes)')
    ax2.text(0.5, 0.95, f'{sig_text} {star}', transform=ax2.transAxes,
             ha='center', fontsize=11, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  {output_path}")


# ========================== Step 1: GTF 解析 ==========================

log("=" * 70)
log("Step 1: GTF 流式解析 — 提取编码转录本的 UTR / CDS 坐标")
log("=" * 70)

t0 = time.time()

tx_info = {}
tx_utr_raw = defaultdict(list)
tx_cds = defaultdict(list)
tx_has_cds = set()

line_count = 0
GTF_ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')

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

        if feature not in ('transcript', 'CDS', 'UTR'):
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
            tx_info[tx_id] = {'gene_id': gid, 'gene_name': gname, 'chrom': chrom, 'strand': strand}
        elif feature == 'CDS':
            tx_cds[tx_id].append((start, end))
            tx_has_cds.add(tx_id)
        elif feature == 'UTR':
            tx_utr_raw[tx_id].append((start, end))

        if line_count % 10000 == 0:
            log(f"  ... {line_count:,} lines, {time.time()-t0:.0f}s")

parsing_time = time.time() - t0
log(f"GTF: {line_count/1e6:.1f}M lines, {parsing_time:.0f}s")
log(f"  transcripts: {len(tx_info)}, w/ CDS: {len(tx_has_cds)}")

cds_tx_ids = set(tx_info.keys()) & tx_has_cds
log(f"  coding retained: {len(cds_tx_ids)} ({(len(cds_tx_ids)/max(len(tx_info),1)*100):.1f}%)")

log("  assigning 5'UTR/3'UTR...")
tx_utr5 = {}
tx_utr3 = {}
tx_cds_coords = {}

for tx_id in cds_tx_ids:
    info = tx_info[tx_id]
    strand = info['strand']
    cds_merged = merge_intervals(tx_cds.get(tx_id, []))
    if not cds_merged:
        continue
    tx_cds_coords[tx_id] = cds_merged
    utr_raw = tx_utr_raw.get(tx_id, [])
    if not utr_raw:
        tx_utr5[tx_id] = []
        tx_utr3[tx_id] = []
        continue
    utr_merged = merge_intervals(utr_raw)
    cds_min = min(s for s, e in cds_merged)
    cds_max = max(e for s, e in cds_merged)
    utr5_list, utr3_list = [], []
    for s, e in utr_merged:
        if strand == '+':
            if e <= cds_min: utr5_list.append((s, e))
            elif s >= cds_max: utr3_list.append((s, e))
        else:
            if s >= cds_max: utr5_list.append((s, e))
            elif e <= cds_min: utr3_list.append((s, e))
    tx_utr5[tx_id] = utr5_list
    tx_utr3[tx_id] = utr3_list

log(f"  w/ 5'UTR: {sum(1 for v in tx_utr5.values() if v)}")
log(f"  w/ 3'UTR: {sum(1 for v in tx_utr3.values() if v)}")
log(f"  Step 1: {time.time()-t0:.0f}s")


# ========================== Step 2: Alu 计数检测（v4.0） ==========================

log("\n" + "=" * 70)
log(f"Step 2: Alu 计数检测 (threshold >= {ALU_OVERLAP_THRESHOLD} bp)")
log("=" * 70)

t1 = time.time()

alu_df = pd.read_csv(ALU_BED_PATH, sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'alu_id', 'score', 'strand'])
alu_trees = {}
for chrom, grp in alu_df.groupby('chrom'):
    tree = IntervalTree()
    for _, row in grp.iterrows():
        tree.addi(row['start'] + 1, row['end'] + 1, row['alu_id'])
    alu_trees[chrom] = tree

log(f"  Alu elements: {len(alu_df)}, chromosomes: {len(alu_trees)}")

# v4.0: 用 count_alu_overlaps 替代 has_alu_overlap
alu_counts = {}  # tx_id -> {'utr5': int, 'cds': int, 'utr3': int}
checked = 0
has_any = 0

for tx_id in cds_tx_ids:
    info = tx_info[tx_id]
    chrom = info['chrom']
    alu_tree = alu_trees.get(chrom, None)

    n_utr5 = count_alu_overlaps(tx_utr5.get(tx_id, []), alu_tree, ALU_OVERLAP_THRESHOLD)
    n_cds = count_alu_overlaps(tx_cds_coords.get(tx_id, []), alu_tree, ALU_OVERLAP_THRESHOLD)
    n_utr3 = count_alu_overlaps(tx_utr3.get(tx_id, []), alu_tree, ALU_OVERLAP_THRESHOLD)

    alu_counts[tx_id] = {'utr5': n_utr5, 'cds': n_cds, 'utr3': n_utr3}
    checked += 1
    if n_utr5 + n_cds + n_utr3 > 0:
        has_any += 1
    if checked % 20000 == 0:
        log(f"  ... {checked} transcripts, Alu+: {has_any}")

log(f"  checked: {checked}, any Alu: {has_any} ({has_any/max(checked,1)*100:.1f}%)")

# 构建注释矩阵（v4.0: 包含计数 + layout 指纹）
annot_rows = []
for tx_id in cds_tx_ids:
    cnt = alu_counts.get(tx_id, {'utr5': 0, 'cds': 0, 'utr3': 0})
    info = tx_info[tx_id]
    n_u, n_c, n_t = cnt['utr5'], cnt['cds'], cnt['utr3']
    layout = f"u{n_u}_c{n_c}_t{n_t}"
    annot_rows.append({
        'transcript_id': tx_id,
        'gene_id': info['gene_id'],
        'gene_name': info['gene_name'],
        'chrom': info['chrom'],
        'strand': info['strand'],
        'n_alu_utr5': n_u,
        'n_alu_cds': n_c,
        'n_alu_utr3': n_t,
        'alu_total': n_u + n_c + n_t,
        'has_alu': int(n_u + n_c + n_t > 0),
        'layout': layout,
    })

annot_df = pd.DataFrame(annot_rows)
log(f"\n  annotation matrix: {annot_df.shape}")
log(f"  has_alu==1: {annot_df['has_alu'].sum()}")
log(f"  has_alu==0: {(annot_df['has_alu']==0).sum()}")
log(f"  unique layouts: {annot_df['layout'].nunique()}")

# 打印 top 10 layouts
layout_dist = annot_df['layout'].value_counts().head(10)
log(f"\n  Top 10 layouts:")
for l, c in layout_dist.items():
    log(f"    {l}: {c} transcripts")

log(f"  Step 2: {time.time()-t1:.0f}s")


# ========================== Step 3: TE 数据对齐 ==========================

log("\n" + "=" * 70)
log("Step 3: TE 数据对齐与极值清洗")
log("=" * 70)

t2 = time.time()

tedd_df = pd.read_csv(TEDD_PATH)

tx_col_candidates = [c for c in tedd_df.columns if 'transcript' in c.lower() or 'tx' in c.lower() or 'enst' in c.lower()]
te_col_candidates = [c for c in tedd_df.columns if c == 'TE' or 'te_' in c.lower() or 'te' == c.lower()]

TX_COL = 'TRANSCRIPT_ID' if 'TRANSCRIPT_ID' in tedd_df.columns else tx_col_candidates[0] if tx_col_candidates else None
TE_COL = 'TE' if 'TE' in tedd_df.columns else te_col_candidates[0] if te_col_candidates else None

if TX_COL is None or TE_COL is None:
    raise ValueError(f"Cannot identify columns: {list(tedd_df.columns)}")

log(f"  transcript_id: '{TX_COL}', TE: '{TE_COL}'")

tedd_sub = tedd_df[[TX_COL, TE_COL]].copy()
tedd_sub.columns = ['transcript_id', 'TE']
tedd_sub['transcript_id'] = tedd_sub['transcript_id'].astype(str).str.split('.').str[0]

n_before = len(tedd_sub)
tedd_sub = tedd_sub.dropna(subset=['TE'])
tedd_sub = tedd_sub[tedd_sub['TE'] > 0]
log(f"  TE filtered: {n_before} -> {len(tedd_sub)} ({n_before-len(tedd_sub)} removed)")

merged = pd.merge(annot_df, tedd_sub, on='transcript_id', how='inner')
log(f"  merged: {len(merged)} transcripts, {merged['gene_id'].nunique()} genes")

alu_plus = merged[merged['has_alu'] == 1]
alu_minus = merged[merged['has_alu'] == 0]
log(f"  Alu+: {len(alu_plus)}, Alu-: {len(alu_minus)}")

# 按 layout 统计
layout_merged_counts = merged['layout'].value_counts()
log(f"  Layouts in merged data: {len(layout_merged_counts)}")
for l, c in layout_merged_counts.head(15).items():
    log(f"    {l}: {c} transcripts")

log(f"  Step 3: {time.time()-t2:.0f}s")


# ========================== Step 4: 基因内中位数聚合 ==========================

log("\n" + "=" * 70)
log("Step 4: 基因内中位数聚合")
log("=" * 70)

t3 = time.time()

paired_data = []
for gene_id, gene_df in merged.groupby('gene_id'):
    alu_plus_vals = gene_df[gene_df['has_alu'] == 1]['TE'].values
    alu_minus_vals = gene_df[gene_df['has_alu'] == 0]['TE'].values
    if len(alu_plus_vals) == 0 or len(alu_minus_vals) == 0:
        continue
    gene_symbol = gene_df['gene_name'].iloc[0]
    paired_data.append({
        'gene_id': gene_id, 'gene_name': gene_symbol,
        'n_alu_plus': len(alu_plus_vals), 'n_alu_minus': len(alu_minus_vals),
        'te_median_alu_plus': np.median(alu_plus_vals),
        'te_median_alu_minus': np.median(alu_minus_vals),
        'diff': np.median(alu_plus_vals) - np.median(alu_minus_vals),
    })

paired_df = pd.DataFrame(paired_data)
paired_df = paired_df[(paired_df['te_median_alu_plus']>0) & (paired_df['te_median_alu_minus']>0)].copy()
log(f"  paired genes: {len(paired_df)}")
log(f"  Step 4: {time.time()-t3:.0f}s")


# ========================== Step 5: Wilcoxon 全区域总览 ==========================

log("\n" + "=" * 70)
log("Step 5: Wilcoxon 符号秩检验（全区域总览）")
log("=" * 70)

t4 = time.time()

if len(paired_df) >= 10:
    w_stat, w_pval = wilcoxon(paired_df['te_median_alu_plus'], paired_df['te_median_alu_minus'], alternative='two-sided')
else:
    w_stat, w_pval = np.nan, np.nan

global_median_plus = paired_df['te_median_alu_plus'].median()
global_median_minus = paired_df['te_median_alu_minus'].median()

log(f"\n  N={len(paired_df)}, Alu+ med={global_median_plus:.4f}, Alu- med={global_median_minus:.4f}")
log(f"  Diff={global_median_plus-global_median_minus:.4f}")
if not np.isnan(w_pval):
    sig_str = "***" if w_pval<0.001 else "**" if w_pval<0.01 else "*" if w_pval<0.05 else "ns"
    log(f"  Wilcoxon p={w_pval:.2e} {sig_str}")

t5 = time.time()
plot_paired_te(paired_df, OUTPUT_VIOLIN, "All exonic regions (Alu+ any vs Alu-)", w_pval, len(paired_df))
log(f"  Step 5: {time.time()-t4:.0f}s (plot: {time.time()-t5:.0f}s)")


# =========================================================================
# Step 6: 组合布局分析（v4.0 全新）
# =========================================================================

log("\n" + "=" * 80)
log("Step 6: 组合式剂量与位置分析 (v4.0)")
log("=" * 80)
log("  Layout fingerprint format: u{N_5UTR}_c{N_CDS}_t{N_3UTR}")
log("  Control: u0_c0_t0 (no Alu anywhere)")
log("=" * 80)

t6 = time.time()

# ---------- 6a: 单布局配对分析 ----------

log("\n  --- 6a: Single-layout paired analysis-V0 ---")

# 所有非零布局（排除 u0_c0_t0）
all_layouts = [l for l in merged['layout'].unique() if l != 'u0_c0_t0']
MIN_PAIRED_GENES = 20  # 最少基因配对数

layout_results = []

for layout in sorted(all_layouts):
    # 筛选该基因同时有目标 layout 和 u0_c0_t0
    layout_paired = []
    for gene_id, gene_df in merged.groupby('gene_id'):
        # 检查基因是否同时包含该 layout 和 control
        gene_layouts = gene_df['layout'].values
        if layout not in gene_layouts or 'u0_c0_t0' not in gene_layouts:
            continue
        layout_te = gene_df[gene_df['layout'] == layout]['TE'].values
        control_te = gene_df[gene_df['layout'] == 'u0_c0_t0']['TE'].values
        if len(layout_te) == 0 or len(control_te) == 0:
            continue
        layout_paired.append({
            'gene_id': gene_id,
            'gene_name': gene_df['gene_name'].iloc[0],
            'n_layout': len(layout_te),
            'n_control': len(control_te),
            'te_median_layout': np.median(layout_te),
            'te_median_control': np.median(control_te),
            'diff': np.median(layout_te) - np.median(control_te),
        })

    if len(layout_paired) < MIN_PAIRED_GENES:
        continue

    layout_df = pd.DataFrame(layout_paired)
    layout_df = layout_df[(layout_df['te_median_layout']>0) & (layout_df['te_median_control']>0)]

    n_pairs = len(layout_df)
    if n_pairs < MIN_PAIRED_GENES:
        continue

    if n_pairs >= 5:
        lw_stat, lw_pval = wilcoxon(layout_df['te_median_layout'], layout_df['te_median_control'], alternative='two-sided')
    else:
        lw_stat, lw_pval = np.nan, np.nan

    l_med_layout = layout_df['te_median_layout'].median()
    l_med_control = layout_df['te_median_control'].median()
    l_diff = l_med_layout - l_med_control
    l_pct = ((l_med_layout - l_med_control) / l_med_control * 100) if l_med_control != 0 else np.nan
    l_l2fc = np.log2(l_med_layout / l_med_control) if l_med_control > 0 and l_med_layout > 0 else np.nan

    # 解析 layout 组件用于后续热图
    parts = layout.split('_')
    n_u = int(parts[0][1:]) if len(parts) >= 3 else 0
    n_c = int(parts[1][1:]) if len(parts) >= 3 else 0
    n_t = int(parts[2][1:]) if len(parts) >= 3 else 0

    sig = "ns"
    if not np.isnan(lw_pval):
        sig = "***" if lw_pval < 0.001 else "**" if lw_pval < 0.01 else "*" if lw_pval < 0.05 else "ns"

    layout_results.append({
        'layout': layout,
        'n_u': n_u, 'n_c': n_c, 'n_t': n_t,
        'n_paired_genes': n_pairs,
        'n_layout_transcripts': int(layout_df['n_layout'].sum()),
        'n_control_transcripts': int(layout_df['n_control'].sum()),
        'median_TE_layout': round(l_med_layout, 4),
        'median_TE_control': round(l_med_control, 4),
        'abs_diff': round(l_diff, 4),
        'pct_change': round(l_pct, 2) if not np.isnan(l_pct) else np.nan,
        'l2fc': round(l_l2fc, 4) if not np.isnan(l_l2fc) else np.nan,
        'Wilcoxon_W': round(lw_stat, 2) if not np.isnan(lw_stat) else 'N/A',
        'Wilcoxon_pval': lw_pval,
        'significance': sig,
    })

    log(f"    {layout:>10} | N_genes={n_pairs:4d} | "
        f"TE_layout={l_med_layout:.3f} TE_ctrl={l_med_control:.3f} | "
        f"L2FC={l_l2fc:+.4f} p={lw_pval:.2e} {sig}")

layout_df_out = pd.DataFrame(layout_results)
layout_df_out.to_csv(OUTPUT_LAYOUT_CSV, index=False, float_format='%.6g')
log(f"\n  Layout analysis-V0 saved: {OUTPUT_LAYOUT_CSV} ({len(layout_df_out)} layouts)")


# ---------- 6b: 剂量效应分析 ----------

log("\n  --- 6b: Dose-response analysis-V0 ---")

# 定义剂量组：每个区域独立的递增剂量
dose_configs = [
    ('5\'UTR',   'n_alu_utr5', [0, 1, 2]),
    ('CDS',      'n_alu_cds',  [0, 1, 2]),
    ('3\'UTR',   'n_alu_utr3', [0, 1, 2]),
]

dose_results = []

for region_label, region_col, doses in dose_configs:
    log(f"    Region: {region_label}")

    # 找到同时具有所有剂量的基因（控制其他区域为0）
    other_cols = [c for c in ['n_alu_utr5', 'n_alu_cds', 'n_alu_utr3'] if c != region_col]

    # 取其他两个区域 Alu 数为 0，目标区域在 doses 中的转录本
    mask = True
    for oc in other_cols:
        mask &= (merged[oc] == 0)
    mask &= (merged[region_col].isin(doses))

    dose_sub = merged[mask].copy()

    # 对每个基因，按剂量分组
    gene_dose_data = defaultdict(dict)
    for gene_id, gene_df in dose_sub.groupby('gene_id'):
        for d in doses:
            vals = gene_df[gene_df[region_col] == d]['TE'].values
            if len(vals) > 0:
                gene_dose_data[gene_id][d] = np.median(vals)

    # 只保留所有剂量都有的基因
    valid_genes = {g for g, v in gene_dose_data.items() if all(d in v for d in doses)}
    log(f"      Genes with all doses {doses}: {len(valid_genes)}")

    if len(valid_genes) < 5:
        log(f"      Too few genes, skipping")
        continue

    # 逐剂量 vs u0_c0_t0（dose=0）配对检验
    for d in doses[1:]:  # skip dose 0
        paired_vals_d = []
        paired_vals_0 = []
        for g in valid_genes:
            paired_vals_d.append(gene_dose_data[g][d])
            paired_vals_0.append(gene_dose_data[g][0])

        if len(paired_vals_d) >= 5:
            dw_stat, dw_pval = wilcoxon(paired_vals_d, paired_vals_0, alternative='two-sided')
            med_d = np.median(paired_vals_d)
            med_0 = np.median(paired_vals_0)
            l2fc_d = np.log2(med_d/med_0) if med_0 > 0 and med_d > 0 else np.nan
            sig = "***" if dw_pval < 0.001 else "**" if dw_pval < 0.01 else "*" if dw_pval < 0.05 else "ns"
        else:
            dw_pval = np.nan
            med_d = np.median(paired_vals_d)
            med_0 = np.median(paired_vals_0)
            l2fc_d = np.nan
            sig = "N/A"

        log(f"      {region_label} dose={d} vs dose=0: "
            f"N={len(paired_vals_d)} med_d={med_d:.3f} med_0={med_0:.3f} "
            f"L2FC={l2fc_d:.4f} p={dw_pval:.2e} {sig}")

        dose_results.append({
            'region': region_label,
            'dose': d,
            'n_paired_genes': len(paired_vals_d),
            'median_TE_dose': round(med_d, 4),
            'median_TE_0': round(med_0, 4),
            'l2fc': round(l2fc_d, 4) if not np.isnan(l2fc_d) else np.nan,
            'Wilcoxon_pval': dw_pval,
            'significance': sig,
        })

    # Kruskal-Wallis 全局检验（所有剂量组）
    kw_groups = []
    for g in valid_genes:
        for d in doses:
            kw_groups.append({'gene_id': g, 'dose': d, 'te': gene_dose_data[g][d]})
    kw_df = pd.DataFrame(kw_groups)
    if len(kw_df) > 0:
        kw_data = [kw_df[kw_df['dose']==d]['te'].values for d in doses]
        kw_data = [d for d in kw_data if len(d) > 0]
        if len(kw_data) > 1 and all(len(d) >= 3 for d in kw_data):
            kw_stat, kw_pval = kruskal(*kw_data)
            log(f"      Kruskal-Wallis: H={kw_stat:.2f}, p={kw_pval:.4f}")

dose_df_out = pd.DataFrame(dose_results)
dose_df_out.to_csv(OUTPUT_DOSE_CSV, index=False, float_format='%.6g')
log(f"\n  Dose analysis-V0 saved: {OUTPUT_DOSE_CSV}")


# ---------- 6c: 协同效应分析 ----------

log("\n  --- 6c: Synergy analysis-V0 ---")

# 寻找同时有 u1_c0_t1, u1_c0_t0, u0_c0_t1 的基因
synergy_layouts = ['u1_c0_t1', 'u1_c0_t0', 'u0_c0_t1']
control_layout = 'u0_c0_t0'

synergy_candidates = []
for gene_id, gene_df in merged.groupby('gene_id'):
    gene_layouts = set(gene_df['layout'].values)
    if all(l in gene_layouts for l in synergy_layouts + [control_layout]):
        te_by_layout = {}
        for l in synergy_layouts + [control_layout]:
            te_by_layout[l] = np.median(gene_df[gene_df['layout'] == l]['TE'].values)
        synergy_candidates.append({
            'gene_id': gene_id,
            'gene_name': gene_df['gene_name'].iloc[0],
            **{f'te_{l}': te_by_layout[l] for l in synergy_layouts + [control_layout]},
        })

log(f"    Genes with all synergy layouts: {len(synergy_candidates)}")

synergy_results = []
if len(synergy_candidates) >= 10:
    syn_df = pd.DataFrame(synergy_candidates)

    # 计算每个基因的 ΔTE
    for l in synergy_layouts:
        syn_df[f'delta_{l}'] = syn_df[f'te_{l}'] - syn_df[f'te_{control_layout}']

    # 计算预期和实际
    syn_df['expected_additive'] = syn_df['delta_u1_c0_t0'] + syn_df['delta_u0_c0_t1']
    syn_df['actual_combined'] = syn_df['delta_u1_c0_t1']
    syn_df['synergy'] = syn_df['actual_combined'] - syn_df['expected_additive']

    # Wilcoxon 检验：协同效应是否 > 0
    syn_vals = syn_df['synergy'].values
    if len(syn_vals) >= 5:
        # 单侧检验：协同 > 0
        sy_stat, sy_pval = wilcoxon(syn_vals, alternative='two-sided')
        sy_med = np.median(syn_vals)
        sy_mean = np.mean(syn_vals)
        sy_sem = np.std(syn_vals) / np.sqrt(len(syn_vals))
    else:
        sy_stat, sy_pval = np.nan, np.nan
        sy_med = sy_mean = sy_sem = np.nan

    # 统计每个独立效应
    for l in ['u1_c0_t0', 'u0_c0_t1', 'u1_c0_t1']:
        dw_stat, dw_pval = wilcoxon(syn_df[f'te_{l}'], syn_df[f'te_{control_layout}'], alternative='two-sided')
        med_l = syn_df[f'te_{l}'].median()
        med_c = syn_df[f'te_{control_layout}'].median()
        log(f"      {l:>10}: median TE={med_l:.3f} vs control={med_c:.3f}, "
            f"Δ={med_l-med_c:+.4f}, p={dw_pval:.2e}")

    log(f"      {'Expected additive':>10}: {syn_df['expected_additive'].median():+.4f}")
    log(f"      {'Actual combined':>10}: {syn_df['actual_combined'].median():+.4f}")
    log(f"      {'Synergy (ΔΔ)':>10}: {sy_med:+.4f} (mean={sy_mean:+.4f}±{sy_sem:.4f})")
    log(f"      Wilcoxon synergy>0: p={sy_pval:.4f}")

    synergy_results.append({
        'comparison': 'u1_c0_t1 vs (u1_c0_t0 + u0_c0_t1)',
        'n_genes': len(syn_df),
        'median_delta_u1c0t0': round(syn_df['delta_u1_c0_t0'].median(), 4),
        'median_delta_u0c0t1': round(syn_df['delta_u0_c0_t1'].median(), 4),
        'median_expected_additive': round(syn_df['expected_additive'].median(), 4),
        'median_actual_u1c0t1': round(syn_df['delta_u1_c0_t1'].median(), 4),
        'median_synergy': round(sy_med, 4),
        'mean_synergy': round(sy_mean, 4),
        'sem_synergy': round(sy_sem, 4) if not np.isnan(sy_sem) else np.nan,
        'Wilcoxon_pval_synergy': sy_pval,
        'n_synergy_positive': int((syn_df['synergy'] > 0).sum()),
        'n_synergy_negative': int((syn_df['synergy'] < 0).sum()),
        'pct_synergy_positive': round((syn_df['synergy'] > 0).mean() * 100, 1),
    })

    # 还检查其他组合（如 u1_c1_t0, u0_c1_t1 等）
    # 寻找 5'UTR + CDS 组合
    other_combos = [
        (['u1_c1_t0', 'u1_c0_t0', 'u0_c1_t0'], 'u1_c1_t0'),
        (['u0_c1_t1', 'u0_c1_t0', 'u0_c0_t1'], 'u0_c1_t1'),
        (['u1_c1_t1', 'u1_c0_t0', 'u0_c1_t0', 'u0_c0_t1'], 'u1_c1_t1'),
    ]
    for combo_layouts, combo_label in other_combos:
        candidate_genes = []
        for gene_id, gene_df in merged.groupby('gene_id'):
            gene_layouts = set(gene_df['layout'].values)
            if all(l in gene_layouts for l in combo_layouts + [control_layout]):
                te_dict = {}
                for l in combo_layouts + [control_layout]:
                    te_dict[l] = np.median(gene_df[gene_df['layout'] == l]['TE'].values)
                candidate_genes.append(te_dict)

        if len(candidate_genes) < 10:
            continue

        cg_df = pd.DataFrame(candidate_genes)
        single_effects = [l for l in combo_layouts if l != combo_label]
        # 注意：candidate_genes 的 dict 直接以 layout 为 key（如 'u1_c1_t0'）
        cg_df['expected'] = sum(cg_df[l] - cg_df[control_layout] for l in single_effects)
        cg_df['actual'] = cg_df[combo_label] - cg_df[control_layout]
        cg_df['synergy'] = cg_df['actual'] - cg_df['expected']

        sy_vals = cg_df['synergy'].values
        if len(sy_vals) >= 5:
            _, sy_p = wilcoxon(sy_vals, alternative='two-sided')
        else:
            sy_p = np.nan

        synergy_results.append({
            'comparison': f'{combo_label} vs sum of singles',
            'n_genes': len(cg_df),
            'median_delta_singles': '+'.join([f"{round(cg_df[l].median()-cg_df[control_layout].median(), 4)}" for l in single_effects]),
            'median_expected_additive': round(cg_df['expected'].median(), 4),
            'median_actual_combined': round(cg_df['actual'].median(), 4),
            'median_synergy': round(cg_df['synergy'].median(), 4),
            'mean_synergy': round(cg_df['synergy'].mean(), 4),
            'sem_synergy': round(np.std(cg_df['synergy'].values)/np.sqrt(len(cg_df)), 4),
            'Wilcoxon_pval_synergy': sy_p,
            'n_synergy_positive': int((cg_df['synergy'] > 0).sum()),
            'n_synergy_negative': int((cg_df['synergy'] < 0).sum()),
            'pct_synergy_positive': round((cg_df['synergy'] > 0).mean() * 100, 1),
        })

        log(f"    {combo_label:>12}: N={len(cg_df)} "
            f"actual={cg_df['actual'].median():+.4f} expected={cg_df['expected'].median():+.4f} "
            f"synergy={cg_df['synergy'].median():+.4f} p={sy_p:.4f}")

else:
    log(f"    Too few genes ({len(synergy_candidates)} < 10), skipping synergy analysis-V0")

synergy_df_out = pd.DataFrame(synergy_results)
synergy_df_out.to_csv(OUTPUT_SYNERGY_CSV, index=False, float_format='%.6g')
log(f"\n  Synergy analysis-V0 saved: {OUTPUT_SYNERGY_CSV}")


# ========================== 可视化 ==========================

log("\n  --- Visualizations ---")

if len(layout_df_out) > 0:
    # ---------- 6d: L2FC 柱状图 ----------
    log("    Plotting L2FC barplot...")

    plot_data = layout_df_out.copy()
    plot_data = plot_data.sort_values('l2fc')

    fig, ax = plt.subplots(figsize=(max(8, len(plot_data)*0.5), 5))
    colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in plot_data['l2fc']]
    alpha_vals = [0.4 if s == 'ns' else 0.9 for s in plot_data['significance']]
    colors_bar = [mcolors.to_rgba(c, a) for c, a in zip(
        ['#e74c3c' if v < 0 else '#2ecc71' for v in plot_data['l2fc']], alpha_vals)]

    bars = ax.bar(range(len(plot_data)), plot_data['l2fc'].values,
                  color=colors_bar, edgecolor='grey', linewidth=0.5)

    # 标注显著性
    for i, (_, row) in enumerate(plot_data.iterrows()):
        if row['significance'] not in ('ns', 'N/A'):
            ax.text(i, row['l2fc'] + (0.02 if row['l2fc'] >= 0 else -0.08),
                    row['significance'], ha='center', fontsize=8, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_data['layout'].values, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Log2 Fold Change (TE layout / TE control)')
    ax.set_title('Alu Layout Effect on TE (within-gene paired)')
    ax.text(0.99, 0.95, f'N layouts = {len(plot_data)} (each > {MIN_PAIRED_GENES} paired genes)',
            transform=ax.transAxes, ha='right', va='top', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_L2FC_BAR, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"    {OUTPUT_L2FC_BAR}")

    # ---------- 6e: 组合热图 ----------
    log("    Plotting layout heatmap...")

    # 构建热图数据：按 5'UTR count × 3'UTR count 分组的 L2FC（固定 CDS=0）
    cds0_mask = (layout_df_out['n_c'] == 0)
    heat_layouts = layout_df_out[cds0_mask].copy()

    if len(heat_layouts) > 0:
        # 找到所有 n_u 和 n_t 的唯一值
        u_vals = sorted(heat_layouts['n_u'].unique())
        t_vals = sorted(heat_layouts['n_t'].unique())

        heat_matrix = np.full((len(t_vals), len(u_vals)), np.nan)
        for _, row in heat_layouts.iterrows():
            ui = u_vals.index(row['n_u'])
            ti = t_vals.index(row['n_t'])
            heat_matrix[ti, ui] = row['l2fc']

        fig, ax = plt.subplots(figsize=(max(5, len(u_vals)*0.8), max(4, len(t_vals)*0.8)))
        vmax = max(abs(np.nanmax(heat_matrix)), abs(np.nanmin(heat_matrix)), 0.1)
        cmap = sns.diverging_palette(240, 10, as_cmap=True)

        im = ax.imshow(heat_matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

        for i in range(len(t_vals)):
            for j in range(len(u_vals)):
                val = heat_matrix[i, j]
                if not np.isnan(val):
                    # 查找对应的显著性
                    matching = heat_layouts[(heat_layouts['n_u']==u_vals[j]) & (heat_layouts['n_t']==t_vals[i])]
                    sig_label = matching['significance'].values[0] if len(matching) > 0 else ''
                    text_color = 'white' if abs(val) > vmax*0.6 else 'black'
                    ax.text(j, i, f'{val:.3f}\n{sig_label}', ha='center', va='center',
                            fontsize=7, color=text_color, fontweight='bold')

        ax.set_xticks(range(len(u_vals)))
        ax.set_yticks(range(len(t_vals)))
        ax.set_xticklabels([str(v) for v in u_vals])
        ax.set_yticklabels([str(v) for v in t_vals])
        ax.set_xlabel('# Alu in 5\'UTR')
        ax.set_ylabel('# Alu in 3\'UTR')
        ax.set_title('TE L2FC by Alu Count (CDS=0)')

        plt.colorbar(im, ax=ax, label='L2FC', shrink=0.8)
        plt.tight_layout()
        plt.savefig(OUTPUT_HEATMAP, dpi=150, bbox_inches='tight')
        plt.close()
        log(f"    {OUTPUT_HEATMAP}")
    else:
        log("    No CDS=0 layouts, skipping heatmap")

    # ---------- 6f: 剂量效应图 ----------
    log("    Plotting dose-response...")

    if len(dose_results) > 0:
        dose_plot_df = pd.DataFrame(dose_results)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        region_colors = {'5\'UTR': '#e41a1c', 'CDS': '#377eb8', '3\'UTR': '#4daf4a'}

        for idx, (region_label, region_col, doses) in enumerate(dose_configs):
            ax = axes[idx]
            region_doses = dose_plot_df[dose_plot_df['region'] == region_label]
            if len(region_doses) == 0:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(region_label)
                continue

            x_vals = [0] + list(region_doses['dose'].values)
            y_vals = [0] + list(region_doses['l2fc'].values)
            sigs = [''] + list(region_doses['significance'].values)

            ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=8,
                    color=region_colors.get(region_label, '#333'))
            ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8)

            for xi, yi, si in zip(x_vals, y_vals, sigs):
                if si not in ('', 'ns', 'N/A'):
                    ax.annotate(si, (xi, yi), textcoords="offset points",
                                xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

            ax.set_xlabel(f'# Alu in {region_label}')
            ax.set_ylabel('L2FC vs dose=0')
            ax.set_title(f'{region_label} dose response')
            ax.set_xticks(range(max(doses)+1))

        plt.tight_layout()
        plt.savefig(OUTPUT_DOSE_PLOT, dpi=150, bbox_inches='tight')
        plt.close()
        log(f"    {OUTPUT_DOSE_PLOT}")

    # ---------- 6g: 协同效应图 ----------
    log("    Plotting synergy...")

    if len(synergy_results) > 0 and len(synergy_candidates) >= 10:
        syn_df = pd.DataFrame(synergy_candidates)
        for l in synergy_layouts:
            syn_df[f'delta_{l}'] = syn_df[f'te_{l}'] - syn_df[f'te_{control_layout}']
        syn_df['expected'] = syn_df['delta_u1_c0_t0'] + syn_df['delta_u0_c0_t1']
        syn_df['actual'] = syn_df['delta_u1_c0_t1']
        syn_df['synergy'] = syn_df['actual'] - syn_df['expected']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 左图: 单个效应 vs 组合效应
        ax1 = axes[0]
        plot_positions = [0, 1, 2, 3]
        labels = ['u1_c0_t0\nalone', 'u0_c0_t1\nalone', 'Expected\n(sum)', 'Actual\nu1_c0_t1']
        data_to_plot = [
            syn_df['delta_u1_c0_t0'].values,
            syn_df['delta_u0_c0_t1'].values,
            syn_df['expected'].values,
            syn_df['actual'].values,
        ]
        bp = ax1.boxplot(data_to_plot, positions=plot_positions, widths=0.5, patch_artist=True)
        colors_bp = ['#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax1.set_xticks(plot_positions)
        ax1.set_xticklabels(labels, fontsize=8)
        ax1.axhline(y=0, color='grey', linestyle='--', linewidth=0.8)
        ax1.set_ylabel('ΔTE (vs u0_c0_t0)')
        ax1.set_title('Individual vs Combined Effects')

        # 中图: 协同效应分布
        ax2 = axes[1]
        sns.histplot(syn_df['synergy'].values, bins=20, kde=True, ax=ax2, color='#984ea3')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
        med_syn = syn_df['synergy'].median()
        ax2.axvline(x=med_syn, color='blue', linestyle='-', linewidth=1, alpha=0.7)
        ax2.text(med_syn, ax2.get_ylim()[1]*0.9, f'median={med_syn:.3f}',
                 ha='left' if med_syn > 0 else 'right', fontsize=9)
        ax2.set_xlabel('Synergy (ΔΔ)')
        ax2.set_title(f'Synergy Distribution\n(N={len(syn_df)} genes)')

        # 右图: 实际 vs 预期散点
        ax3 = axes[2]
        ax3.scatter(syn_df['expected'], syn_df['actual'], alpha=0.5, s=20, c='#333')
        lims = [min(ax3.get_xlim()[0], ax3.get_ylim()[0]),
                max(ax3.get_xlim()[1], ax3.get_ylim()[1])]
        ax3.plot(lims, lims, 'r--', linewidth=1, alpha=0.7, label='y=x (additive)')
        ax3.set_xlabel('Expected combined ΔTE')
        ax3.set_ylabel('Actual combined ΔTE')
        ax3.set_title(f'Synergy: {synergy_results[0]["pct_synergy_positive"]}% positive')
        ax3.legend(fontsize=8)
        ax3.axis('square')

        plt.tight_layout()
        plt.savefig(OUTPUT_SYNERGY_PLOT, dpi=150, bbox_inches='tight')
        plt.close()
        log(f"    {OUTPUT_SYNERGY_PLOT}")


# ========================== 输出结果 ==========================

log("\n" + "=" * 80)
log("Saving summary results")
log("=" * 80)

summary_rows = []
if len(paired_df) >= 10:
    main_w, main_p = wilcoxon(paired_df['te_median_alu_plus'], paired_df['te_median_alu_minus'], alternative='two-sided')
else:
    main_w, main_p = np.nan, np.nan

main_star = "***" if (not np.isnan(main_p) and main_p<0.001) else "**" if (not np.isnan(main_p) and main_p<0.01) else "*" if (not np.isnan(main_p) and main_p<0.05) else "ns" if not np.isnan(main_p) else "N/A"
main_abs_diff = global_median_plus - global_median_minus
main_pct = (main_abs_diff/global_median_minus*100) if global_median_minus!=0 else np.nan
main_l2fc = np.log2(global_median_plus/global_median_minus) if global_median_minus>0 and global_median_plus>0 else np.nan

summary_rows.append({
    'region': 'All exonic', 'n_paired_genes': len(paired_df),
    'median_TE_alu_plus': round(global_median_plus, 4),
    'median_TE_alu_minus': round(global_median_minus, 4),
    'abs_diff': round(main_abs_diff, 4),
    'pct_change': round(main_pct, 2) if not np.isnan(main_pct) else np.nan,
    'l2fc': round(main_l2fc, 4) if not np.isnan(main_l2fc) else np.nan,
    'Wilcoxon_W': round(main_w, 2) if not np.isnan(main_w) else 'N/A',
    'Wilcoxon_pval': main_p, 'significance': main_star,
})

# 也加入 top 布局结果
for _, row in layout_df_out.head(10).iterrows():
    summary_rows.append({
        'region': f"layout_{row['layout']}",
        'n_paired_genes': row['n_paired_genes'],
        'median_TE_alu_plus': row['median_TE_layout'],
        'median_TE_alu_minus': row['median_TE_control'],
        'abs_diff': row['abs_diff'],
        'pct_change': row['pct_change'],
        'l2fc': row['l2fc'],
        'Wilcoxon_W': row['Wilcoxon_W'],
        'Wilcoxon_pval': row['Wilcoxon_pval'],
        'significance': row['significance'],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_CSV, index=False, float_format='%.6g')
log(f"  Results: {OUTPUT_CSV}")

paired_df.to_csv(OUTPUT_PAIRED_GENES, index=False, float_format='%.6g')
log(f"  Gene-level: {OUTPUT_PAIRED_GENES}")


# ========================== 打印最终汇总 ==========================

log("\n" + "=" * 100)
log("v4.0 最终分析报告 — 组合式剂量与位置分析")
log("=" * 100)

log("\n  ┌─ v3.2 风格对比（Alu+ vs Alu- all）")
for _, row in summary_df[summary_df['region']=='All exonic'].iterrows():
    log(f"  └─ N={row['n_paired_genes']}  "
        f"TE+={row['median_TE_alu_plus']}  TE-={row['median_TE_alu_minus']}  "
        f"L2FC={row['l2fc']}  p={row['Wilcoxon_pval']:.2e} {row['significance']}")

log(f"\n  ┌─ v4.0 组合布局分析 ({len(layout_df_out)} layouts > {MIN_PAIRED_GENES} paired genes)")
for _, row in layout_df_out.iterrows():
    log(f"  ├─ {row['layout']:>10} | N={row['n_paired_genes']:4d} | "
        f"TE={row['median_TE_layout']:.3f} vs {row['median_TE_control']:.3f} | "
        f"L2FC={row['l2fc']:+.4f} | {row['significance']}")

if len(dose_results) > 0:
    log(f"\n  ┌─ 剂量效应分析")
    for _, row in pd.DataFrame(dose_results).iterrows():
        log(f"  ├─ {row['region']:>5} dose={row['dose']} | "
            f"L2FC={row['l2fc']:+.4f} | N={row['n_paired_genes']} | {row['significance']}")

if len(synergy_results) > 0:
    log(f"\n  ┌─ 协同效应分析")
    for _, row in synergy_df_out.iterrows():
        log(f"  ├─ {row['comparison'][:40]:40s} | "
            f"synergy={row['median_synergy']:+.4f} | "
            f"p={row['Wilcoxon_pval_synergy']:.4f} | "
            f"{row['pct_synergy_positive']}% positive")

log(f"\n总运行耗时: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} 分钟)")
log("v4.0 分析完成！")
log("=" * 100)
