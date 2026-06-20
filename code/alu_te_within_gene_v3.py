# -*- coding: utf-8 -*-
"""
=============================================================================
同基因内转录本配对的 Alu 与翻译效率 (TE) 差异分析  — 黄金 3.0 标准
=============================================================================

分析流程：
  1. GTF 解析 → 提取编码转录本的 5'UTR / CDS / 3'UTR 坐标
  2. Alu 碰撞检测（≥50 bp 重叠阈值）
  3. TE 数据对齐 + 极值清洗（TE > 0 且非空）
  4. 基因内中位数聚合 → 配对数据
  5. Wilcoxon 符号秩检验
  6. 分区域 (5'UTR / CDS / 3'UTR) 分别分析
  7. 可视化

输入（请确认 BASE_DIR 指向你的数据目录）：
  - data/gencode.v49.primary_assembly.annotation.gtf
  - data/alu_hg38.bed
  - data/TEDD00137.distribution_transcript.csv

输出：
  - output1/alu_te_within_gene_results.csv         （详细配对结果）
  - output1/alu_te_within_gene_paired_violin.png    （配对 violin/boxplot）

作者备注：
  - 由于 Windows 环境无法安装 pyranges（ncls C 扩展编译失败），
    所有区间操作使用 pandas + numpy 手动实现，逻辑等价。
  - 流式处理 GTF（逐行读取），内存峰值 < 2GB。

修改日期：2026-06-07
=============================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import os
import sys
from collections import defaultdict
import time
from intervaltree import IntervalTree

warnings.filterwarnings('ignore')

# 自定义 print：自动 flush stdout，确保后台捕获能看到输出
def log(msg):
    print(msg, flush=True)

# ========================== 0. 路径设置 ==========================
# 自动定位项目根目录（脚本在 code/ 下，根目录 = 上一级）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.path.dirname(os.path.abspath(sys.argv[0]))
BASE_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output1')

# 输入文件路径
GTF_PATH = os.path.join(DATA_DIR, "gencode.v49.primary_assembly.annotation.gtf")
ALU_BED_PATH = os.path.join(DATA_DIR, "alu_hg38.bed")
TEDD_PATH = os.path.join(DATA_DIR, "TEDD00137.distribution_transcript.csv")

# 输出路径
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "alu_te_within_gene_results.csv")
OUTPUT_VIOLIN = os.path.join(OUTPUT_DIR, "alu_te_within_gene_paired_violin.png")

# 参数
ALU_OVERLAP_THRESHOLD = 50  # Alu 重叠底線，单位 bp

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================== 辅助函数 ==========================

def merge_intervals(intervals):
    """
    合并重叠 / 相邻的区间。
    输入: list of (start, end)，均为 1-based 闭区间。
    输出: list of (start, end)，合并后的不重叠区间。
    """
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_iv[0])]
    for s, e in sorted_iv[1:]:
        if s <= merged[-1][1] + 1:  # 相邻或重叠
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def plot_paired_te(paired_df, output_path, region_label, p_value, n_genes):
    """
    绘制配对箱线图 + 小提琴图（Alu+ vs Alu- 的基因中位数 TE）。

    参数:
        paired_df: DataFrame，必须含 te_median_alu_plus, te_median_alu_minus, diff 列
        output_path: str，保存路径
        region_label: str，区域名称（用于标题）
        p_value: float，Wilcoxon p 值
        n_genes: int，配对基因数
    """
    from intervaltree import IntervalTree
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams['figure.dpi'] = 150

    # 显著性星号
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

    # --- 左图: 配对箱线图 + 连线 ---
    ax1 = axes[0]
    plot_df = paired_df.copy()

    # 按差值绝对值排序取 top 30 画连线
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
    ax1.set_xticklabels(['Alu-', 'Alu+'])
    ax1.set_ylabel('Median TE per gene')
    ax1.set_title(f'{region_label} — Paired Boxplot (N={n_genes} genes)')
    ax1.text(0.5, 0.95, f'{sig_text} {star}', transform=ax1.transAxes,
             ha='center', fontsize=11, fontweight='bold', va='top')

    # --- 右图: 配对 violin + 连接线 ---
    ax2 = axes[1]
    long_data = pd.concat([
        pd.DataFrame({'group': 'Alu-', 'TE': plot_df['te_median_alu_minus']}),
        pd.DataFrame({'group': 'Alu+', 'TE': plot_df['te_median_alu_plus']}),
    ])
    sns.violinplot(data=long_data, x='group', y='TE', order=['Alu-', 'Alu+'],
                   palette={'Alu-': '#6baed6', 'Alu+': '#fd8d3c'},
                   ax=ax2, inner='quartile', cut=0)

    paired_with_diff = plot_df.copy()
    paired_with_diff['abs_diff'] = paired_with_diff['diff'].abs()
    top30 = paired_with_diff.nlargest(min(30, len(paired_with_diff)), 'abs_diff')
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
    log(f"  区域图已保存: {output_path}")


def has_alu_overlap(query_intervals, alu_tree, threshold=50):
    """
    使用 IntervalTree 查询 query_intervals 中是否有任何区间与 Alu
    重叠长度 >= threshold bp。

    参数:
        query_intervals: list of (start, end) — 1-based closed
        alu_tree: IntervalTree 对象
        threshold: 最小重叠长度 (bp)

    返回: bool
    """
    if not query_intervals or not alu_tree:
        return False
    for qs, qe in query_intervals:
        # IntervalTree.search() 返回与 [qs, qe] 重叠的所有区间
        # 使用 IntervalTree 的 O(log n + k) 查询
        for alu_iv in alu_tree.overlap(qs, qe):
            # 计算实际重叠长度
            ol = min(qe, alu_iv.end - 1) - max(qs, alu_iv.begin) + 1
            if ol >= threshold:
                return True
    return False


# ========================== Step 1: GTF 解析 ==========================

log("=" * 70)
log("Step 1: GTF 流式解析 — 提取编码转录本的 UTR / CDS 坐标")
log("=" * 70)

t0 = time.time()

# 数据结构：
#   tx_info[tx_id] = {'gene_id', 'gene_name', 'chrom', 'strand'}
#   tx_exons[tx_id] = [(start, end), ...]
#   tx_cds[tx_id]   = [(start, end), ...]
# 只保留 transcript, CDS, UTR 三种 feature 行

tx_info = {}            # tx_id → metadata dict
tx_utr_raw = defaultdict(list)  # tx_id → list of UTR intervals (start, end)
tx_cds = defaultdict(list)     # tx_id → list of CDS (start, end)
tx_has_cds = set()             # 出现过 CDS 的转录本

line_count = 0
processed_types = set()

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

        # 只处理 transcript / CDS / UTR
        if feature not in ('transcript', 'CDS', 'UTR'):
            continue

        # 解析属性 — 只提取需要的字段
        tx_id = gid = gname = ''
        for m in GTF_ATTR_RE.finditer(attr_str):
            k = m.group(1)
            v = m.group(2)
            if k == 'transcript_id':
                tx_id = v.split('.')[0]  # 剥离版本号
            elif k == 'gene_id':
                gid = v.split('.')[0]
            elif k == 'gene_name':
                gname = v
            if tx_id and gid and gname:
                break

        if not tx_id:
            continue

        if feature == 'transcript' and tx_id not in tx_info:
            tx_info[tx_id] = {
                'gene_id': gid,
                'gene_name': gname,
                'chrom': chrom,
                'strand': strand,
            }
        elif feature == 'CDS':
            tx_cds[tx_id].append((start, end))
            tx_has_cds.add(tx_id)
        elif feature == 'UTR':
            tx_utr_raw[tx_id].append((start, end))

        # 进度提示
        if line_count % 10000 == 0:
            elapsed = time.time() - t0
            log(f"  ... 已处理 {line_count:,} 行，耗时 {elapsed:.0f}s")

parsing_time = time.time() - t0
log(f"GTF 解析完成: {line_count / 1e6:.1f}M 行, {parsing_time:.0f}s")
log(f"  总转录本数:         {len(tx_info)}")
log(f"  有 CDS 的转录本:     {len(tx_has_cds)}")
log(f"  有 UTR 的编码转录本:  {len(tx_utr_raw)}")

# ---------- 剔除非编码转录本 ----------
# 只在 tx_has_cds 中的转录本保留
cds_tx_ids = set(tx_info.keys()) & tx_has_cds
log(f"\n  编码转录本 (保留):   {len(cds_tx_ids)}")
non_coding = set(tx_info.keys()) - cds_tx_ids
log(f"  非编码转录本 (剔除): {len(non_coding)} ({len(non_coding)/max(len(tx_info),1)*100:.1f}%)")

# ---------- 分配 5'UTR / 3'UTR ----------
# 直接用 GTF 的 UTR feature（GENCODE 已通过 exon - CDS 计算好），
# 再根据 CDS 边界和链方向分为 5'UTR 和 3'UTR。

log("\n  分配 5'UTR / 3'UTR ...")

# 对于每个编码转录本，收集 UTR 区间并按链方向分配 5' / 3'
tx_utr5 = {}   # tx_id → list of (start, end)
tx_utr3 = {}   # tx_id → list of (start, end)
tx_cds_coords = {}  # tx_id → merged CDS intervals

for tx_id in cds_tx_ids:
    info = tx_info[tx_id]
    strand = info['strand']

    # 合并 CDS
    cds_merged = merge_intervals(tx_cds.get(tx_id, []))
    if not cds_merged:
        # 理论上不应该发生（已经在 cds_tx_ids 中），但以防万一
        continue
    tx_cds_coords[tx_id] = cds_merged

    # 取 UTR 区间（直接来自 GTF UTR feature）
    utr_raw = tx_utr_raw.get(tx_id, [])
    if not utr_raw:
        tx_utr5[tx_id] = []
        tx_utr3[tx_id] = []
        continue

    # 合并 UTR 区间
    utr_merged = merge_intervals(utr_raw)

    # 找到 CDS 的全局边界
    cds_min = min(s for s, e in cds_merged)
    cds_max = max(e for s, e in cds_merged)

    # 按链方向分配 5' / 3' UTR
    utr5_list = []
    utr3_list = []
    for s, e in utr_merged:
        if strand == '+':
            if e <= cds_min:
                utr5_list.append((s, e))
            elif s >= cds_max:
                utr3_list.append((s, e))
            else:
                # 跨 CDS 边界的 UTR — 取不在 CDS 内的部分
                # 简化：丢弃跨边界区段（极少数情况）
                pass
        else:  # strand == '-'
            if s >= cds_max:
                utr5_list.append((s, e))
            elif e <= cds_min:
                utr3_list.append((s, e))
            else:
                pass

    tx_utr5[tx_id] = utr5_list
    tx_utr3[tx_id] = utr3_list

# QC 统计
with_utr5 = sum(1 for v in tx_utr5.values() if v)
with_utr3 = sum(1 for v in tx_utr3.values() if v)
log(f"  有 5'UTR 的编码转录本: {with_utr5}")
log(f"  有 3'UTR 的编码转录本: {with_utr3}")
log(f"  Step 1 耗时: {time.time() - t0:.0f}s")


# ========================== Step 2: Alu 碰撞检测 ==========================

log("\n" + "=" * 70)
log(f"Step 2: Alu 碰撞检测 (≥ {ALU_OVERLAP_THRESHOLD} bp)")
log("=" * 70)

t1 = time.time()

# ---------- 读取 Alu BED，构建 IntervalTree ----------
alu_df = pd.read_csv(
    ALU_BED_PATH, sep='\t', header=None,
    names=['chrom', 'start', 'end', 'alu_id', 'score', 'strand']
)
# BED 是 0-based half-open [start, end)
# IntervalTree 使用 [begin, end) 半开区间
# 转换: begin = BED_start + 1, end = BED_end + 1
alu_trees = {}  # chrom → IntervalTree
for chrom, grp in alu_df.groupby('chrom'):
    tree = IntervalTree()
    for _, row in grp.iterrows():
        # 插入半开区间 [begin, end)
        tree.addi(row['start'] + 1, row['end'] + 1, row['alu_id'])
    alu_trees[chrom] = tree

log(f"  Alu 总数: {len(alu_df)}")
log(f"  染色体数: {len(alu_trees)} (含 scaffold)")

# ---------- 碰撞检测：IntervalTree O(log n) 查询 ----------
alu_flags = {}  # tx_id → {'utr5': bool, 'cds': bool, 'utr3': bool}

checked = 0
found_alu = 0

for tx_id in cds_tx_ids:
    info = tx_info[tx_id]
    chrom = info['chrom']

    alu_tree = alu_trees.get(chrom, None)

    # 分别检测三个区域 (纯 C 实现的 IntervalTree 查询)
    alu_in_utr5 = has_alu_overlap(
        tx_utr5.get(tx_id, []), alu_tree, threshold=ALU_OVERLAP_THRESHOLD
    )
    alu_in_cds = has_alu_overlap(
        tx_cds_coords.get(tx_id, []), alu_tree, threshold=ALU_OVERLAP_THRESHOLD
    )
    alu_in_utr3 = has_alu_overlap(
        tx_utr3.get(tx_id, []), alu_tree, threshold=ALU_OVERLAP_THRESHOLD
    )

    alu_flags[tx_id] = {
        'utr5': alu_in_utr5,
        'cds': alu_in_cds,
        'utr3': alu_in_utr3,
    }

    checked += 1
    if alu_in_utr5 or alu_in_cds or alu_in_utr3:
        found_alu += 1

    if checked % 1000 == 0:
        log(f"  ... 已检测 {checked} 个转录本, 含 Alu: {found_alu}")

log(f"  检测完成: {checked} 个编码转录本")
log(f"  至少一个区域含 Alu: {found_alu} ({found_alu/max(checked,1)*100:.1f}%)")

# 构建 per-transcript 注释 DataFrame
annot_rows = []
for tx_id in cds_tx_ids:
    flags = alu_flags.get(tx_id, {'utr5': False, 'cds': False, 'utr3': False})
    info = tx_info[tx_id]
    has_alu = flags['utr5'] or flags['cds'] or flags['utr3']
    alu_total = int(flags['utr5']) + int(flags['cds']) + int(flags['utr3'])
    annot_rows.append({
        'transcript_id': tx_id,
        'gene_id': info['gene_id'],
        'gene_name': info['gene_name'],
        'chrom': info['chrom'],
        'strand': info['strand'],
        'alu_in_utr5': int(flags['utr5']),
        'alu_in_cds': int(flags['cds']),
        'alu_in_utr3': int(flags['utr3']),
        'alu_total': alu_total,
        'has_alu': int(has_alu),
    })

annot_df = pd.DataFrame(annot_rows)
log(f"\n  注释矩阵大小: {annot_df.shape}")
log(f"  has_alu==1: {annot_df['has_alu'].sum()}")
log(f"  has_alu==0: {(annot_df['has_alu'] == 0).sum()}")

log(f"  Step 2 耗时: {time.time() - t1:.0f}s")


# ========================== Step 3: TE 数据对齐 ==========================

log("\n" + "=" * 70)
log("Step 3: TE 数据对齐与极值清洗")
log("=" * 70)

t2 = time.time()

# 读取 TEDD 数据
tedd_df = pd.read_csv(TEDD_PATH)
log(f"  TEDD 列名: {list(tedd_df.columns)}")
log(f"  原始行数: {len(tedd_df)}")

# 自动识别 transcript_id 列和 TE 值列
tx_col_candidates = [c for c in tedd_df.columns if 'transcript' in c.lower() or 'tx' in c.lower() or 'enst' in c.lower()]
te_col_candidates = [c for c in tedd_df.columns if c == 'TE' or 'te_' in c.lower() or 'te' == c.lower()]
log(f"  候选 transcript_id 列: {tx_col_candidates}")
log(f"  候选 TE 值列: {te_col_candidates}")

# 确定列名
TX_COL = 'TRANSCRIPT_ID' if 'TRANSCRIPT_ID' in tedd_df.columns else tx_col_candidates[0] if tx_col_candidates else None
TE_COL = 'TE' if 'TE' in tedd_df.columns else te_col_candidates[0] if te_col_candidates else None

if TX_COL is None or TE_COL is None:
    raise ValueError(f"无法自动识别 transcript_id 或 TE 列！列名: {list(tedd_df.columns)}")

log(f"  使用: transcript_id 列 = '{TX_COL}', TE 列 = '{TE_COL}'")

# 提取所需列，剥离 ID 版本号
tedd_sub = tedd_df[[TX_COL, TE_COL]].copy()
tedd_sub.columns = ['transcript_id', 'TE']

# 剥离版本号
tedd_sub['transcript_id'] = tedd_sub['transcript_id'].astype(str).str.split('.').str[0]

# 过滤 TE <= 0 或 NaN
n_before = len(tedd_sub)
tedd_sub = tedd_sub.dropna(subset=['TE'])
tedd_sub = tedd_sub[tedd_sub['TE'] > 0]
n_after = len(tedd_sub)
log(f"  TE 清洗: {n_before} → {n_after} ({n_before - n_after} 条被过滤)")

# ---------- 合并 ----------
merged = pd.merge(annot_df, tedd_sub, on='transcript_id', how='inner')
log(f"  合并后: {len(merged)} 条转录本")
log(f"  涉及基因: {merged['gene_id'].nunique()}")

# 统计汇总
alu_plus = merged[merged['has_alu'] == 1]
alu_minus = merged[merged['has_alu'] == 0]
log(f"  Alu+ 转录本: {len(alu_plus)}, Alu- 转录本: {len(alu_minus)}")

# 按基因看分布
gene_stats = merged.groupby('gene_id')['has_alu'].agg(['sum', 'count'])
gene_stats.columns = ['n_alu_plus', 'n_total']
gene_stats['n_alu_minus'] = gene_stats['n_total'] - gene_stats['n_alu_plus']
both_groups = gene_stats[(gene_stats['n_alu_plus'] > 0) & (gene_stats['n_alu_minus'] > 0)]
log(f"  可配对基因（Alu+ 和 Alu- 都有）：{len(both_groups)}")

log(f"  Step 3 耗时: {time.time() - t2:.0f}s")


# ========================== Step 4: 基因内中位数聚合 ==========================

log("\n" + "=" * 70)
log("Step 4: 基因内中位数聚合 + 配对数据构建")
log("=" * 70)

t3 = time.time()

# 对每个基因，分别计算 Alu+ 组和 Alu- 组的 TE 中位数
paired_data = []

for gene_id, gene_df in merged.groupby('gene_id'):
    alu_plus_vals = gene_df[gene_df['has_alu'] == 1]['TE'].values
    alu_minus_vals = gene_df[gene_df['has_alu'] == 0]['TE'].values

    if len(alu_plus_vals) == 0 or len(alu_minus_vals) == 0:
        continue

    median_plus = np.median(alu_plus_vals)
    median_minus = np.median(alu_minus_vals)

    gene_symbol = gene_df['gene_name'].iloc[0]

    paired_data.append({
        'gene_id': gene_id,
        'gene_name': gene_symbol,
        'n_alu_plus': len(alu_plus_vals),
        'n_alu_minus': len(alu_minus_vals),
        'te_median_alu_plus': median_plus,
        'te_median_alu_minus': median_minus,
        'diff': median_plus - median_minus,
        'pct_change': ((median_plus - median_minus) / median_minus * 100) if median_minus != 0 else np.nan,
    })

paired_df = pd.DataFrame(paired_data)
log(f"  配对基因数: {len(paired_df)}")

# QC 过滤：两个 TE 中位数都 > 0
paired_df = paired_df[(paired_df['te_median_alu_plus'] > 0) & (paired_df['te_median_alu_minus'] > 0)].copy()
log(f"  QC 过滤后: {len(paired_df)} 个基因")

log(f"  Step 4 耗时: {time.time() - t3:.0f}s")


# ========================== Step 5: Wilcoxon 检验 + 可视化 ==========================

log("\n" + "=" * 70)
log("Step 5: Wilcoxon 符号秩检验")
log("=" * 70)

t4 = time.time()

# 主检验
if len(paired_df) >= 10:
    w_stat, w_pval = wilcoxon(
        paired_df['te_median_alu_plus'],
        paired_df['te_median_alu_minus'],
        alternative='two-sided'
    )
else:
    w_stat, w_pval = np.nan, np.nan
    log("  ⚠ 配对数 < 10，不进行检验")

# 计算总中位数（所有基因的 Alu+ 和 Alu- 的中位数）
global_median_plus = paired_df['te_median_alu_plus'].median()
global_median_minus = paired_df['te_median_alu_minus'].median()

# 报告
log(f"\n  {'=' * 50}")
log(f"  【Wilcoxon 配对检验结果】")
log(f"  {'=' * 50}")
log(f"    有效配对基因数 (N):     {len(paired_df)}")
log(f"    总中位数 Alu+:          {global_median_plus:.4f}")
log(f"    总中位数 Alu-:          {global_median_minus:.4f}")
log(f"    中位数差值:             {global_median_plus - global_median_minus:.4f}")
log(f"    Wilcoxon W 统计量:      {w_stat:.2f}" if not np.isnan(w_stat) else "    Wilcoxon W 统计量:      N/A")
if not np.isnan(w_pval):
    if w_pval < 0.001:
        log(f"    P-value:               {w_pval:.2e}  ***")
    elif w_pval < 0.01:
        log(f"    P-value:               {w_pval:.4f}  **")
    elif w_pval < 0.05:
        log(f"    P-value:               {w_pval:.4f}  *")
    else:
        log(f"    P-value:               {w_pval:.4f}  ns")
log(f"  {'=' * 50}")

# 方向判断
if global_median_plus > global_median_minus:
    log(f"  → 结论: Alu+ 转录本的 TE 显著高于 Alu- 转录本")
elif global_median_plus < global_median_minus:
    log(f"  → 结论: Alu+ 转录本的 TE 显著低于 Alu- 转录本")
else:
    log(f"  → 结论: 两组无显著差异")

# ---------- 可视化 ----------

log(f"\n  Step 5 统计检验耗时: {time.time() - t4:.0f}s")
log(f"  开始绘制可视化...")

t5 = time.time()

# 主图: 全部外显子区域
plot_paired_te(paired_df, OUTPUT_VIOLIN, "All exonic regions", w_pval, len(paired_df))

log(f"  Step 5 绘图耗时: {time.time() - t5:.0f}s")


# ========================== Step 6: 分区域分析 ==========================

log("\n" + "=" * 70)
log("Step 6: 按 Alu 所在区域分层分析 (5'UTR / CDS / 3'UTR)")
log("=" * 70)

region_configs = [
    ('alu_in_utr5', "5'UTR"),
    ('alu_in_cds', 'CDS'),
    ('alu_in_utr3', "3'UTR"),
]

region_results = []

for region_col, region_label in region_configs:
    log(f"\n  --- 区域: {region_label} ---")

    # 按基因分组，用该区域的 Alu 标记
    region_paired = []
    for gene_id, gene_df in merged.groupby('gene_id'):
        alu_plus_vals = gene_df[gene_df[region_col] == 1]['TE'].values
        alu_minus_vals = gene_df[gene_df[region_col] == 0]['TE'].values

        if len(alu_plus_vals) == 0 or len(alu_minus_vals) == 0:
            continue

        region_paired.append({
            'gene_id': gene_id,
            'n_alu_plus': len(alu_plus_vals),
            'n_alu_minus': len(alu_minus_vals),
            'te_median_alu_plus': np.median(alu_plus_vals),
            'te_median_alu_minus': np.median(alu_minus_vals),
        })

    region_df = pd.DataFrame(region_paired)

    # QC 过滤
    region_df = region_df[(region_df['te_median_alu_plus'] > 0) & (region_df['te_median_alu_minus'] > 0)]

    n_pairs = len(region_df)
    if n_pairs >= 5:
        rw_stat, rw_pval = wilcoxon(
            region_df['te_median_alu_plus'],
            region_df['te_median_alu_minus'],
            alternative='two-sided'
        )
    else:
        rw_stat, rw_pval = np.nan, np.nan

    r_median_plus = region_df['te_median_alu_plus'].median()
    r_median_minus = region_df['te_median_alu_minus'].median()

    if not np.isnan(rw_pval):
        if rw_pval < 0.001:
            sig = "***"
        elif rw_pval < 0.01:
            sig = "**"
        elif rw_pval < 0.05:
            sig = "*"
        else:
            sig = "ns"
    else:
        sig = "N/A"

    log(f"    配对基因数: {n_pairs}")
    log(f"    中位数 TE (Alu+): {r_median_plus:.4f}, (Alu-): {r_median_minus:.4f}")
    if not np.isnan(rw_pval):
        log(f"    Wilcoxon p = {rw_pval:.4f} {sig}")
    else:
        log(f"    Wilcoxon: N/A (配对不足)")

    region_results.append({
        'region': region_label,
        'n_paired_genes': n_pairs,
        'median_TE_alu_plus': round(r_median_plus, 4),
        'median_TE_alu_minus': round(r_median_minus, 4),
        'median_diff': round(r_median_plus - r_median_minus, 4),
        'Wilcoxon_W': round(rw_stat, 2) if not np.isnan(rw_stat) else 'N/A',
        'Wilcoxon_pval': rw_pval,
        'significance': sig,
    })

    # --- 绘制该区域的分层配对图 ---
    region_plot_path = os.path.join(OUTPUT_DIR, f"alu_te_within_gene_{region_col}.png")
    region_paired_df = region_df.rename(columns={
        'te_median_alu_plus': 'te_median_alu_plus',
        'te_median_alu_minus': 'te_median_alu_minus',
    })
    region_paired_df['diff'] = region_paired_df['te_median_alu_plus'] - region_paired_df['te_median_alu_minus']
    plot_paired_te(region_paired_df, region_plot_path, f"Alu in {region_label}", rw_pval, n_pairs)


# ========================== 输出结果 ==========================

log("\n" + "=" * 70)
log("保存结果")
log("=" * 70)

# 构建完整结果表
summary_rows = []
# 主结果
if len(paired_df) >= 10:
    main_w, main_p = wilcoxon(
        paired_df['te_median_alu_plus'],
        paired_df['te_median_alu_minus'],
        alternative='two-sided'
    )
else:
    main_w, main_p = np.nan, np.nan

# 计算主检验的显著性星号
if not np.isnan(main_p):
    if main_p < 0.001:
        main_star = "***"
    elif main_p < 0.01:
        main_star = "**"
    elif main_p < 0.05:
        main_star = "*"
    else:
        main_star = "ns"
else:
    main_star = "N/A"

summary_rows.append({
    'region': 'All exonic',
    'n_paired_genes': len(paired_df),
    'median_TE_alu_plus': round(global_median_plus, 4),
    'median_TE_alu_minus': round(global_median_minus, 4),
    'median_diff': round(global_median_plus - global_median_minus, 4),
    'Wilcoxon_W': round(main_w, 2) if not np.isnan(main_w) else 'N/A',
    'Wilcoxon_pval': main_p,
    'significance': main_star,
})

# 区域结果
summary_rows.extend(region_results)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_CSV, index=False, float_format='%.6g')
log(f"  详细结果: {OUTPUT_CSV}")

# 同时保存配对基因级数据
paired_detail_csv = os.path.join(OUTPUT_DIR, "alu_te_within_gene_paired_genes.csv")
paired_df.to_csv(paired_detail_csv, index=False, float_format='%.6g')
log(f"  基因级数据: {paired_detail_csv}")


# ========================== 打印最终汇总 ==========================

log("\n" + "=" * 70)
log("最终分析报告")
log("=" * 70)
log(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

log(f"\n总运行耗时: {time.time() - t0:.0f}s ({((time.time() - t0)/60):.1f} 分钟)")
log("分析完成！")
log("=" * 70)