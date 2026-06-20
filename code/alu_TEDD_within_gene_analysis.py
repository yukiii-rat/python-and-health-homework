# -*- coding: utf-8 -*-
"""
Alu 外显子 overlap 与 TEDD TE 表达值的基因内配对分析

流程：
1. 读取 alu_exonic_per_transcript.csv（每个 ENST 的 Alu 外显子 overlap 情况）
2. 读取 TEDD00137.distribution_transcript.csv（每个 ENST 的 TE 表达值）
3. 合并后，对每个基因：
   - 分组：含 Alu 的转录本 vs 不含 Alu 的转录本
   - 统计检验：Mann-Whitney U（基因内两组间比较）
   - 跨基因汇总：Wilcoxon signed-rank（配对比较基因中位数）
4. 按条件（CONDITION）和细胞类型分层分析
5. 可视化输出

输入：
  - output1/alu_exonic_per_transcript.csv
  - data/TEDD00137.distribution_transcript.csv

输出：
  - output1/within_gene_per_gene_results.csv   (每个基因的检验结果)
  - output1/within_gene_global_results.csv      (跨基因汇总)
  - output1/within_gene_violin.png              (小提琴图)
  - output1/within_gene_paired_dot.png          (配对点图)

修改日期：2026-06-07
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 参数设置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output1')

ALU_TRANSCRIPT_CSV = os.path.join(OUTPUT_DIR, "alu_exonic_per_transcript.csv")
TEDD_CSV = os.path.join(DATA_DIR, "TEDD00137.distribution_transcript.csv")

OUTPUT_GENE_RESULTS = os.path.join(OUTPUT_DIR, "within_gene_per_gene_results.csv")
OUTPUT_GLOBAL_RESULTS = os.path.join(OUTPUT_DIR, "within_gene_global_results.csv")
OUTPUT_VIOLIN = os.path.join(OUTPUT_DIR, "within_gene_violin.png")
OUTPUT_DOT = os.path.join(OUTPUT_DIR, "within_gene_paired_dot.png")

MIN_TRANSCRIPTS_PER_GROUP = 1  # 每组最少需要的转录本数

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 1. 读取数据 ====================
print("=" * 60)
print("Step 1: 读取数据")
print("=" * 60)

# 读取 Alu 外显子 overlap 结果
alu_df = pd.read_csv(ALU_TRANSCRIPT_CSV)
print(f"Alu overlap 数据: {len(alu_df)} 转录本, {alu_df['has_alu_exonic'].sum()} 含 Alu")

# 读取 TEDD 数据
tedd_df = pd.read_csv(TEDD_CSV)
print(f"TEDD 数据: {len(tedd_df)} 行")

# 查看 TEDD 的列
print(f"TEDD 列: {list(tedd_df.columns)}")
print(f"TEDD 条件 (CONDITION): {tedd_df['CONDITION'].unique()}")
print(f"TEDD 细胞类型: {tedd_df['CELL_LINE'].unique()}")
print(f"TEDD 组织: {tedd_df['TISSUECELLTYPE'].unique()}")

# ==================== 2. 合并数据 ====================
print("=" * 60)
print("Step 2: 合并 Alu overlap 与 TEDD 数据")
print("=" * 60)

# 合并
merged_df = pd.merge(
    tedd_df, alu_df, on='transcript_id', how='inner', suffixes=('_tedd', '_alu')
)

print(f"合并后记录数: {len(merged_df)}")
print(f"合并后基因数: {merged_df['gene_id'].nunique()}")

# 检查是否有 gene_id 不匹配的情况
tedd_only = set(tedd_df['GENE_ID']) - set(alu_df['gene_id'])
alu_only = set(alu_df['gene_id']) - set(tedd_df['GENE_ID'])
print(f"仅在 TEDD 中的基因: {len(tedd_only)}")
print(f"仅在 Alu 数据中的基因: {len(alu_only)}")

# 统一 gene_id 列名（TEDD 用 GENE_ID，alu 用 gene_id）
# 合并时两者可能不同名，确认一下
if 'gene_id' in merged_df.columns and 'GENE_ID' in merged_df.columns:
    # 检查是否一致
    mismatch = (merged_df['gene_id'] != merged_df['GENE_ID']).sum()
    if mismatch > 0:
        print(f"警告: {mismatch} 行的 gene_id 不匹配")
    # 移除冗余列
    merged_df = merged_df.drop(columns=['GENE_ID', 'GENE_SYMBOL'])
    merged_df = merged_df.rename(columns={'gene_id': 'GENE_ID', 'gene_name': 'GENE_SYMBOL'})
elif 'GENE_ID' in merged_df.columns:
    # TEDD 的 GENE_ID 已被保留
    pass

# 确保 TE 列是数值
merged_df['TE'] = pd.to_numeric(merged_df['TE'], errors='coerce')

# ==================== 3. 数据探索 ====================
print("=" * 60)
print("Step 3: 数据探索")
print("=" * 60)

# 含 Alu 和不含 Alu 的转录本分布
alu_counts = merged_df.groupby(['GENE_ID', 'has_alu_exonic']).size().unstack(fill_value=0)
print(f"\n基因内转录本分布:")
print(f"  只有 Alu+ 转录本的基因: {(alu_counts[1] > 0).sum()}")
if 0 in alu_counts.columns:
    print(f"  只有 Alu- 转录本的基因: {(alu_counts[0] > 0).sum()}")
    print(f"  同时有 Alu+ 和 Alu- 的基因: {((alu_counts.get(0, 0) > 0) & (alu_counts.get(1, 0) > 0)).sum()}")
else:
    print(f"  只有 Alu- 转录本的基因: 0")
    print(f"  同时有 Alu+ 和 Alu- 的基因: 0")

# 按 CONDITION 分组分析
conditions = merged_df['CONDITION'].unique()
print(f"\n按条件分析: {conditions}")

# ==================== 4. 基因内配对检验 ====================
print("=" * 60)
print("Step 4: 基因内 Mann-Whitney U 检验")
print("=" * 60)

all_gene_results = []

for condition in conditions:
    cond_df = merged_df[merged_df['CONDITION'] == condition]
    n_tx = len(cond_df)
    n_genes = cond_df['GENE_ID'].nunique()
    print(f"\n条件: {condition} ({n_tx} 转录本, {n_genes} 基因)")

    for gene_id, gene_df in cond_df.groupby('GENE_ID'):
        gene_symbol = gene_df['GENE_SYMBOL'].iloc[0]

        alu_pos = gene_df[gene_df['has_alu_exonic'] == 1]['TE'].dropna()
        alu_neg = gene_df[gene_df['has_alu_exonic'] == 0]['TE'].dropna()

        n_pos = len(alu_pos)
        n_neg = len(alu_neg)

        # 两组都要有数据
        if n_pos < MIN_TRANSCRIPTS_PER_GROUP or n_neg < MIN_TRANSCRIPTS_PER_GROUP:
            continue

        # Mann-Whitney U test
        try:
            u_stat, p_val = mannwhitneyu(alu_pos, alu_neg, alternative='two-sided')
        except ValueError:
            continue

        # 效应量：Cohen's d 近似 (Z / sqrt(N))
        n_total = n_pos + n_neg
        z_score = (u_stat - n_pos * n_neg / 2) / np.sqrt(n_pos * n_neg * (n_total + 1) / 12)
        effect_size = z_score / np.sqrt(n_total) if n_total > 0 else 0

        # 中位数
        median_pos = alu_pos.median()
        median_neg = alu_neg.median()
        median_diff = median_pos - median_neg
        # 百分比变化
        pct_change = ((median_pos - median_neg) / median_neg * 100) if median_neg != 0 else 0

        all_gene_results.append({
            'CONDITION': condition,
            'GENE_ID': gene_id,
            'GENE_SYMBOL': gene_symbol,
            'n_alu_plus': n_pos,
            'n_alu_minus': n_neg,
            'median_TE_alu_plus': round(median_pos, 4),
            'median_TE_alu_minus': round(median_neg, 4),
            'median_diff': round(median_diff, 4),
            'pct_change': round(pct_change, 2),
            'MannWhitney_U': u_stat,
            'p_value': p_val,
            'effect_size': round(effect_size, 4),
            'z_score': round(z_score, 4),
        })

gene_results_df = pd.DataFrame(all_gene_results)

if gene_results_df.empty:
    print("没有找到可检验的基因！")
    # 输出空结果
    empty_df = pd.DataFrame(columns=[
        'CONDITION', 'GENE_ID', 'GENE_SYMBOL',
        'n_alu_plus', 'n_alu_minus',
        'median_TE_alu_plus', 'median_TE_alu_minus',
        'median_diff', 'pct_change',
        'MannWhitney_U', 'p_value', 'effect_size'
    ])
    empty_df.to_csv(OUTPUT_GENE_RESULTS, index=False)
    print(f"已输出空文件: {OUTPUT_GENE_RESULTS}")
    exit()

# 多重假设检验校正（Benjamini-Hochberg）
from scipy.stats import rankdata

def bh_correction(p_values):
    n = len(p_values)
    ranked = rankdata(p_values)
    corrected = np.minimum(1, p_values * n / ranked)
    # 确保单调性
    sorted_idx = np.argsort(corrected)
    for i in range(n - 2, -1, -1):
        corrected[sorted_idx[i]] = min(corrected[sorted_idx[i]], corrected[sorted_idx[i + 1]])
    return corrected

gene_results_df['padj_BH'] = bh_correction(gene_results_df['p_value'].values)
gene_results_df['significant'] = gene_results_df['padj_BH'] < 0.05

# 标记方向
gene_results_df['direction'] = np.where(
    gene_results_df['median_diff'] > 0, 'Alu+ > Alu-',
    np.where(gene_results_df['median_diff'] < 0, 'Alu+ < Alu-', 'equal')
)

# 保存
gene_results_df.to_csv(OUTPUT_GENE_RESULTS, index=False)
print(f"\n基因内检验结果保存: {OUTPUT_GENE_RESULTS}")
print(f"检验基因数: {len(gene_results_df)}")
print(f"显著基因数 (padj<0.05): {gene_results_df['significant'].sum()}")
print(f"显著基因方向分布:")
print(gene_results_df[gene_results_df['significant']]['direction'].value_counts().to_string())

# ==================== 5. 跨基因汇总（配对检验） ====================
print("=" * 60)
print("Step 5: 跨基因配对检验")
print("=" * 60)

global_results = []

for condition, cond_genes in gene_results_df.groupby('CONDITION'):
    # 对每个基因，计算 Alu+ 和 Alu- 转录本的中位数 TE 差值
    # 所有基因的 paired comparison
    medians = merged_df[merged_df['CONDITION'] == condition].groupby(
        ['GENE_ID', 'has_alu_exonic']
    )['TE'].median().unstack()

    if 0 not in medians.columns or 1 not in medians.columns:
        print(f"  条件 {condition}: 缺少一组数据，跳过")
        continue

    paired_data = medians.dropna()
    if len(paired_data) < 3:
        print(f"  条件 {condition}: 配对数据不足 ({len(paired_data)} 个基因)")
        continue

    diff = paired_data[1] - paired_data[0]
    median_diff = diff.median()
    mean_diff = diff.mean()

    # Wilcoxon signed-rank test
    try:
        w_stat, w_pval = wilcoxon(paired_data[1], paired_data[0], alternative='two-sided')
    except (ValueError, RuntimeError):
        w_stat, w_pval = np.nan, np.nan

    # 方向
    if median_diff > 0:
        global_direction = "Alu+ median > Alu- median"
    elif median_diff < 0:
        global_direction = "Alu+ median < Alu- median"
    else:
        global_direction = "equal"

    # 各区域的单独分析
    print(f"\n条件: {condition}")
    print(f"  配对基因数: {len(paired_data)}")
    print(f"  中位 TE 差值 (Alu+ minus Alu-): {median_diff:.4f}")
    print(f"  Wilcoxon p-value: {w_pval:.6f}")

    global_results.append({
        'CONDITION': condition,
        'n_paired_genes': len(paired_data),
        'median_TE_alu_plus': round(paired_data[1].median(), 4),
        'median_TE_alu_minus': round(paired_data[0].median(), 4),
        'median_diff': round(median_diff, 4),
        'mean_diff': round(mean_diff, 4),
        'pct_change': round((median_diff / paired_data[0].median() * 100) if paired_data[0].median() != 0 else 0, 2),
        'Wilcoxon_stat': w_stat,
        'Wilcoxon_pval': w_pval,
        'direction': global_direction,
    })

    # ----- 按区域分析 -----
    for region_col, region_label in [('alu_5utr', '5\'UTR'), ('alu_cds', 'CDS'), ('alu_3utr', '3\'UTR')]:
        try:
            # 对有该区域 Alu 的转录本和没有的分别取基因中位数
            region_medians = merged_df[merged_df['CONDITION'] == condition].copy()
            region_medians['has_region_alu'] = region_medians[region_col]

            reg_med = region_medians.groupby(
                ['GENE_ID', 'has_region_alu']
            )['TE'].median().unstack()

            if 0 not in reg_med.columns or 1 not in reg_med.columns:
                continue

            reg_paired = reg_med.dropna()
            if len(reg_paired) < 3:
                continue

            reg_diff = reg_paired[1] - reg_paired[0]
            try:
                rw_stat, rw_pval = wilcoxon(reg_paired[1], reg_paired[0], alternative='two-sided')
            except (ValueError, RuntimeError):
                rw_stat, rw_pval = np.nan, np.nan

            global_results.append({
                'CONDITION': f"{condition}_{region_label}",
                'n_paired_genes': len(reg_paired),
                'median_TE_alu_plus': round(reg_paired[1].median(), 4),
                'median_TE_alu_minus': round(reg_paired[0].median(), 4),
                'median_diff': round(reg_diff.median(), 4),
                'mean_diff': round(reg_diff.mean(), 4),
                'pct_change': round((reg_diff.median() / reg_paired[0].median() * 100) if reg_paired[0].median() != 0 else 0, 2),
                'Wilcoxon_stat': rw_stat,
                'Wilcoxon_pval': rw_pval,
                'direction': "Alu+ median > Alu- median" if reg_diff.median() > 0 else "Alu+ median < Alu- median",
            })

            print(f"  [{region_label}] 配对基因数: {len(reg_paired)}, Wilcoxon p={rw_pval:.6f}")
        except Exception as e:
            print(f"  [{region_label}] 分析失败: {e}")
            continue

global_results_df = pd.DataFrame(global_results)
global_results_df.to_csv(OUTPUT_GLOBAL_RESULTS, index=False)
print(f"\n跨基因汇总结果: {OUTPUT_GLOBAL_RESULTS}")

# ==================== 6. 可视化 ====================
print("=" * 60)
print("Step 6: 可视化")
print("=" * 60)

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# --- 6a. 小提琴图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, condition in enumerate(conditions):
    if idx >= len(axes):
        break
    ax = axes[idx]

    plot_df = merged_df[merged_df['CONDITION'] == condition].copy()
    plot_df['Alu in exons'] = plot_df['has_alu_exonic'].map({0: 'No Alu', 1: 'Alu+'})

    sns.violinplot(
        data=plot_df, x='Alu in exons', y='TE',
        order=['No Alu', 'Alu+'],
        palette={'No Alu': '#6baed6', 'Alu+': '#fd8d3c'},
        ax=ax, inner='quartile'
    )

    # 标注显著性
    cond_global = global_results_df[global_results_df['CONDITION'] == condition]
    if not cond_global.empty and pd.notna(cond_global['Wilcoxon_pval'].iloc[0]):
        pval = cond_global['Wilcoxon_pval'].iloc[0]
        sig_text = f"Wilcoxon p = {pval:.2e}" if pval < 0.001 else f"Wilcoxon p = {pval:.4f}"
        ax.text(0.5, 0.95, sig_text, transform=ax.transAxes,
                ha='center', fontsize=10, fontweight='bold')

    ax.set_title(f'{condition}', fontsize=13, fontweight='bold')
    ax.set_ylabel('TE expression value')
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig(OUTPUT_VIOLIN, dpi=150, bbox_inches='tight')
plt.close()
print(f"小提琴图: {OUTPUT_VIOLIN}")

# --- 6b. 配对点图 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, condition in enumerate(conditions):
    if idx >= len(axes):
        break
    ax = axes[idx]

    medians = merged_df[merged_df['CONDITION'] == condition].groupby(
        ['GENE_ID', 'GENE_SYMBOL', 'has_alu_exonic']
    )['TE'].median().unstack()

    if 0 not in medians.columns or 1 not in medians.columns:
        continue

    paired = medians.dropna().reset_index()
    if len(paired) < 3:
        continue

    # 取 top 30 基因（按差值绝对值排序）
    paired['diff'] = abs(paired[1] - paired[0])
    top_genes = paired.nlargest(min(30, len(paired)), 'diff')

    # 绘制配对连线
    for _, row in top_genes.iterrows():
        ax.plot([0, 1], [row[0], row[1]], 'o-', color='grey', alpha=0.4, linewidth=0.8)

    # 散点
    ax.scatter([0] * len(top_genes), top_genes[0], color='#6baed6', s=40, zorder=5, label='No Alu')
    ax.scatter([1] * len(top_genes), top_genes[1], color='#fd8d3c', s=40, zorder=5, label='Alu+')

    # 中位数连线
    median_neg = top_genes[0].median()
    median_pos = top_genes[1].median()
    ax.plot([0, 1], [median_neg, median_pos], 'r--', linewidth=2, label=f'Median: {median_neg:.3f} → {median_pos:.3f}')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Alu', 'Alu+'])
    ax.set_ylabel('Median TE per gene')
    ax.set_title(f'{condition}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"配对点图: {OUTPUT_DOT}")

# ==================== 7. 保存合并数据 ====================
print("=" * 60)
print("Step 7: 保存合并数据供后续分析")
print("=" * 60)

merged_output = os.path.join(OUTPUT_DIR, "tedd_alu_merged.csv")
merged_df.to_csv(merged_output, index=False)
print(f"合并数据: {merged_output}")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)