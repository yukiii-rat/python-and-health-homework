# -*- coding: utf-8 -*-
"""
Mann-Whitney U 检验：比较有/无特定区域 SINE/Alu 元件的基因间 TE 差异
输出：stat_results.txt + violin_plots.pdf + violin_plots.png

修改日期：2026-05-31
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局绘图设置（SCI 论文级）
# ============================================================
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.4)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
# 科研配色
PALETTE = {'Absent': '#7f8c8d', 'Present': '#e74c3c'}
ABSENT_COLOR = '#7f8c8d'
PRESENT_COLOR = '#e74c3c'


# ============================================================
# 核心统计函数
# ============================================================
def mannwhitney_compare(df, feature_col, te_col='TE', group_name=''):
    """
    对指定二值特征列执行 Mann-Whitney U 检验。

    参数
    ----
    df : DataFrame
    feature_col : str — 二值特征列名（1=实验组，0=对照组）
    te_col : str — TE 列名
    group_name : str — 显示用组名

    返回
    ----
    result_dict : 包含各组统计量和检验结果
    """
    grp_yes = df[df[feature_col] > 0][te_col].dropna()
    grp_no  = df[df[feature_col] == 0][te_col].dropna()

    # Mann-Whitney U 检验（双侧）
    u_stat, p_val = mannwhitneyu(grp_yes, grp_no, alternative='two-sided')

    # 中位数
    med_yes = grp_yes.median()
    med_no  = grp_no.median()

    result = {
        'feature':      feature_col,
        'group':        group_name,
        'N_yes':        len(grp_yes),
        'N_no':         len(grp_no),
        'median_yes':   med_yes,
        'median_no':    med_no,
        'U_statistic':  u_stat,
        'p_value':      p_val,
    }
    return result


def format_pvalue(p):
    """P-value 转星号标注"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("加载数据...")
print("=" * 60)

# 人类
human = pd.read_csv("output1/human_full_dataset.csv")
# 小鼠全量
mouse = pd.read_csv("output1/mouse_full_dataset.csv")
# 小鼠 B1 特征矩阵 + 合并 TE
mouse_b1 = pd.read_csv("output1/feature_matrix_mouse_B1.csv")
te_map = mouse[['gene_id', 'TE']].drop_duplicates('gene_id')
mouse_b1 = pd.merge(mouse_b1, te_map, on='gene_id', how='inner')
# 小鼠 B2 特征矩阵 + 合并 TE
mouse_b2 = pd.read_csv("output1/feature_matrix_mouse_B2.csv")
mouse_b2 = pd.merge(mouse_b2, te_map, on='gene_id', how='inner')

print(f"  人类: {len(human)} 基因")
print(f"  小鼠全量: {len(mouse)} 基因")
print(f"  小鼠 B1: {len(mouse_b1)} 基因")
print(f"  小鼠 B2: {len(mouse_b2)} 基因")

# ============================================================
# 2. 定义待检验的特征对
# ============================================================
# 格式：(数据集, 特征列, 数据集名称)
tests = [
    # 人类 — 2 个核心特征
    (human, 'has_3utr_alu',      'Human'),
    (human, 'has_antisense_alu', 'Human'),
    # 小鼠 B1 — 2 个核心特征
    (mouse_b1, 'has_3utr_B1',      'Mouse_B1'),
    (mouse_b1, 'has_antisense_B1', 'Mouse_B1'),
]

# ============================================================
# 3. 遍历执行检验
# ============================================================
print("\n" + "=" * 60)
print("Mann-Whitney U 检验结果")
print("=" * 60)

results = []
for df, feat, grp in tests:
    r = mannwhitney_compare(df, feat, group_name=grp)
    results.append(r)

    # 打印到控制台
    print(f"\n{'─' * 50}")
    print(f"  {grp} | {feat}")
    print(f"{'─' * 50}")
    print(f"  有 (N={r['N_yes']}):  中位数 TE = {r['median_yes']:.4f}")
    print(f"  无 (N={r['N_no']}):   中位数 TE = {r['median_no']:.4f}")
    print(f"  Mann-Whitney U = {r['U_statistic']:.1f}, p = {r['p_value']:.4e}")
    p_str = format_pvalue(r['p_value'])
    direction = "更高" if r['median_yes'] > r['median_no'] else "更低"
    print(f"  结论: 有元件的基因 TE 显著{direction}于无元件的基因 (p{format_pvalue(r['p_value'])})")

# ============================================================
# 4. 高颜值小提琴图
# ============================================================
print("\n\n绘制小提琴图...")

# 定义 4 个子图的参数
plot_configs = [
    {
        'data':      human,
        'feature':   'has_3utr_alu',
        'title':     'Human: Alu in 3\'UTR',
        'ax_label':  '3\'UTR Alu Status',
    },
    {
        'data':      human,
        'feature':   'has_antisense_alu',
        'title':     'Human: Antisense Alu',
        'ax_label':  'Antisense Alu Status',
    },
    {
        'data':      mouse_b1,
        'feature':   'has_3utr_B1',
        'title':     'Mouse B1: B1 in 3\'UTR',
        'ax_label':  '3\'UTR B1 Status',
    },
    {
        'data':      mouse_b1,
        'feature':   'has_antisense_B1',
        'title':     'Mouse B1: Antisense B1',
        'ax_label':  'Antisense B1 Status',
    },
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

for idx, cfg in enumerate(plot_configs):
    ax = axes[idx // 2][idx % 2]
    df = cfg['data']
    feat = cfg['feature']

    # 构建绘图数据：添加分组标签
    plot_df = df[[feat, 'TE']].copy()
    plot_df['Group'] = np.where(plot_df[feat] > 0, 'Present', 'Absent')

    # 小提琴图 + 内嵌箱线图
    order = ['Absent', 'Present']
    parts = sns.violinplot(
        data=plot_df, x='Group', y='TE', order=order,
        palette={'Absent': ABSENT_COLOR, 'Present': PRESENT_COLOR},
        inner='box', linewidth=1.2, cut=0,
        width=0.7, ax=ax,
    )

    # 散点（可选小点）
    sns.stripplot(
        data=plot_df, x='Group', y='TE', order=order,
        color='black', size=1.5, alpha=0.3, jitter=True, ax=ax,
    )

    # 自动显著性标注
    r = [res for res in results if res['feature'] == feat][0]
    p_val = r['p_value']
    star_str = format_pvalue(p_val)

    # 获取 y 轴范围确定标注位置
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_annot = y_max + y_range * 0.05

    # 画显著性连线和星号
    x1, x2 = 0, 1  # Absent=0, Present=1
    ax.plot([x1, x1, x2, x2], [y_annot, y_annot + y_range * 0.03,
                                y_annot + y_range * 0.03, y_annot],
            lw=1.2, color='black')
    ax.text((x1 + x2) / 2, y_annot + y_range * 0.06, star_str,
            ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 标注样本量
    n_absent = r['N_no']
    n_present = r['N_yes']
    ax.set_xticklabels([f"Absent\n(n={n_absent})", f"Present\n(n={n_present})"])

    # 中位数标注
    ax.text(0, r['median_no'], f"{r['median_no']:.2f}",
            ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
    ax.text(1, r['median_yes'], f"{r['median_yes']:.2f}",
            ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')

    # 标题
    ax.set_title(cfg['title'], fontsize=13, fontweight='bold')
    ax.set_xlabel(cfg['ax_label'], fontsize=11)
    ax.set_ylabel('log2(TE+1)' if cfg['ax_label'].startswith('Human') else 'log2 TE', fontsize=11)

    # 恢复 y 轴范围（留出标注空间）
    ax.set_ylim(y_min, y_annot + y_range * 0.12)

plt.suptitle('Mann-Whitney U Test: TE Comparison by Alu/SINE Status',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('output1/violin_plots.pdf')
plt.savefig('output1/violin_plots.png')
plt.close()
print("  已保存: output1/violin_plots.pdf / .png")

# ============================================================
# 5. 保存统计表格
# ============================================================
result_df = pd.DataFrame(results)
result_df['significance'] = result_df['p_value'].apply(format_pvalue)
result_df['direction'] = np.where(
    result_df['median_yes'] > result_df['median_no'],
    'TE_higher_with_element',
    'TE_lower_with_element'
)
result_df = result_df[[
    'group', 'feature', 'N_yes', 'N_no',
    'median_yes', 'median_no',
    'U_statistic', 'p_value', 'significance', 'direction',
]]

result_df.to_csv("output1/stat_results.csv", index=False, float_format='%.6e')
print("\n统计表格已保存: output1/stat_results.csv")
print(result_df.to_string(index=False))
print("\n完成！")