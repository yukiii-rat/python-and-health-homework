# -*- coding: utf-8 -*-
"""
合并 feature_matrix.csv 与 TEDD00137.genes_transcript.csv
按 gene_name inner join，打印 TE 分布 + 直方图，异常值做 log2(TE+1) 变换
输出：human_full_dataset.csv

修改日期：2026-05-31
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 1. 读取并合并 ====================
# 修改日期：2026-05-31
fm = pd.read_csv("feature_matrix.csv")
tedd = pd.read_csv("TEDD00137.genes_transcript.csv")

print(f"feature_matrix: {len(fm)} 行")
print(f"TEDD: {len(tedd)} 行")

# inner join
df = pd.merge(fm, tedd, left_on='gene_name', right_on='GENE_SYMBOL', how='inner')
print(f"inner join 后: {len(df)} 行")

# ==================== 2. TE 分布统计 ====================
# 修改日期：2026-05-31
te = df['TE']
print("\n===== TE 分布统计 =====")
print(f"均值: {te.mean():.4f}")
print(f"标准差: {te.std():.4f}")
print(f"最小值: {te.min():.4f}")
print(f"25%: {te.quantile(0.25):.4f}")
print(f"50%: {te.median():.4f}")
print(f"75%: {te.quantile(0.75):.4f}")
print(f"最大值: {te.max():.4f}")
print(f"偏度: {te.skew():.2f}")
print(f"峰度: {te.kurtosis():.2f}")

q99 = te.quantile(0.99)
print(f"99% 分位数: {q99:.4f}")

# ==================== 3. 直方图 ====================
# 修改日期：2026-05-31
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 变换前
axes[0].hist(te, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('TE')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'TE Distribution (raw)\nskew={te.skew():.2f}, max={te.max():.2f}')

# 判断是否需要 log2 变换
need_log = te.max() > q99 * 5
if need_log:
    print("\n>>> 存在极端异常值，执行 log2(TE+1) 变换")
    te_transformed = np.log2(te + 1)
    df['TE'] = te_transformed
    print(f"变换后 TE 均值: {te_transformed.mean():.4f}")
    print(f"变换后 TE 标准差: {te_transformed.std():.4f}")
    print(f"变换后 TE 偏度: {te_transformed.skew():.2f}")
else:
    te_transformed = te

# 变换后
axes[1].hist(te_transformed, bins=80, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('log2(TE+1)' if need_log else 'TE')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'TE Distribution ({"log2 transformed" if need_log else "raw"})')

plt.tight_layout()
plt.savefig('TE_distribution.png', dpi=150)
print("直方图已保存: TE_distribution.png")

# ==================== 4. 输出完整数据集 ====================
# 修改日期：2026-05-31
df.to_csv("human_full_dataset.csv", index=False)
print(f"\n输出: human_full_dataset.csv ({len(df)} 行, {len(df.columns)} 列)")
print("列:", list(df.columns))