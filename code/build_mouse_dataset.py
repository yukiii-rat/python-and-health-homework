# -*- coding: utf-8 -*-
"""
合并 feature_matrix_mouse.csv 与 mmc1_updated.csv
按 gene_name inner join，打印 TE 分布 + 直方图
输出：mouse_full_dataset.csv

修改日期：2026-05-31
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== 1. 读取并合并 ====================
# 修改日期：2026-05-31
fm = pd.read_csv("feature_matrix_mouse.csv")
mmc1 = pd.read_csv("mmc1_updated.csv")

print(f"feature_matrix_mouse: {len(fm)} 行")
print(f"mmc1_updated: {len(mmc1)} 行")

# inner join: gene_name <-> New_Official_Symbol
df = pd.merge(fm, mmc1, left_on='gene_name', right_on='New_Official_Symbol', how='inner')
# 去除重复的列（Gene, New_Official_Symbol 等）
if 'Gene' in df.columns:
    df = df.drop(columns=['Gene', 'New_Official_Symbol'])
print(f"inner join 后: {len(df)} 行, {len(df.columns)} 列")

# ==================== 2. TE 分布统计 ====================
# 修改日期：2026-05-31
te = df['log2 TE']
print("\n===== log2 TE 分布统计 =====")
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
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(te, bins=80, color='mediumseagreen', edgecolor='white', alpha=0.8)
ax.set_xlabel('log2 TE')
ax.set_ylabel('Frequency')
ax.set_title(f'Mouse log2 TE Distribution (already log-transformed)\nskew={te.skew():.2f}')
plt.tight_layout()
plt.savefig('TE_distribution_mouse.png', dpi=150)
print("直方图已保存: TE_distribution_mouse.png")

# ==================== 4. 输出 ====================
# 修改日期：2026-05-31
# 重命名 log2 TE 为 TE 保持与人类数据一致
df = df.rename(columns={'log2 TE': 'TE'})
df.to_csv("mouse_full_dataset.csv", index=False)
print(f"\n输出: mouse_full_dataset.csv ({len(df)} 行, {len(df.columns)} 列)")
print("列:", list(df.columns))