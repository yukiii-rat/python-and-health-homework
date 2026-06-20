# -*- coding: utf-8 -*-
"""
TE 预测失败诊断脚本
============================================
Step 1: 验证数据分布
Step 2: 改回归为分类 (AUC)
Step 3: 诊断结论
============================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress, mannwhitneyu
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 加载数据
# ============================================================
print("=" * 70)
print("TE 预测失败诊断报告")
print("=" * 70)

# 人类完整数据集
human = pd.read_csv("output1/human_full_dataset.csv")
# 小鼠完整数据集（含 ribosome/mRNA 原始计数）
mouse = pd.read_csv("output1/mouse_full_dataset.csv")

# ============================================================
# Step 1: y 分布验证
# ============================================================
print("\n\n" + "=" * 70)
print("STEP 1: 验证数据分布")
print("=" * 70)

y_h = human['TE'].values
y_m = mouse['TE'].values

print("\n--- 人类 TE (log2(TE+1)) ---")
print(f"  N = {len(y_h)}")
print(f"  mean ± std = {y_h.mean():.4f} ± {y_h.std():.4f}")
print(f"  min = {y_h.min():.4f}, max = {y_h.max():.4f}")
print(f"  median = {np.median(y_h):.4f}")
print(f"  IQR = {np.percentile(y_h, 75) - np.percentile(y_h, 25):.4f}")
print(f"  skewness = {pd.Series(y_h).skew():.2f}")
print(f"  kurtosis = {pd.Series(y_h).kurtosis():.2f}")

print("\n--- 小鼠 TE (log2 TE) ---")
print(f"  N = {len(y_m)}")
print(f"  mean ± std = {y_m.mean():.4f} ± {y_m.std():.4f}")
print(f"  min = {y_m.min():.4f}, max = {y_m.max():.4f}")
print(f"  median = {np.median(y_m):.4f}")
print(f"  IQR = {np.percentile(y_m, 75) - np.percentile(y_m, 25):.4f}")
print(f"  skewness = {pd.Series(y_m).skew():.2f}")
print(f"  kurtosis = {pd.Series(y_m).kurtosis():.2f}")

# 直方图：人类 vs 小鼠 TE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y_h, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(y_h.mean(), color='red', ls='--', lw=2, label=f'mean={y_h.mean():.2f}')
axes[0].axvline(np.median(y_h), color='green', ls=':', lw=2, label=f'median={np.median(y_h):.2f}')
axes[0].set_xlabel('TE (log2(TE+1))'); axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Human TE Distribution (N={len(y_h)})'); axes[0].legend()

axes[1].hist(y_m, bins=60, color='mediumseagreen', edgecolor='white', alpha=0.8)
axes[1].axvline(y_m.mean(), color='red', ls='--', lw=2, label=f'mean={y_m.mean():.2f}')
axes[1].axvline(np.median(y_m), color='green', ls=':', lw=2, label=f'median={np.median(y_m):.2f}')
axes[1].set_xlabel('TE (log2 TE)'); axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Mouse TE Distribution (N={len(y_m)})'); axes[1].legend()
plt.tight_layout(); plt.savefig('output1/diagnostic_TE_dist.png', dpi=150)
print("\n  直方图已保存: output1/diagnostic_TE_dist.png")

# ============================================================
# Step 1.2: TE=0 的比例
# ============================================================
print("\n--- 原始 TE = 0 的比例 ---")
# 人类 TE 已 log2 变换，接近 0 的为 TE≈1 (log2(1+1)=1)
# 研究原始值
mouse_orig = pd.read_csv("output1/mouse_full_dataset.csv")
# 在小鼠数据中查看原始 TE (TE ratio 列)
te_ratio = mouse_orig['TE ratio'].dropna()
te_zero_frac = (te_ratio == 0).mean()
print(f"  小鼠 TE ratio = 0 的比例: {te_zero_frac:.4f} ({int((te_ratio==0).sum())}/{len(te_ratio)})")

# 查看小鼠中极低 TE 的比例 (log2 TE < -3)
low_te_frac = (y_m < -3).mean()
print(f"  小鼠 log2 TE < -3 (极低TE): {low_te_frac:.4f} ({(y_m < -3).sum()}/{len(y_m)})")

# ============================================================
# Step 1.3 & 1.4: 表达量过滤
# ============================================================
print("\n--- 表达量过滤分析 (小鼠数据) ---")
# 小鼠原始数据有 ribosome 和 mRNA 列
# 这些是 read counts 而非 TPM，改用较低阈值
RPM_THRESHOLD = 10  # 改用 10 read count 作为低表达阈值

low_mrna = (mouse['mRNA'] < RPM_THRESHOLD).sum()
low_ribo = (mouse['ribosome'] < RPM_THRESHOLD).sum()
low_both = ((mouse['mRNA'] < RPM_THRESHOLD) | (mouse['ribosome'] < RPM_THRESHOLD)).sum()
print(f"  数据可能为 read counts，非 TPM")
print(f"  默认阈值: read count >= {RPM_THRESHOLD}")
print(f"  mRNA < {RPM_THRESHOLD}: {low_mrna}/{len(mouse)} ({100*low_mrna/len(mouse):.1f}%)")
print(f"  Ribo < {RPM_THRESHOLD}: {low_ribo}/{len(mouse)} ({100*low_ribo/len(mouse):.1f}%)")
print(f"  任一低于阈值: {low_both}/{len(mouse)} ({100*low_both/len(mouse):.1f}%)")

# 过滤后做 gene_length vs TE
mask = (mouse['mRNA'] >= RPM_THRESHOLD) & (mouse['ribosome'] >= RPM_THRESHOLD)
mouse_filt = mouse[mask].copy()
print(f"\n  过滤后基因数: {len(mouse_filt)} ({100*len(mouse_filt)/len(mouse):.1f}%)")

lr = linregress(mouse_filt['gene_length'], mouse_filt['TE'])
print(f"  过滤后 gene_length vs TE: r2 = {lr.rvalue**2:.4f}, p = {lr.pvalue:.2e}")

lr_all = linregress(mouse['gene_length'], mouse['TE'])
print(f"  过滤前 gene_length vs TE: r2 = {lr_all.rvalue**2:.4f}, p = {lr_all.pvalue:.2e}")

# 多种阈值测试
print(f"\n{'阈值':>10} {'基因数':>8} {'r2':>8} {'p':>10}")
for thresh in [1, 5, 10, 20, 50]:
    m = (mouse['mRNA'] >= thresh) & (mouse['ribosome'] >= thresh)
    n = m.sum()
    if n > 50:
        lr_t = linregress(mouse.loc[m, 'gene_length'], mouse.loc[m, 'TE'])
        print(f"  {thresh:>8}  {n:>8}  {lr_t.rvalue**2:.6f}  {lr_t.pvalue:.2e}")

# ============================================================
# Step 1.5: B1 基因的子集效应量
# ============================================================
print("\n--- B1 3'UTR 效应量分析 ---")
b1 = pd.read_csv("output1/feature_matrix_mouse_B1.csv")
b1_m = pd.merge(b1, mouse[['gene_id', 'TE', 'mRNA', 'ribosome']], on='gene_id', how='inner')

for feature in ['has_3utr_B1', 'has_antisense_B1']:
    g1 = b1_m[b1_m[feature] > 0]['TE'].dropna()
    g0 = b1_m[b1_m[feature] == 0]['TE'].dropna()
    if len(g1) > 5 and len(g0) > 5:
        u, p = mannwhitneyu(g1, g0, alternative='two-sided')
        d = (g1.mean() - g0.mean()) / np.sqrt((g1.var()+g0.var())/2)  # Cohen's d
        print(f"  {feature}: N1={len(g1)}, N0={len(g0)}, "
              f"med1={g1.median():.3f}, med0={g0.median():.3f}, "
              f"MW p={p:.4e}, Cohen d={d:.3f}")

# ============================================================
# Step 2: 回归 → 分类
# ============================================================
print("\n\n" + "=" * 70)
print("STEP 2: 回归 → 分类 (Top 25% vs Bottom 25%)")
print("=" * 70)

# 特征列（去掉非特征列）
feature_cols = [c for c in human.columns if c not in (
    'TE', 'gene_id', 'gene_name', 'GENE_ID', 'GENE_SYMBOL',
    'DATASET_ID', 'CHROMOSOME', 'TISSUECELLTYPE', 'CELL_LINE',
    'CONDITION', 'TR', 'EVI'
)]

print(f"\n总样本: {len(human)}, 特征数: {len(feature_cols)}")

def classification_auc(df, name):
    """将 TE 二值化 (top 25% vs bottom 25%) 做分类 AUC"""
    te = df['TE'].values
    upper = np.percentile(te, 75)
    lower = np.percentile(te, 25)

    # 仅保留 top 25% 和 bottom 25%
    high_mask = te >= upper
    low_mask = te <= lower
    keep = high_mask | low_mask

    df_cls = df[keep].copy()
    y_cls = (df_cls['TE'].values >= upper).astype(int)  # 1=高TE, 0=低TE
    X_cls = df_cls[feature_cols].copy()
    groups = df_cls['gene_id'].values

    n_high = y_cls.sum()
    n_low = len(y_cls) - n_high
    print(f"\n  {name}: 高TE={n_high}, 低TE={n_low}, 总={len(y_cls)}")

    # 5-fold CV AUC
    gkf = GroupKFold(n_splits=5)
    aucs = []
    imp = []

    for fold, (train_i, test_i) in enumerate(gkf.split(X_cls, y_cls, groups)):
        X_tr, X_te = X_cls.iloc[train_i], X_cls.iloc[test_i]
        y_tr, y_te = y_cls[train_i], y_cls[test_i]

        clf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1, class_weight='balanced')
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y_te, y_prob)
            aucs.append(auc)
            imp.append(clf.feature_importances_)
        except:
            pass

    if aucs:
        print(f"  5-fold CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        # 特征重要性
        mean_imp = np.mean(imp, axis=0)
        top_idx = np.argsort(mean_imp)[-5:][::-1]
        print(f"  Top 5 特征:")
        for idx in top_idx:
            print(f"    {feature_cols[idx]}: {mean_imp[idx]:.4f}")
    else:
        print(f"  无法计算 AUC (可能样本太少)")

    return aucs

# 人类分类
human_aucs = classification_auc(human, "Human")

# ============================================================
# Step 2.2: 小鼠分类 (B1)
# ============================================================
print("\n" + "-" * 70)
print("小鼠 B1 分类诊断")
print("-" * 70)

b1_feat = [c for c in b1.columns if c not in ('gene_id', 'gene_name')]
mouse_b1_tmp = pd.merge(b1, mouse[['gene_id', 'TE', 'mRNA', 'ribosome']], on='gene_id', how='inner')

te_b1 = mouse_b1_tmp['TE'].values
upper_b1 = np.percentile(te_b1, 75)
lower_b1 = np.percentile(te_b1, 25)
keep_b1 = (te_b1 >= upper_b1) | (te_b1 <= lower_b1)
df_b1_cls = mouse_b1_tmp[keep_b1].copy()
y_b1_cls = (df_b1_cls['TE'].values >= upper_b1).astype(int)
X_b1_cls = df_b1_cls[b1_feat].values
groups_b1 = df_b1_cls['gene_id'].values
print(f"  B1: 高TE={y_b1_cls.sum()}, 低TE={len(y_b1_cls)-y_b1_cls.sum()}, 总={len(y_b1_cls)}")

gkf = GroupKFold(n_splits=5)
aucs_b1 = []
for train_i, test_i in gkf.split(X_b1_cls, y_b1_cls, groups_b1):
    clf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5,
                                  random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_b1_cls[train_i], y_b1_cls[train_i])
    try:
        aucs_b1.append(roc_auc_score(y_b1_cls[test_i], clf.predict_proba(X_b1_cls[test_i])[:, 1]))
    except:
        pass
if aucs_b1:
    print(f"  B1 5-fold CV AUC: {np.mean(aucs_b1):.4f} ± {np.std(aucs_b1):.4f}")

# ============================================================
# Step 2.3: 仅用表达量预测 TE (正对照)
# ============================================================
print("\n" + "-" * 70)
print("正对照: 用 mRNA / Ribosome 表达量预测 TE 分类")
print("-" * 70)

mouse_cls = mouse.copy()
te_m = mouse_cls['TE'].values
upper_m = np.percentile(te_m, 75)
lower_m = np.percentile(te_m, 25)
keep_m = (te_m >= upper_m) | (te_m <= lower_m)
df_m_cls = mouse_cls[keep_m].copy()
y_m_cls = (df_m_cls['TE'].values >= upper_m).astype(int)

for feat in ['mRNA', 'ribosome']:
    X_m = df_m_cls[[feat]].values
    g_m = df_m_cls['gene_id'].values
    aucs_f = []
    gkf = GroupKFold(n_splits=5)
    for ti, vi in gkf.split(X_m, y_m_cls, g_m):
        clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight='balanced')
        clf.fit(X_m[ti], y_m_cls[ti])
        try:
            aucs_f.append(roc_auc_score(y_m_cls[vi], clf.predict_proba(X_m[vi])[:, 1]))
        except:
            pass
    print(f"  {feat}: AUC = {np.mean(aucs_f):.4f} ± {np.std(aucs_f):.4f}" if aucs_f else f"  {feat}: NA")

# ============================================================
# Step 3: 诊断结论
# ============================================================
print("\n\n" + "=" * 70)
print("STEP 3: 诊断结论")
print("=" * 70)

# 判断标准
y_range_h = y_h.max() - y_h.min()
y_iqr_h = np.percentile(y_h, 75) - np.percentile(y_h, 25)
signal_to_noise = y_iqr_h / y_h.std()

print(f"\n  信号评估:")
print(f"    TE IQR/标准差 = {signal_to_noise:.3f} (越接近0说明变异越小)")
print(f"    TE 范围 = {y_h.min():.2f} ~ {y_h.max():.2f} (range={y_range_h:.2f})")
print(f"    TE Q1/Q3 = {np.percentile(y_h, 25):.2f} / {np.percentile(y_h, 75):.2f}")

# 判断 noise 程度
te_ratio_orig = mouse_orig['TE ratio'].dropna()
te_ratio_cv = te_ratio_orig.std() / te_ratio_orig.mean()
print(f"    TE ratio CV (小鼠) = {te_ratio_cv:.3f}")

# 是否"跨基因 TE 比较不可靠"
print(f"\n  诊断分析:")
print(f"  ① 正对照 (gene_length vs TE) r2 = {lr_all.rvalue**2:.4f}")
print(f"     → 连公认与 TE 相关的基因长度都无法解释任何方差")
if lr_all.rvalue**2 < 0.01:
    print(f"     判断: 【确认】TE 标签确实存在严重噪声或跨基因比较不适用")
else:
    print(f"     判断: TE 有一定信号")

print(f"\n  ② 分类 AUC (人类): ", end="")
mean_auc_h = np.mean(human_aucs) if human_aucs else 0
if mean_auc_h > 0.7:
    print(f"AUC={mean_auc_h:.3f} → 分类有一定区分力")
elif mean_auc_h > 0.6:
    print(f"AUC={mean_auc_h:.3f} → 分类能力中等")
elif mean_auc_h > 0.55:
    print(f"AUC={mean_auc_h:.3f} → 分类能力弱")
else:
    print(f"AUC={mean_auc_h:.3f} → 几乎无分类能力")

print(f"\n  ③ 小鼠表达量正对照 (mRNA→TE): ", end="")
if aucs_f:
    mean_auc_mrna = np.mean(aucs_f)
    if mean_auc_mrna > 0.6:
        print(f"AUC={mean_auc_mrna:.3f} → 表达量中蕴含 TE 信号")
    else:
        print(f"AUC={mean_auc_mrna:.3f} → 表达量与 TE 关系也很弱")

print(f"\n  >>> 综合结论:")
if lr_all.rvalue**2 < 0.01 and mean_auc_h < 0.6:
    print(f"      A) TE 标签噪声过大（跨基因 TE 比较本身不适用）: ✓ 主要问题")
    print(f"         原因: 正对照 r2≈0, 分类 AUC≈0.5, 数据不包含生物学重复")
    print(f"         建议: 改用配对设计 (如 KO vs WT 的 ΔTE), 或寻找更可靠的 TE 数据集")
    print(f"      B) Alu 特征信息量确实不足: ✓ 次要问题")
    print(f"         原因: 即使噪声问题解决，Alu 特征的有限变异也不足以解释 TE")
    print(f"         建议: 引入更多特征 (RNA-seq/ribo-seq 表达量, UTR 长度, 二级结构)")
    print(f"      C) 样本量不够: ✗ 不是主因")
    print(f"         样本已达 3,700+，即使只取 top/bottom 25% 也有 1,800+")
    print(f"      D) 需要换策略: ✓ 建议")
    print(f"         推荐: 基因内比较 (ΔTE = 含元件基因的 TE - 不含的), 或元素富集分析")
elif lr_all.rvalue**2 >= 0.01:
    print(f"      TE 数据本身有一定信号，但 Alu 特征预测力有限。")
    print(f"      建议补充更多特征后再尝试。")
else:
    print(f"      分类有一定区分力但回归失败，可能是 TE 动态范围大导致回归困难。")
    print(f"      建议维持分类策略，并尝试集成更多特征。")

print("\n" + "=" * 70)