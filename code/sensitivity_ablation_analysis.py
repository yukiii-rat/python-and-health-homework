# -*- coding: utf-8 -*-
"""
正对照 + 敏感性分析 + 特征消融（人类 Alu）
修改日期：2026-05-31
"""

import pandas as pd, numpy as np, re, sys, io, warnings
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

GTF_FILE = "data/gencode.v49.primary_assembly.annotation.gtf"
RANDOM_SEED = 42

# ============================================================
# 辅助：从原始 alu_exonic_mapping 重建特征矩阵（指定阈值）
# ============================================================
def build_feature_matrix_at_threshold(threshold):
    """根据 overlap_fraction 阈值重建基因级别特征矩阵"""
    exonic = pd.read_csv("output1/alu_exonic_mapping.csv")
    exonic = exonic[exonic['overlap_fraction'] >= threshold].copy()

    if len(exonic) == 0:
        return None

    rc = exonic.groupby(['gene_id','region']).size().unstack(fill_value=0)
    for c in ['5UTR','CDS','3UTR']: rc[c] = rc.get(c, 0)
    oc = exonic.groupby(['gene_id','orientation']).size().unstack(fill_value=0)
    for c in ['sense','antisense']: oc[c] = oc.get(c, 0)

    gn = exonic[['gene_id','gene_name']].drop_duplicates().set_index('gene_id')['gene_name'].to_dict()

    gf = pd.DataFrame(index=rc.index)
    gf['gene_name'] = gf.index.map(gn)
    gf['alu_exonic_total'] = exonic.groupby('gene_id').size()
    gf['alu_5utr'] = rc['5UTR'].values
    gf['alu_cds'] = rc['CDS'].values
    gf['alu_3utr'] = rc['3UTR'].values
    gf['alu_sense'] = oc['sense'].values
    gf['alu_antisense'] = oc['antisense'].values
    gf['alu_density_per_kb'] = 0.0  # will fill later
    gf['alu_3utr_ratio'] = (gf['alu_3utr'] / gf['alu_exonic_total']).replace([np.inf,-np.inf],0).fillna(0)
    gf['alu_antisense_ratio'] = (gf['alu_antisense'] / gf['alu_exonic_total']).replace([np.inf,-np.inf],0).fillna(0)
    gf['has_antisense_alu'] = (gf['alu_antisense'] > 0).astype(int)
    gf['has_3utr_alu'] = (gf['alu_3utr'] > 0).astype(int)

    # 亚家族
    def sf(sid):
        if sid.startswith('AluJ'): return 'aluJ'
        elif sid.startswith('AluS'): return 'aluS'
        elif sid.startswith('AluY'): return 'aluY'
        return 'other'
    exonic['subfamily'] = exonic['alu_id'].apply(sf)
    sc = exonic.groupby(['gene_id','subfamily']).size().unstack(fill_value=0)
    for c in ['aluJ','aluS','aluY','other']: gf[f'{c}_count'] = sc.get(c, 0).values

    # 内含子
    ig = pd.read_csv("output1/intronic_alu_per_transcript.csv").groupby('gene_id')['intronic_repeat_count'].sum()
    gf['intronic_alu_count'] = gf.index.map(lambda g: ig.get(g, 0))

    # 基因长度 / 外显子长度
    glens, ctx, tx2g = {}, set(), {}
    with open(GTF_FILE, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.strip().split('\t')
            if len(p) < 9: continue
            a = dict(re.findall(r'(\w+)\s+"([^"]*)"', p[8]))
            if p[2] == 'gene':
                gid = a['gene_id'].split('.')[0]; glens[gid] = (int(p[3]), int(p[4]))
            elif p[2] == 'transcript':
                if any(t in ('MANE_Select','basic') for t in re.findall(r'tag\s+"([^"]*)"', p[8])):
                    tid = a['transcript_id'].split('.')[0]; ctx.add(tid); tx2g[tid] = a['gene_id'].split('.')[0]

    gxi = defaultdict(set)
    with open(GTF_FILE, encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'): continue
            p = line.strip().split('\t')
            if len(p) < 9 or p[2] != 'exon': continue
            m = re.search(r'transcript_id "([^"]+)"', p[8])
            if not m: continue
            tid = m.group(1).split('.')[0]
            if tid not in ctx: continue
            gid = tx2g.get(tid,'')
            if gid: gxi[gid].add((int(p[3]), int(p[4])))

    gel = {}
    for gid, ivs in gxi.items():
        mg = []
        for s,e in sorted(ivs):
            if not mg or s > mg[-1][1]+1: mg.append([s,e])
            else: mg[-1][1] = max(mg[-1][1], e)
        gel[gid] = sum(e-s+1 for s,e in mg)

    gf['gene_length'] = gf.index.map(lambda g: glens[g][1]-glens[g][0]+1 if g in glens else 0)
    gf['exonic_length'] = gf.index.map(lambda g: gel.get(g, 0))
    gf['alu_density_per_kb'] = (gf['alu_exonic_total'] / (gf['exonic_length']/1000)).replace([np.inf,-np.inf],0).fillna(0)

    # 合并 TE
    te = pd.read_csv("output1/human_full_dataset.csv")[['gene_id','TE']].drop_duplicates('gene_id')
    gf = pd.merge(gf, te, on='gene_id', how='inner')
    return gf


# ============================================================
# 辅助：运行 XGBoost + RF 并返回 Test R²
# ============================================================
def run_models(df, feature_cols, group_labels=None, label_suffix=""):
    """执行标准 ML pipeline，返回各模型 Test R²"""
    binary_feats = [c for c in feature_cols if c in ('has_antisense_alu','has_3utr_alu')]
    numeric_feats = [c for c in feature_cols if c not in binary_feats]

    X = df[feature_cols].copy()
    y = df['TE'].values

    # 用 df 自带的 gene_id 构造 groups
    groups = df['gene_id'].values

    ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED).split(X, y, groups))
    X_tr, X_te = X.iloc[ti], X.iloc[vi]; y_tr, y_te = y[ti], y[vi]

    sc = StandardScaler()
    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
    if numeric_feats:
        X_tr_s[numeric_feats] = sc.fit_transform(X_tr[numeric_feats])
        X_te_s[numeric_feats] = sc.transform(X_te[numeric_feats])

    baseline_r2 = r2_score(y_te, np.full_like(y_te, y_tr.mean()))

    xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1)
    xgb_m.fit(X_tr_s, y_tr); xgb_r2 = r2_score(y_te, xgb_m.predict(X_te_s))

    rf_m = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=5,
                                  random_state=RANDOM_SEED, n_jobs=-1)
    rf_m.fit(X_tr_s, y_tr); rf_r2 = r2_score(y_te, rf_m.predict(X_te_s))

    return {'baseline': baseline_r2, 'XGBoost': xgb_r2, 'RF': rf_r2}


# ============================================================
# 1. 正对照：单变量预测
# ============================================================
print("=" * 60)
print("1. 正对照 (Positive Controls)")
print("=" * 60)

human = pd.read_csv("output1/human_full_dataset.csv")
te = human['TE'].values

pos_ctrl_features = ['gene_length', 'exonic_length', 'intronic_alu_count']
pos_results = []

for feat in pos_ctrl_features:
    X = human[[feat]].copy()
    ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED).split(X, te, human['gene_id'].values))
    y_tr, y_te = te[ti], te[vi]
    X_tr, X_te = X.iloc[ti], X.iloc[vi]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    # XGBoost with single feature
    xgb_m = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=-1)
    xgb_m.fit(X_tr_s, y_tr)
    xgb_r2 = r2_score(y_te, xgb_m.predict(X_te_s))

    # RF with single feature
    rf_m = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1)
    rf_m.fit(X_tr_s, y_tr)
    rf_r2 = r2_score(y_te, rf_m.predict(X_te_s))

    # Simple linear regression (Pearson r²)
    from scipy.stats import linregress
    lr = linregress(human[feat].values, te)
    lr_r2 = lr.rvalue ** 2

    print(f"\n  {feat}:")
    print(f"    Linear r² = {lr_r2:.4f} (p={lr.pvalue:.2e})")
    print(f"    XGBoost   = {xgb_r2:.4f}")
    print(f"    RF        = {rf_r2:.4f}")
    pos_results.append({'feature': feat, 'linear_r2': lr_r2, 'XGBoost': xgb_r2, 'RF': rf_r2})

pd.DataFrame(pos_results).to_csv("output1/positive_controls.csv", index=False)
print("\n已保存: output1/positive_controls.csv")


# ============================================================
# 2. 敏感性分析：不同 overlap 阈值
# ============================================================
print("\n" + "=" * 60)
print("2. 敏感性分析 (Overlap Threshold Sensitivity)")
print("=" * 60)

# 全量特征列
full_feature_cols = [
    'alu_exonic_total', 'alu_5utr', 'alu_cds', 'alu_3utr',
    'alu_sense', 'alu_antisense', 'alu_density_per_kb',
    'alu_3utr_ratio', 'alu_antisense_ratio',
    'has_antisense_alu', 'has_3utr_alu',
    'aluJ_count', 'aluS_count', 'aluY_count',
    'intronic_alu_count', 'gene_length', 'exonic_length',
]

thresholds = [0.5, 0.7, 0.8, 0.9, 1.0]
sens_results = []

for thresh in thresholds:
    df = build_feature_matrix_at_threshold(thresh)
    if df is None:
        print(f"\n  阈值 {thresh}: 无数据")
        continue
    avail_cols = [c for c in full_feature_cols if c in df.columns]
    res = run_models(df, avail_cols, label_suffix=f"_t{thresh}")
    res['threshold'] = thresh
    res['n_genes'] = len(df)
    sens_results.append(res)
    print(f"\n  阈值 {thresh} (基因数={len(df)}):")
    print(f"    Baseline R² = {res['baseline']:.4f}")
    print(f"    XGBoost R²  = {res['XGBoost']:.4f}")
    print(f"    RF R²       = {res['RF']:.4f}")

sens_df = pd.DataFrame(sens_results)
sens_df.to_csv("output1/sensitivity_analysis.csv", index=False)
print("\n已保存: output1/sensitivity_analysis.csv")


# ============================================================
# 3. 特征消融
# ============================================================
print("\n" + "=" * 60)
print("3. 特征消融 (Feature Ablation)")
print("=" * 60)

# 全量模型（所有特征）
df_full = pd.read_csv("output1/human_full_dataset.csv")
full_res = run_models(df_full, full_feature_cols)
print(f"\n  全量特征: XGBoost R²={full_res['XGBoost']:.4f}, RF R²={full_res['RF']:.4f}")

# 消融分组
ablation_groups = {
    '3UTR_group':     ['alu_3utr', 'alu_3utr_ratio', 'has_3utr_alu'],
    'Direction_group': ['alu_sense', 'alu_antisense', 'alu_antisense_ratio', 'has_antisense_alu'],
    'Subfamily_group': ['aluJ_count', 'aluS_count', 'aluY_count'],
    'Density_group':  ['alu_density_per_kb', 'alu_exonic_total'],
}

abl_results = [{'group': 'full', 'XGBoost': full_res['XGBoost'], 'RF': full_res['RF']}]

for grp_name, grp_feats in ablation_groups.items():
    remaining = [c for c in full_feature_cols if c not in grp_feats]
    res = run_models(df_full, remaining)
    res['group'] = grp_name
    res['n_features'] = len(remaining)
    abl_results.append(res)
    print(f"\n  去掉 {grp_name} (剩余{len(remaining)}特征):")
    print(f"    XGBoost R² = {res['XGBoost']:.4f} (delta={res['XGBoost']-full_res['XGBoost']:+.4f})")
    print(f"    RF R²      = {res['RF']:.4f} (delta={res['RF']-full_res['RF']:+.4f})")

abl_df = pd.DataFrame(abl_results)
abl_df.to_csv("output1/feature_ablation.csv", index=False)
print("\n已保存: output1/feature_ablation.csv")


# ============================================================
# 可视化：敏感性 + 消融
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左：敏感性分析
ax = axes[0]
ax.plot(sens_df['threshold'], sens_df['XGBoost'], 'o-', color='#e74c3c', lw=2, label='XGBoost')
ax.plot(sens_df['threshold'], sens_df['RF'], 's--', color='#3498db', lw=2, label='Random Forest')
ax.axhline(y=sens_df[sens_df['threshold']==0.8]['XGBoost'].values[0], color='#e74c3c', ls=':', alpha=0.5)
ax.axhline(y=sens_df[sens_df['threshold']==0.8]['RF'].values[0], color='#3498db', ls=':', alpha=0.5)
ax.set_xlabel('Exonic Overlap Threshold'); ax.set_ylabel('Test R²')
ax.set_title('Sensitivity Analysis: Overlap Threshold')
ax.legend(); ax.grid(True, alpha=0.3)
# 标注样本数
for _, row in sens_df.iterrows():
    ax.annotate(f"n={int(row['n_genes'])}", (row['threshold'], row['XGBoost']+0.005), ha='center', fontsize=8)

# 右：特征消融
ax = axes[1]
abl_plot = [r for r in abl_results if r['group'] != 'full']
grp_names = [r['group'].replace('_','\n') for r in abl_plot]
x = np.arange(len(grp_names))
w = 0.35
ax.bar(x - w/2, [r['XGBoost'] for r in abl_plot], w, label='XGBoost', color='#e74c3c')
ax.bar(x + w/2, [r['RF'] for r in abl_plot], w, label='RF', color='#3498db')
ax.axhline(y=full_res['XGBoost'], color='#e74c3c', ls='--', lw=1, label=f'Full XGBoost={full_res["XGBoost"]:.3f}')
ax.axhline(y=full_res['RF'], color='#3498db', ls=':', lw=1, label=f'Full RF={full_res["RF"]:.3f}')
ax.set_xticks(x); ax.set_xticklabels(grp_names, fontsize=9)
ax.set_ylabel('Test R²'); ax.set_title('Feature Ablation: R² after Removing Group')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output1/sensitivity_ablation.pdf', dpi=300)
plt.savefig('output1/sensitivity_ablation.png', dpi=150)
print("\n图已保存: output1/sensitivity_ablation.pdf / .png")
print("\n全部完成！")