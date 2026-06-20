# -*- coding: utf-8 -*-
"""
通用：构建指定 SINE 亚家族特征矩阵 + ML Pipeline
用法：python code/ml_pipeline_mouse_subfamily.py B2
"""
import pandas as pd, numpy as np, re, sys, io, warnings
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

SF = sys.argv[1]  # e.g. B1, B2
print(f"\n========== SINE {SF} Pipeline ==========")

# 亚家族 ID 映射
SF_IDS = {
    'B1': ['B1F','B1F1','B1F2','B1_Mm','B1_Mur1','B1_Mur2','B1_Mur3','B1_Mur4','B1_Mus1','B1_Mus2','ID_B1'],
    'B2': ['B2_Mm1a','B2_Mm1t','B2_Mm2'],
}
sf_ids = SF_IDS.get(SF, [sid for sid in pd.read_csv('output1/sine_exonic_mapping.csv')['sine_id'].unique() if sid.startswith(SF)])
print(f"{SF} IDs: {sf_ids}")

# 1. 特征矩阵
exonic_df = pd.read_csv("output1/sine_exonic_mapping.csv")
exonic_df = exonic_df[exonic_df['sine_id'].isin(sf_ids)].copy()
print(f"{SF} exonic records: {len(exonic_df)}")

region_counts = exonic_df.groupby(['gene_id','region']).size().unstack(fill_value=0)
for c in ['5UTR','CDS','3UTR']: region_counts[c] = region_counts.get(c, 0)
ori_counts = exonic_df.groupby(['gene_id','orientation']).size().unstack(fill_value=0)
for c in ['sense','antisense']: ori_counts[c] = ori_counts.get(c, 0)
gn_map = exonic_df[['gene_id','gene_name']].drop_duplicates().set_index('gene_id')['gene_name'].to_dict()

gf = pd.DataFrame(index=region_counts.index)
gf['gene_name'] = gf.index.map(gn_map)
gf[f'{SF}_exonic_total'] = exonic_df.groupby('gene_id').size()
gf[f'{SF}_5utr'] = region_counts['5UTR'].values
gf[f'{SF}_cds'] = region_counts['CDS'].values
gf[f'{SF}_3utr'] = region_counts['3UTR'].values
gf[f'{SF}_sense'] = ori_counts['sense'].values
gf[f'{SF}_antisense'] = ori_counts['antisense'].values

# 内含子
intronic_df = pd.read_csv("output1/intronic_sine_per_transcript.csv")
ig = intronic_df.groupby('gene_id')['intronic_repeat_count'].sum()
gf[f'intronic_{SF}_count'] = gf.index.map(lambda g: ig.get(g, 0))

# 基因长度
gtf = "data/gencode.vM38.primary_assembly.annotation.gtf"
glens, ctx, tx2g = {}, set(), {}
with open(gtf, encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.strip().split('\t')
        if len(p) < 9: continue
        a = dict(re.findall(r'(\w+)\s+"([^"]*)"', p[8]))
        if p[2] == 'gene':
            gid = a['gene_id'].split('.')[0]; glens[gid] = (int(p[3]), int(p[4]))
        elif p[2] == 'transcript':
            if any(t == 'basic' for t in re.findall(r'tag\s+"([^"]*)"', p[8])):
                tid = a['transcript_id'].split('.')[0]; ctx.add(tid); tx2g[tid] = a['gene_id'].split('.')[0]

gxi = defaultdict(set)
with open(gtf, encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'): continue
        p = line.strip().split('\t')
        if len(p) < 9 or p[2] != 'exon': continue
        m = re.search(r'transcript_id "([^"]+)"', p[8])
        if not m: continue
        tid = m.group(1).split('.')[0]
        if tid not in ctx: continue
        gid = tx2g.get(tid, '')
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

gf[f'{SF}_density_per_kb'] = (gf[f'{SF}_exonic_total'] / (gf['exonic_length']/1000)).replace([np.inf,-np.inf],0).fillna(0)
gf[f'{SF}_3utr_ratio'] = (gf[f'{SF}_3utr'] / gf[f'{SF}_exonic_total']).replace([np.inf,-np.inf],0).fillna(0)
gf[f'{SF}_antisense_ratio'] = (gf[f'{SF}_antisense'] / gf[f'{SF}_exonic_total']).replace([np.inf,-np.inf],0).fillna(0)
gf[f'has_antisense_{SF}'] = (gf[f'{SF}_antisense'] > 0).astype(int)
gf[f'has_3utr_{SF}'] = (gf[f'{SF}_3utr'] > 0).astype(int)

out_cols = ['gene_name',f'{SF}_exonic_total',f'{SF}_5utr',f'{SF}_cds',f'{SF}_3utr',
            f'{SF}_sense',f'{SF}_antisense',f'{SF}_density_per_kb',f'{SF}_3utr_ratio',f'{SF}_antisense_ratio',
            f'has_antisense_{SF}',f'has_3utr_{SF}',f'intronic_{SF}_count','gene_length','exonic_length']
odf = gf[out_cols].copy().sort_values(f'{SF}_exonic_total', ascending=False).reset_index().rename(columns={'index':'gene_id'})
odf.to_csv(f"output1/feature_matrix_mouse_{SF}.csv", index=False)
print(f"Feature matrix: {len(odf)} genes")

# 2. Merge TE
te = pd.read_csv("output1/mouse_full_dataset.csv")[['gene_id','TE']].drop_duplicates('gene_id')
df = pd.merge(odf, te, on='gene_id', how='inner')
print(f"After TE merge: {len(df)} genes")

fcols = [c for c in df.columns if c not in ('TE','gene_id','gene_name')]
X, y, groups = df[fcols].copy(), df['TE'].values, df['gene_id'].values
bf = [c for c in [f'has_antisense_{SF}',f'has_3utr_{SF}'] if c in fcols]
nf = [c for c in fcols if c not in bf]

# 3. 分割
ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
X_tr, X_te = X.iloc[ti], X.iloc[vi]; y_tr, y_te = y[ti], y[vi]
print(f"Train: {len(ti)}, Test: {len(vi)}")

# 4. Scale
sc = StandardScaler()
X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
X_tr_s[nf] = sc.fit_transform(X_tr[nf]); X_te_s[nf] = sc.transform(X_te[nf])

# 5. Baseline
print(f"\nBaseline: Mean R2={r2_score(y_te,np.full_like(y_te,y_tr.mean())):.4f}, Median R2={r2_score(y_te,np.full_like(y_te,np.median(y_tr))):.4f}")

# 6. 模型
for name, model in [
    ("XGBoost", xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
    ("Random Forest", RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1))
]:
    model.fit(X_tr_s, y_tr); yp = model.predict(X_te_s)
    rr, pp = pearsonr(y_te, yp)
    print(f"\n{name}: R2={r2_score(y_te,yp):.4f}, r={rr:.4f}, p={pp:.2e}, MSE={mean_squared_error(y_te,yp):.4f}")
    kf = KFold(5, shuffle=True, random_state=42)
    cvr2, cvr = [], []
    for ti2, vi2 in kf.split(X_tr_s, y_tr, groups[ti]):
        m = model.__class__(**model.get_params()); m.fit(X_tr_s.iloc[ti2], y_tr[ti2])
        yvp = m.predict(X_tr_s.iloc[vi2])
        cvr2.append(r2_score(y_tr[vi2], yvp)); cvr.append(pearsonr(y_tr[vi2], yvp)[0])
    print(f"CV R2={np.mean(cvr2):.4f}+/-{np.std(cvr2):.4f}, CV r={np.mean(cvr):.4f}+/-{np.std(cvr):.4f}")

    # Feature importance (XGBoost only)
    if name == "XGBoost":
        fi = pd.DataFrame({'feature': fcols, 'importance': model.feature_importances_}).sort_values('importance', ascending=True)
        irank = fi['feature'].tolist().index(f'intronic_{SF}_count') + 1 if f'intronic_{SF}_count' in fi['feature'].values else 'N/A'
        print(f"\nintronic_{SF}_count rank: {irank}/{len(fi)}")
        print(fi.sort_values('importance', ascending=False).to_string(index=False))

        # 画图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes[0,0].scatter(y_te, yp, alpha=0.5, s=15, c='steelblue')
        vm = [min(y_te.min(),yp.min()), max(y_te.max(),yp.max())]
        axes[0,0].plot(vm, vm, 'r--', lw=1)
        axes[0,0].set_xlabel('True TE'); axes[0,0].set_ylabel('Predicted TE')
        axes[0,0].set_title(f'{SF} XGBoost: R2={r2_score(y_te,yp):.3f}, r={rr:.3f}')

        axes[0,1].scatter(yp, y_te-yp, alpha=0.5, s=15, c='coral')
        axes[0,1].axhline(0, color='gray', ls='--', lw=1)
        axes[0,1].set_xlabel('Predicted TE'); axes[0,1].set_ylabel('Residuals')

        axes[1,0].hist(y_te, bins=50, alpha=0.6, label='True', color='steelblue', density=True)
        axes[1,0].hist(yp, bins=50, alpha=0.6, label='Predicted', color='coral', density=True)
        axes[1,0].set_xlabel('TE'); axes[1,0].legend()

        c = ['coral' if f == f'intronic_{SF}_count' else 'steelblue' for f in fi['feature']]
        axes[1,1].barh(range(len(fi)), fi['importance'], color=c)
        axes[1,1].set_yticks(range(len(fi))); axes[1,1].set_yticklabels(fi['feature'], fontsize=8)
        axes[1,1].set_title(f'intronic_{SF}_count: #{irank}/{len(fi)}')

        plt.tight_layout(); plt.savefig(f'output1/mouse_{SF}_ml_results.png', dpi=150)
        print(f"\nSaved: output1/mouse_{SF}_ml_results.png")

    if name == "Random Forest":
        rf_pred = yp

# 最后统一写入结果
all_results = {}
for name, model in [
    ("XGBoost", xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
    ("Random Forest", RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1))
]:
    model.fit(X_tr_s, y_tr); yp = model.predict(X_te_s)
    rr, pp = pearsonr(y_te, yp)
    all_results[name] = (r2_score(y_te, yp), rr, pp, mean_squared_error(y_te, yp), mean_absolute_error(y_te, yp))

with open(f"output1/model_results_mouse_{SF}.txt", "w", encoding='utf-8') as f:
    f.write(f"===== Mouse {SF} SINE TE Prediction Results =====\n")
    f.write(f"Samples: {len(df)}, Features: {len(fcols)}\n\n")
    for name, (r2v, rv, pv, mse, mae) in all_results.items():
        f.write(f"{name}: R2={r2v:.4f}, r={rv:.4f}, p={pv:.2e}, MSE={mse:.4f}, MAE={mae:.4f}\n")
    f.write(f"\nintronic_{SF}_count rank: {irank}/{len(fi)}\n")

print(f"\nDone! Results: output1/model_results_mouse_{SF}.txt")
