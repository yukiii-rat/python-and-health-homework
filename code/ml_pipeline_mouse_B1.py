# -*- coding: utf-8 -*-
"""
小鼠 B1 SINE 亚族 TE 预测 ML Pipeline
输入：feature_matrix_mouse_B1.csv + mouse_full_dataset.csv (TE)
输出：model_results_mouse_B1.txt + 图
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# 1. 读取 B1 特征矩阵
fm = pd.read_csv("output1/feature_matrix_mouse_B1.csv")

# 2. 从 mouse_full_dataset.csv 取 TE 列
mouse_full = pd.read_csv("output1/mouse_full_dataset.csv")
te_map = mouse_full[['gene_id', 'TE']].drop_duplicates('gene_id')

# merge
df = pd.merge(fm, te_map, on='gene_id', how='inner')
print(f"B1 feature matrix: {len(fm)} genes")
print(f"Merged with TE: {len(df)} genes")

feature_cols = [c for c in df.columns if c not in ('TE', 'gene_id', 'gene_name')]
X = df[feature_cols].copy()
y = df['TE'].values
groups = df['gene_id'].values

binary_features = ['has_antisense_B1', 'has_3utr_B1']
binary_features = [c for c in binary_features if c in feature_cols]
numeric_features = [c for c in feature_cols if c not in binary_features]

print(f"特征数: {len(feature_cols)}, 二值: {len(binary_features)}")

# 3. 80/20 分组分割
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# 4. Scale
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# 5. Baseline
baseline_mean = np.full_like(y_test, y_train.mean())
baseline_median = np.full_like(y_test, np.median(y_train))
print(f"\n===== Baseline =====")
print(f"Mean pred R2 : {r2_score(y_test, baseline_mean):.4f}")
print(f"Median pred R2: {r2_score(y_test, baseline_median):.4f}")

# 6. 模型
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    r2 = r2_score(y_te, yp); rr, pp = pearsonr(y_te, yp)
    mse = mean_squared_error(y_te, yp); mae = mean_absolute_error(y_te, yp)
    print(f"\n  {name}: R2={r2:.4f}, r={rr:.4f}, p={pp:.2e}, MSE={mse:.4f}, MAE={mae:.4f}")
    return model, yp

def cv_eval(name, model, X_tr, y_tr, g_tr):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s, rs = [], []
    for ti, vi in kf.split(X_tr, y_tr, g_tr):
        m = model.__class__(**model.get_params())
        m.fit(X_tr.iloc[ti], y_tr[ti])
        yvp = m.predict(X_tr.iloc[vi])
        r2s.append(r2_score(y_tr[vi], yvp)); rs.append(pearsonr(y_tr[vi], yvp)[0])
    print(f"  CV R2={np.mean(r2s):.4f}+/-{np.std(r2s):.4f}, CV r={np.mean(rs):.4f}+/-{np.std(rs):.4f}")
    return np.mean(r2s), np.mean(rs)

print("\n========== B1 模型训练 ==========")

xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_m, xgb_p = evaluate("XGBoost", xgb_m, X_train_scaled, y_train, X_test_scaled, y_test)
cv_eval("XGBoost", xgb_m, X_train_scaled, y_train, groups[train_idx])

rf_m = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=5,
                              random_state=42, n_jobs=-1)
rf_m, rf_p = evaluate("Random Forest", rf_m, X_train_scaled, y_train, X_test_scaled, y_test)
cv_eval("Random Forest", rf_m, X_train_scaled, y_train, groups[train_idx])

# 7. 特征重要性
imp = xgb_m.feature_importances_
fi = pd.DataFrame({'feature': feature_cols, 'importance': imp}).sort_values('importance', ascending=True)
irank = fi['feature'].tolist().index('intronic_B1_count') + 1
print(f"\nintronic_B1_count 排位: #{irank}/{len(fi)}")
print(fi.sort_values('importance', ascending=False).to_string(index=False))

# 8. 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
ax.scatter(y_test, xgb_p, alpha=0.5, s=15, c='steelblue')
vm = [min(y_test.min(), xgb_p.min()), max(y_test.max(), xgb_p.max())]
ax.plot(vm, vm, 'r--', lw=1)
ax.set_xlabel('True TE'); ax.set_ylabel('Predicted TE')
rr, pp = pearsonr(y_test, xgb_p)
ax.set_title(f'B1 XGBoost: Predicted vs True\nR2={r2_score(y_test, xgb_p):.3f}, r={rr:.3f}, p={pp:.2e}')

ax = axes[0, 1]
res = y_test - xgb_p
ax.scatter(xgb_p, res, alpha=0.5, s=15, c='coral')
ax.axhline(y=0, color='gray', ls='--', lw=1)
ax.set_xlabel('Predicted TE'); ax.set_ylabel('Residuals'); ax.set_title(f'Residual Plot\nsigma={res.std():.3f}')

ax = axes[1, 0]
ax.hist(y_test, bins=50, alpha=0.6, label='True', color='steelblue', density=True)
ax.hist(xgb_p, bins=50, alpha=0.6, label='Predicted', color='coral', density=True)
ax.set_xlabel('TE'); ax.set_ylabel('Density'); ax.set_title('TE Distribution'); ax.legend()

ax = axes[1, 1]
colors = ['coral' if f == 'intronic_B1_count' else 'steelblue' for f in fi['feature']]
ax.barh(range(len(fi)), fi['importance'], color=colors)
ax.set_yticks(range(len(fi))); ax.set_yticklabels(fi['feature'], fontsize=8)
ax.set_xlabel('Importance')
ax.set_title(f'Feature Importance (XGBoost)\nintronic_B1_count: #{irank}/{len(fi)}')
for i, (_, r) in enumerate(fi.iterrows()):
    if r['feature'] == 'intronic_B1_count':
        ax.annotate(f'  #{irank}', xy=(r['importance'], i), va='center', color='coral', fontweight='bold')

plt.tight_layout()
plt.savefig('output1/mouse_B1_ml_results.png', dpi=150)
print("\n图已保存: output1/mouse_B1_ml_results.png")

# 9. 输出
with open("output1/model_results_mouse_B1.txt", "w", encoding='utf-8') as f:
    f.write("===== Mouse B1 SINE TE Prediction Results =====\n")
    f.write(f"Samples: {len(df)}, Features: {len(feature_cols)}\n\n")
    for name, yp in [("XGBoost", xgb_p), ("Random Forest", rf_p)]:
        rr, pp = pearsonr(y_test, yp)
        f.write(f"----- {name} -----\n")
        f.write(f"Test R2  = {r2_score(y_test, yp):.4f}\n")
        f.write(f"r        = {rr:.4f}\n")
        f.write(f"p        = {pp:.2e}\n")
        f.write(f"MSE      = {mean_squared_error(y_test, yp):.4f}\n")
        f.write(f"MAE      = {mean_absolute_error(y_test, yp):.4f}\n\n")
    f.write("----- Feature Importance -----\n")
    for _, r in fi.sort_values('importance', ascending=False).iterrows():
        f.write(f"  {r['feature']}: {r['importance']:.4f}\n")
    f.write(f"\nintronic_B1_count rank: #{irank}/{len(fi)}\n")

print("结果已保存: output1/model_results_mouse_B1.txt")