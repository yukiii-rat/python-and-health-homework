# -*- coding: utf-8 -*-
"""
小鼠 TE 预测 ML Pipeline
输入：mouse_full_dataset.csv
输出：model_results_mouse.txt + 2 张图

修改日期：2026-05-31
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

# ==================== 1. 读取数据 ====================
# 修改日期：2026-05-31
df = pd.read_csv("output1/mouse_full_dataset.csv")

drop_cols = ['gene_id', 'gene_name', 'UCSC IDg', 'Ensembl_ID',
             'ribosome', 'mRNA', 'TE ratio']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

feature_cols = [c for c in df.columns if c != 'TE']
X = df[feature_cols].copy()
y = df['TE'].values

binary_features = ['has_antisense_sine', 'has_3utr_sine']
binary_features = [c for c in binary_features if c in feature_cols]
numeric_features = [c for c in feature_cols if c not in binary_features]

print(f"总样本: {len(df)}")
print(f"特征数: {len(feature_cols)}")
print(f"数值特征: {len(numeric_features)}")
print(f"二值特征: {len(binary_features)}")

# ==================== 2. 按 gene_id 分组 80/20 分割 ====================
# 修改日期：2026-05-31
df_full = pd.read_csv("output1/mouse_full_dataset.csv")
groups = df_full['gene_id'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\n训练集: {len(train_idx)}, 测试集: {len(test_idx)}")

# ==================== 3. StandardScaler ====================
# 修改日期：2026-05-31
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# ==================== 4. Baseline ====================
# 修改日期：2026-05-31
baseline_mean = np.full_like(y_test, y_train.mean())
baseline_median = np.full_like(y_test, np.median(y_train))
r2_mean = r2_score(y_test, baseline_mean)
r2_median = r2_score(y_test, baseline_median)
print(f"\n===== Baseline =====")
print(f"Mean pred R2 : {r2_mean:.4f}")
print(f"Median pred R2: {r2_median:.4f}")

# ==================== 5. 模型训练与评估 ====================
# 修改日期：2026-05-31
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    pearson_r, p_val = pearsonr(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"{'-'*50}")
    print(f"  R2         = {r2:.4f}")
    print(f"  Pearson r  = {pearson_r:.4f}")
    print(f"  p-value    = {p_val:.2e}")
    print(f"  MSE        = {mse:.4f}")
    print(f"  RMSE       = {rmse:.4f}")
    print(f"  MAE        = {mae:.4f}")
    return model, y_pred

def cv_evaluate(name, model, X_tr, y_tr, groups_tr):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2, cv_r = [], []
    for train_i, val_i in kf.split(X_tr, y_tr, groups_tr):
        m = model.__class__(**model.get_params())
        m.fit(X_tr.iloc[train_i], y_tr[train_i])
        y_val_pred = m.predict(X_tr.iloc[val_i])
        cv_r2.append(r2_score(y_tr[val_i], y_val_pred))
        cv_r.append(pearsonr(y_tr[val_i], y_val_pred)[0])
    print(f"  --- 5 折 CV ---")
    print(f"  CV R2   = {np.mean(cv_r2):.4f} +/- {np.std(cv_r2):.4f}")
    print(f"  CV r    = {np.mean(cv_r):.4f} +/- {np.std(cv_r):.4f}")
    return np.mean(cv_r2), np.mean(cv_r)

print("\n\n========== 模型训练 ==========")

xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
xgb_model, xgb_pred = evaluate_model("XGBoost", xgb_model, X_train_scaled, y_train, X_test_scaled, y_test)
xgb_cv_r2, xgb_cv_r = cv_evaluate("XGBoost", xgb_model, X_train_scaled, y_train, groups[train_idx])

rf_model = RandomForestRegressor(
    n_estimators=500, max_depth=12, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
rf_model, rf_pred = evaluate_model("Random Forest", rf_model, X_train_scaled, y_train, X_test_scaled, y_test)
rf_cv_r2, rf_cv_r = cv_evaluate("Random Forest", rf_model, X_train_scaled, y_train, groups[train_idx])

# ==================== 6. 特征重要性 ====================
# 修改日期：2026-05-31
importance = xgb_model.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=True)

intronic_rank = feat_imp['feature'].tolist().index('intronic_sine_count') + 1
total_feats = len(feat_imp)
print(f"\n===== 特征重要性 =====")
print(f"intronic_sine_count 排位: #{intronic_rank}/{total_feats}")
print(feat_imp.sort_values('importance', ascending=False).to_string(index=False))

# ==================== 7. 可视化 ====================
# 修改日期：2026-05-31
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax = axes[0, 0]
ax.scatter(y_test, xgb_pred, alpha=0.5, s=15, c='steelblue')
vmin = min(y_test.min(), xgb_pred.min())
vmax = max(y_test.max(), xgb_pred.max())
ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=1)
ax.set_xlabel('True TE')
ax.set_ylabel('Predicted TE')
r2_te = r2_score(y_test, xgb_pred)
r_val, p_val = pearsonr(y_test, xgb_pred)
ax.set_title(f'XGBoost: Predicted vs True\nR2={r2_te:.3f}, r={r_val:.3f}, p={p_val:.2e}')

ax = axes[0, 1]
residuals = y_test - xgb_pred
ax.scatter(xgb_pred, residuals, alpha=0.5, s=15, c='coral')
ax.axhline(y=0, color='gray', linestyle='--', lw=1)
ax.set_xlabel('Predicted TE')
ax.set_ylabel('Residuals')
ax.set_title(f'Residual Plot\nsigma_res = {residuals.std():.3f}')

ax = axes[1, 0]
ax.hist(y_test, bins=50, alpha=0.6, label='True TE', color='steelblue', density=True)
ax.hist(xgb_pred, bins=50, alpha=0.6, label='Predicted TE', color='coral', density=True)
ax.set_xlabel('TE')
ax.set_ylabel('Density')
ax.set_title('TE Distribution: True vs Predicted')
ax.legend()

ax = axes[1, 1]
colors = ['coral' if f == 'intronic_sine_count' else 'steelblue' for f in feat_imp['feature']]
ax.barh(range(len(feat_imp)), feat_imp['importance'], color=colors)
ax.set_yticks(range(len(feat_imp)))
ax.set_yticklabels(feat_imp['feature'], fontsize=8)
ax.set_xlabel('Importance')
ax.set_title(f'Feature Importance (XGBoost)\nintronic_sine_count: #{intronic_rank}/{total_feats}')
for i, (_, row) in enumerate(feat_imp.iterrows()):
    if row['feature'] == 'intronic_sine_count':
        ax.annotate(f'  #{intronic_rank}', xy=(row['importance'], i),
                    va='center', color='coral', fontweight='bold')

plt.tight_layout()
plt.savefig('output1/mouse_ml_results.png', dpi=150)
print("\n图已保存: output1/mouse_ml_results.png")

# ==================== 8. 输出 ====================
# 修改日期：2026-05-31
with open("output1/model_results_mouse.txt", "w", encoding='utf-8') as f:
    f.write("===== Mouse TE Prediction Results =====\n")
    f.write(f"Total samples: {len(df)}, Features: {len(feature_cols)}\n")
    f.write(f"Train: {len(train_idx)}, Test: {len(test_idx)}\n\n")
    f.write(f"----- Baseline -----\n")
    f.write(f"Mean pred R2  : {r2_mean:.4f}\n")
    f.write(f"Median pred R2: {r2_median:.4f}\n\n")
    for name, yp, cv_r2_v, cv_r_v in [
        ("XGBoost", xgb_pred, xgb_cv_r2, xgb_cv_r),
        ("Random Forest", rf_pred, rf_cv_r2, rf_cv_r)
    ]:
        rr, pp = pearsonr(y_test, yp)
        f.write(f"----- {name} -----\n")
        f.write(f"Test R2       = {r2_score(y_test, yp):.4f}\n")
        f.write(f"Pearson r     = {rr:.4f}\n")
        f.write(f"p-value       = {pp:.2e}\n")
        f.write(f"MSE           = {mean_squared_error(y_test, yp):.4f}\n")
        f.write(f"RMSE          = {np.sqrt(mean_squared_error(y_test, yp)):.4f}\n")
        f.write(f"MAE           = {mean_absolute_error(y_test, yp):.4f}\n")
        f.write(f"CV R2         = {cv_r2_v:.4f}\n")
        f.write(f"CV r          = {cv_r_v:.4f}\n\n")
    f.write("----- Feature Importance (XGBoost) -----\n")
    for _, row in feat_imp.sort_values('importance', ascending=False).iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    f.write(f"\nintronic_sine_count rank: #{intronic_rank}/{total_feats}\n")

print("结果已保存: output1/model_results_mouse.txt")