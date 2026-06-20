"""
============================================================================
Full-region XGBoost regression — region_inserted as a feature
============================================================================
Input:  output/ml_right/micro_ml_features_final_integrated.csv (11,704 rows)
Output: output/ml_right/ml_full_region/

Strategy:
  - All regions combined (5'UTR + CDS + 3'UTR), region_inserted as feature
  - Tuned hyperparams from 3'UTR optimization
  - SHAP to show region directionality
============================================================================
"""
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ====================================================================
# Paths
# ====================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
INPUT_CSV = os.path.join(PROJECT_ROOT, 'output', 'ml_right',
                         'micro_ml_features_final_integrated.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'ml_right',
                          'ml_full_region')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# Config
# ====================================================================
DROP_COLS = [
    'transcript_id', 'gene_id', 'gene_name', 'alu_id',
    'alu_unique_id', 'alu_chr_start', 'alu_chr_end', 'alu_strand',
    'strand', 'chrom', 'log2_dist_to_aug', 'log2_dist_to_stop',
]

# Tuned hyperparams from 3'UTR optimization
BEST_PARAMS = dict(
    n_estimators=1500,
    max_depth=10,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.4,
    reg_lambda=2,
    reg_alpha=0.1,
    random_state=42,
    verbosity=0,
)

DEP_VARS = [
    'gc_rich_stem_density', 'overlap_bp', 'sl1_terminal_mfe',
    'spliced_utr_dist', 'alu_full_mfe', 'alu_gc_content', 'rrna_18s_score',
]


def write_report(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    print('=' * 60)
    print("Full-region XGBoost regression (region_inserted as feature)")
    print('=' * 60)

    # ---- 1. Load data ----
    print(f'\n[1] Loading data...')
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=['l2fc'])
    print(f'  Total: {len(df)} rows')

    # Per-region stats
    for r in ['CDS', "5'UTR", '3UTR']:
        if r == "5'UTR":
            sub = df[df['region_inserted'].str.contains("5'", na=False)]
        else:
            sub = df[df['region_inserted'] == r]
        print(f'  {r}: {len(sub):>5}  l2fc mean={sub["l2fc"].mean():+.4f}  '
              f'median={sub["l2fc"].median():+.4f}')

    # ---- 2. Feature engineering ----
    print(f'\n[2] Feature engineering...')
    y = df['l2fc'].copy()
    X = df.drop(columns=DROP_COLS, errors='ignore')
    X = X.drop(columns=['l2fc'], errors='ignore')

    # One-hot: region_inserted + subfamilies
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f'  Categorical columns: {cat_cols}')
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)
    X = X.drop(columns=[c for c in X.columns if c.startswith('chrom_')],
               errors='ignore')
    assert X.shape[1] == X.select_dtypes(include=[np.number]).shape[1]
    print(f'  Features: {X.shape[1]}')
    print(f'  Samples:  {X.shape[0]}')

    # Get region dummy column names
    region_cols = [c for c in X.columns
                   if c.startswith('region_inserted_')]
    print(f'  Region dummies: {region_cols}')

    # ---- 3. 5-fold CV ----
    print(f'\n[3] 5-fold cross-validation...')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2s, cv_rmses = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        m = xgb.XGBRegressor(**BEST_PARAMS)
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)

        r2 = r2_score(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        cv_r2s.append(r2)
        cv_rmses.append(rmse)
        print(f'  Fold {fold}: R2={r2:.4f}, RMSE={rmse:.4f}')

    print(f'\n  CV R2 = {np.mean(cv_r2s):.4f} +/- {np.std(cv_r2s):.4f}')
    print(f'  CV RMSE = {np.mean(cv_rmses):.4f} +/- {np.std(cv_rmses):.4f}')

    # ---- 4. Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'\n[4] Train/test split: {X_train.shape[0]}/{X_test.shape[0]}')

    # ---- 5. Train ----
    print(f'\n[5] Training XGBoost with tuned params...')
    for k, v in BEST_PARAMS.items():
        print(f'    {k} = {v}')

    model = xgb.XGBRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f'\n  MSE  = {test_mse:.6f}')
    print(f'  RMSE = {test_rmse:.6f}')
    print(f'  MAE  = {test_mae:.6f}')
    print(f'  R2   = {test_r2:.4f}')

    # ---- 5. Full-data final model + SHAP ----
    print(f'\n[6] Training final model on full data + SHAP...')
    final_model = xgb.XGBRegressor(**BEST_PARAMS)
    final_model.fit(X, y)

    gain_imp = final_model.feature_importances_
    gain_order = np.argsort(gain_imp)[::-1]
    print(f'  Gain importance Top 5:')
    for rank, i in enumerate(gain_order[:5], 1):
        print(f'    {rank}. {X.columns[i]}: {gain_imp[i]:.4f}')

    # SHAP
    print(f'  Computing SHAP (this may take a moment)...')
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    shap_imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(shap_imp)[::-1]

    # Beeswarm
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> shap_beeswarm.png')

    # Bar
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type='bar', show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> shap_bar.png')

    # Dependence: region dummies + DEP_VARS
    plot_vars = region_cols + [v for v in DEP_VARS if v in X.columns]
    for var in plot_vars:
        if var not in X.columns:
            continue
        idx = list(X.columns).index(var)
        vals = X[var].values
        sv = shap_values[:, idx]
        corr = np.corrcoef(vals, sv)[0, 1]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        ax.scatter(vals, sv, alpha=0.15, s=6, c='steelblue', edgecolors='none')
        o = np.argsort(vals)
        rn = max(1, min(len(vals) // 50, 200))
        ax.plot(np.sort(vals), uniform_filter1d(sv[o], size=rn),
                'r-', lw=2.5, alpha=0.9)
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xlabel(var, fontsize=10)
        ax.set_ylabel('SHAP value', fontsize=10)
        ax.set_title(f"Full-region -- {var}  (r={corr:+.3f})", fontsize=11)
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        # Clean var name for filename
        fname = var.replace('region_inserted_', 'region_')
        fig.savefig(os.path.join(OUTPUT_DIR, f'dep_{fname}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  -> dep_{fname}.png')

    # ---- 6. Write report ----
    print(f'\n[7] Writing report...')

    lines = [
        '=' * 70,
        "  Full-region XGBoost Regression Results",
        '=' * 70,
        f'  N={len(X)}, features={X.shape[1]}',
        '',
        '  Per-region l2fc:',
    ]
    for r in ['CDS', "5'UTR", '3UTR']:
        if r == "5'UTR":
            sub = df[df['region_inserted'].str.contains("5'", na=False)]
        else:
            sub = df[df['region_inserted'] == r]
        lines.append(f'    {r}: n={len(sub):>5}, mean={sub["l2fc"].mean():+.4f}, '
                     f'median={sub["l2fc"].median():+.4f}')

    lines.extend([
        '',
        '  Hyperparameters (tuned):',
    ])
    for k, v in BEST_PARAMS.items():
        if k in ('random_state', 'verbosity'):
            continue
        lines.append(f'    {k} = {v}')

    lines.extend([
        '',
        f'  5-fold CV:',
        f'    R2   = {np.mean(cv_r2s):.4f} +/- {np.std(cv_r2s):.4f}',
        f'    RMSE = {np.mean(cv_rmses):.4f} +/- {np.std(cv_rmses):.4f}',
        '',
        f'  Test set (20% holdout):',
        f'    MSE  = {test_mse:.6f}',
        f'    RMSE = {test_rmse:.6f}',
        f'    MAE  = {test_mae:.6f}',
        f'    R2   = {test_r2:.4f}',
        '',
        '=' * 70,
        '  SHAP global importance (top 20)',
        '=' * 70,
        f'  {"#":>3s}  {"Feature":<38s}  {"mean|SHAP|":<12s}  {"SHAP%":<8s}  {"r(SHAP,val)":<12s}',
        f'  {"-"*3}  {"-"*38}  {"-"*12}  {"-"*8}  {"-"*12}',
    ])

    total_imp = shap_imp.sum()
    for rank, i in enumerate(order[:20], 1):
        col = X.columns[i]
        vals = X[col].values
        s = shap_values[:, i]
        corr = np.corrcoef(vals, s)[0, 1] if np.std(vals) * np.std(s) > 0 else 0
        pct = shap_imp[i] / total_imp * 100
        lines.append(
            f'  {rank:3d}  {col:<38s}  {shap_imp[i]:<12.6f}  {pct:<8.2f}  {corr:<+12.4f}'
        )

    # Region SHAP direction summary
    lines.extend([
        '',
        '=' * 70,
        '  Region dummy SHAP direction',
        '=' * 70,
    ])
    for col in region_cols:
        idx = list(X.columns).index(col)
        vals = X[col].values
        sv = shap_values[:, idx]
        mean_shap = sv.mean()
        lines.append(f'  {col:<35s}  mean_SHAP={mean_shap:+.6f}  '
                     f'({"+" if mean_shap > 0 else "-"} -> TE up)')

    write_report(os.path.join(OUTPUT_DIR, 'full_region_report.txt'), lines)
    print(f'  -> full_region_report.txt')

    # ---- 7. Feature importance CSV ----
    imp_df = pd.DataFrame({
        'feature': X.columns,
        'shap_mean_abs': shap_imp,
        'gain_importance': gain_imp,
    }).sort_values('shap_mean_abs', ascending=False)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    print(f'  -> feature_importance.csv')

    print(f'\n{"=" * 60}')
    print(f'Done! Results saved to: {OUTPUT_DIR}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
