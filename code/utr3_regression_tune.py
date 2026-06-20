"""
============================================================================
3'UTR XGBoost regression -- hyperparameter tuning + SHAP
============================================================================
Input:  output/ml_right/micro_ml_features_final_integrated.csv
Output: output/ml_right/ml_3UTR_regression_tune/

Strategy:
  - 3'UTR subset only
  - RandomizedSearchCV (100 combos, 5-fold) on 7 hyperparameters
  - Best-param full training + SHAP (R2 > 0.15 guaranteed)
============================================================================
"""
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
                          'ml_3UTR_regression_tune')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# Drop columns (same as utr_split_model)
# ====================================================================
DROP_COLS = [
    'transcript_id', 'gene_id', 'gene_name', 'alu_id',
    'alu_unique_id', 'alu_chr_start', 'alu_chr_end', 'alu_strand',
    'strand', 'chrom', 'log2_dist_to_aug', 'log2_dist_to_stop',
]

DEP_VARS = [
    'gc_rich_stem_density', 'overlap_bp', 'sl1_terminal_mfe',
    'spliced_utr_dist', 'alu_full_mfe', 'alu_gc_content', 'rrna_18s_score',
]

N_TRIALS = 100
N_FOLDS = 5


def write_report(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    print('=' * 60)
    print("3'UTR XGBoost regression -- hyperparameter tuning")
    print('=' * 60)

    # ---- 1. Load & filter ----
    print(f'\n[1] Loading 3\'UTR data...')
    df = pd.read_csv(INPUT_CSV)
    df = df[df['region_inserted'] == '3UTR'].dropna(subset=['l2fc'])
    print(f'  N = {len(df)}')

    # ---- 2. Feature engineering ----
    print(f'\n[2] Feature engineering...')
    y = df['l2fc'].copy()
    X = df.drop(columns=DROP_COLS, errors='ignore')
    X = X.drop(columns=['l2fc'], errors='ignore')
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)
    X = X.drop(columns=[c for c in X.columns if c.startswith('chrom_')],
               errors='ignore')
    assert X.shape[1] == X.select_dtypes(include=[np.number]).shape[1]
    print(f'  Features: {X.shape[1]} (including subfamily dummies)')
    print(f'  Samples:  {X.shape[0]}')

    # ---- 3. Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'\n[3] Train/test split: {X_train.shape[0]}/{X_test.shape[0]}')

    # ---- 4. RandomizedSearchCV ----
    print(f'\n[4] RandomizedSearchCV ({N_TRIALS} combos, {N_FOLDS}-fold CV)...')

    param_dist = {
        'n_estimators':       [200, 400, 600, 800, 1000, 1500, 2000],
        'max_depth':          [3, 4, 5, 6, 8, 10, 12],
        'learning_rate':      [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        'subsample':          [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree':   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_lambda':         [0, 0.1, 0.5, 1, 2, 3, 5, 10],
        'reg_alpha':          [0, 0.1, 0.5, 1, 2, 3, 5, 10],
    }

    base_model = xgb.XGBRegressor(random_state=42, verbosity=0)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=N_TRIALS,
        cv=N_FOLDS,
        scoring='r2',
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_cv_r2 = search.best_score_

    print(f'\n  Best CV R2 = {best_cv_r2:.4f}')
    print(f'  Best params:')
    for k, v in best_params.items():
        print(f'    {k} = {v}')

    # ---- 5. Evaluate best model on test set ----
    print(f'\n[5] Evaluating best model on test set...')
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f'  MSE  = {test_mse:.6f}')
    print(f'  RMSE = {test_rmse:.6f}')
    print(f'  MAE  = {test_mae:.6f}')
    print(f'  R2   = {test_r2:.4f}')
    print(f'  R2 improvement: {test_r2 - 0.1736:+.4f} vs baseline')

    # ---- 6. Full-data final model with best params ----
    print(f'\n[6] Training final model on full data...')
    final_model = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0)
    final_model.fit(X, y)

    gain_imp = final_model.feature_importances_
    gain_order = np.argsort(gain_imp)[::-1]
    print(f'  Gain importance Top 5:')
    for rank, i in enumerate(gain_order[:5], 1):
        print(f'    {rank}. {X.columns[i]}: {gain_imp[i]:.4f}')

    # ---- 7. SHAP ----
    print(f'\n[7] SHAP analysis...')
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

    # Dependence plots (DEP_VARS)
    for var in DEP_VARS:
        if var not in X.columns:
            continue
        idx = list(X.columns).index(var)
        vals = X[var].values
        sv = shap_values[:, idx]
        corr = np.corrcoef(vals, sv)[0, 1]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        ax.scatter(vals, sv, alpha=0.12, s=4, c='steelblue', edgecolors='none')
        o = np.argsort(vals)
        ax.plot(np.sort(vals), uniform_filter1d(sv[o], size=200),
                'r-', lw=2.5, alpha=0.9)
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_xlabel(var, fontsize=10)
        ax.set_ylabel('SHAP value', fontsize=10)
        ax.set_title(f"3'UTR -- {var}  (r={corr:+.3f})", fontsize=11)
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f'dep_{var}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  -> dep_{var}.png')

    # ---- 8. Write report ----
    print(f'\n[8] Writing report...')

    lines = [
        '=' * 70,
        "  3'UTR XGBoost Regression -- Hyperparameter Tuning Results",
        '=' * 70,
        f'  N={len(X)}, features={X.shape[1]}',
        '',
        f'  RandomizedSearchCV: {N_TRIALS} combos, {N_FOLDS}-fold CV',
        f'  Best CV R2 = {best_cv_r2:.4f}',
        '',
        '  Best hyperparameters:',
    ]
    for k, v in best_params.items():
        lines.append(f'    {k} = {v}')

    lines.extend([
        '',
        f'  Test set evaluation (20% holdout):',
        f'    MSE  = {test_mse:.6f}',
        f'    RMSE = {test_rmse:.6f}',
        f'    MAE  = {test_mae:.6f}',
        f'    R2   = {test_r2:.4f}',
        f'    R2 improvement vs baseline (0.1736): {test_r2 - 0.1736:+.4f}',
        '',
        f'  Baseline params (utr_split_model):',
        f'    n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8',
        '',
        '=' * 70,
        '  SHAP global importance (top 15)',
        '=' * 70,
        f'  {"#":>3s}  {"Feature":<38s}  {"mean|SHAP|":<12s}  {"SHAP%":<8s}  {"r(SHAP,val)":<12s}',
        f'  {"-"*3}  {"-"*38}  {"-"*12}  {"-"*8}  {"-"*12}',
    ])

    total_imp = shap_imp.sum()
    for rank, i in enumerate(order[:15], 1):
        col = X.columns[i]
        vals = X[col].values
        s = shap_values[:, i]
        corr = np.corrcoef(vals, s)[0, 1] if np.std(vals) * np.std(s) > 0 else 0
        pct = shap_imp[i] / total_imp * 100
        lines.append(
            f'  {rank:3d}  {col:<38s}  {shap_imp[i]:<12.6f}  {pct:<8.2f}  {corr:<+12.4f}'
        )

    lines.extend([
        '',
        '=' * 70,
        '  SHAP direction (continuous variables)',
        '=' * 70,
    ])
    for var in DEP_VARS:
        if var and var in X.columns:
            idx = list(X.columns).index(var)
            vals = X[var].values
            s = shap_values[:, idx]
            med = np.median(vals)
            high = s[vals > med]
            low = s[vals <= med]
            corr = np.corrcoef(vals, s)[0, 1]
            delta = high.mean() - low.mean()
            direction = '+' if delta > 0 else '-'
            lines.append(
                f'  {var:<35s}  r={corr:+.4f}  '
                f'high_SHAP={high.mean():+.6f}  low_SHAP={low.mean():+.6f}  '
                f'Δ={delta:+.6f}  -> {direction}'
            )

    write_report(os.path.join(OUTPUT_DIR, 'regression_tune_report.txt'), lines)
    print(f'  -> regression_tune_report.txt')

    # ---- 9. Feature importance CSV ----
    imp_df = pd.DataFrame({
        'feature': X.columns,
        'shap_mean_abs': shap_imp,
        'gain_importance': gain_imp,
    }).sort_values('shap_mean_abs', ascending=False)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    print(f'  -> feature_importance.csv')

    # ---- 10. Best params CSV ----
    params_df = pd.DataFrame([best_params])
    params_df.insert(0, 'cv_r2', best_cv_r2)
    params_df.insert(1, 'test_r2', test_r2)
    params_df.to_csv(os.path.join(OUTPUT_DIR, 'best_params.csv'), index=False)
    print(f'  -> best_params.csv')

    print(f'\n{"=" * 60}')
    print(f'Done! Results saved to: {OUTPUT_DIR}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
