"""
============================================================================
3'UTR Alu binary classification -- upregulate vs downregulate TE
============================================================================
Input:  output/ml_right/micro_ml_features_final_integrated.csv
Output: output/ml_right/ml_5UTR_classify/

Strategy:
  - Target: l2fc > 0 -> up (1), l2fc <= 0 -> down (0)
  - 14 core continuous features (no subfamily dummies)
  - XGBoost classifier + strong regularization, 5-fold CV
  - SHAP only if AUC >= 0.65
============================================================================
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, recall_score, precision_score,
                             confusion_matrix)
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'ml_right', 'ml_3UTR_classify')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['axes.unicode_minus'] = False

# ====================================================================
# Core features (14, no subfamily dummies)
# ====================================================================
CORE_FEATURES = [
    'spliced_utr_dist', 'overlap_bp', 'region_relative_pos',
    'alu_full_mfe', 'sl1_terminal_mfe', 'alu_gc_content', 'gc_rich_stem_density',
    'agg_motif_density', 'ugg_motif_density', 'sl1_ugg_count',
    'alu_length', 'rrna_18s_score', 'subfamily_age', 'is_antisense',
]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.02,
    subsample=0.6,
    colsample_bytree=0.4,
    reg_lambda=5,
    scale_pos_weight=1.0,
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    verbosity=0,
)

N_FOLDS = 5
AUC_THRESHOLD = 0.65


def write_report(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    print('=' * 60)
    print("3'UTR Alu binary classification (up/down TE)")
    print('=' * 60)

    # ---- 1. Load data ----
    print(f'\n[1] Loading 3\'UTR subset...')
    df = pd.read_csv(INPUT_CSV)
    df = df[df['region_inserted'] == '3UTR'].copy()
    print(f'  N={len(df)}')

    # ---- 2. Features and labels ----
    print(f'\n[2] Building feature matrix...')
    y = (df['l2fc'] > 0).astype(int)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f'  Up (l2fc>0):  {n_pos} ({n_pos/len(y)*100:.1f}%)')
    print(f'  Down (l2fc<=0): {n_neg} ({n_neg/len(y)*100:.1f}%)')
    print(f'  Ratio: {n_pos/n_neg:.1f}:1')

    X = df[CORE_FEATURES].copy()
    for col in X.columns:
        if X[col].dtype in (np.float64, np.float32, np.int64, np.int32):
            X[col] = X[col].fillna(X[col].median())
    print(f'  Features: {X.shape[1]}')
    print(f'  Names: {list(X.columns)}')

    # ---- 3. 5-fold CV ----
    print(f'\n[3] {N_FOLDS}-fold stratified CV...')
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    cv_aucs, cv_aps, cv_f1s = [], [], []
    cv_recalls, cv_precisions = [], []
    all_y_true, all_y_pred_prob = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        cv_aucs.append(auc)
        cv_aps.append(ap)
        cv_f1s.append(f1)
        cv_recalls.append(recall)
        cv_precisions.append(precision)

        all_y_true.extend(y_test.tolist())
        all_y_pred_prob.extend(y_prob.tolist())

        print(f'  Fold {fold}: AUC={auc:.4f}, AP={ap:.4f}, '
              f'F1={f1:.4f}, Recall={recall:.4f}, Prec={precision:.4f}')

    mean_auc = np.mean(cv_aucs)
    print(f'\n  CV averages:')
    print(f'    AUC      = {mean_auc:.4f} +/- {np.std(cv_aucs):.4f}')
    print(f'    AP       = {np.mean(cv_aps):.4f} +/- {np.std(cv_aps):.4f}')
    print(f'    F1       = {np.mean(cv_f1s):.4f} +/- {np.std(cv_f1s):.4f}')
    print(f'    Recall   = {np.mean(cv_recalls):.4f} +/- {np.std(cv_recalls):.4f}')
    print(f'    Precision= {np.mean(cv_precisions):.4f} +/- {np.std(cv_precisions):.4f}')

    # ---- 4. Full-data final model ----
    print(f'\n[4] Training final model on full data...')
    final_model = xgb.XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y, verbose=False)

    gain_imp = final_model.feature_importances_
    gain_order = np.argsort(gain_imp)[::-1]
    print(f'  Gain importance Top 5:')
    for rank, i in enumerate(gain_order[:5], 1):
        print(f'    {rank}. {X.columns[i]}: {gain_imp[i]:.4f}')

    # ---- 5. SHAP (only if AUC >= threshold) ----
    do_shap = mean_auc >= AUC_THRESHOLD
    print(f'\n[5] SHAP (AUC={mean_auc:.4f}, threshold={AUC_THRESHOLD})...')
    if do_shap:
        print(f'  AUC >= threshold, running SHAP')
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)
        shap_imp = np.abs(shap_values).mean(axis=0)
        order = np.argsort(shap_imp)[::-1]

        # SHAP beeswarm
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, 'shap_beeswarm.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  -> shap_beeswarm.png')

        # SHAP bar
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X, plot_type='bar', show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, 'shap_bar.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  -> shap_bar.png')

        # Dependence plots (top 6)
        for rank, i in enumerate(order[:6], 1):
            var = X.columns[i]
            sv = shap_values[:, i]
            vals = X[var].values
            corr = np.corrcoef(vals, sv)[0, 1]

            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('white')
            ax.scatter(vals, sv, alpha=0.15, s=6, c='steelblue', edgecolors='none')
            o = np.argsort(vals)
            ax.plot(np.sort(vals), uniform_filter1d(sv[o], size=60),
                    'r-', lw=2.5, alpha=0.9)
            ax.axhline(0, color='gray', ls='--', lw=0.5)
            ax.set_xlabel(var, fontsize=10)
            ax.set_ylabel('SHAP (positive -> upregulate TE)', fontsize=10)
            ax.set_title(f"3'UTR -- {var}  (r={corr:+.3f})", fontsize=11)
            ax.grid(True, alpha=0.15)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, f'dep_{var}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  -> dep_{var}.png')
    else:
        print(f'  AUC < threshold, skipping SHAP')

    # ---- 6. Write report ----
    print(f'\n[6] Writing report...')
    cm = confusion_matrix(all_y_true,
                          (np.array(all_y_pred_prob) >= 0.5).astype(int))

    lines = [
        '=' * 70,
        "  3'UTR Alu Binary Classification Results",
        '=' * 70,
        f'  N={len(df)}, features={X.shape[1]}',
        f'  Up (l2fc>0):  {n_pos} ({n_pos/len(y)*100:.1f}%)',
        f'  Down (l2fc<=0): {n_neg} ({n_neg/len(y)*100:.1f}%)',
        '',
        f'  {N_FOLDS}-fold CV results:',
        f'    AUC      = {mean_auc:.4f} +/- {np.std(cv_aucs):.4f}',
        f'    AP       = {np.mean(cv_aps):.4f} +/- {np.std(cv_aps):.4f}',
        f'    F1       = {np.mean(cv_f1s):.4f} +/- {np.std(cv_f1s):.4f}',
        f'    Recall   = {np.mean(cv_recalls):.4f} +/- {np.std(cv_recalls):.4f}',
        f'    Precision= {np.mean(cv_precisions):.4f} +/- {np.std(cv_precisions):.4f}',
        '',
        f'  Confusion matrix (CV aggregate):',
        f'    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}',
        f'    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}',
        '',
    ]

    if do_shap:
        lines.extend([
            f'  SHAP global importance:',
            f'  {"#":>3s}  {"Feature":<28s}  {"mean|SHAP|":<12s}  {"SHAP%":<8s}  {"r(SHAP,val)":<12s}',
            f'  {"-"*3}  {"-"*28}  {"-"*12}  {"-"*8}  {"-"*12}',
        ])
        total_imp = shap_imp.sum()
        for rank, i in enumerate(order, 1):
            col = X.columns[i]
            sv = shap_values[:, i]
            vals = X[col].values
            corr = np.corrcoef(vals, sv)[0, 1] if np.std(vals) * np.std(sv) > 0 else 0
            pct = shap_imp[i] / total_imp * 100
            lines.append(
                f'  {rank:3d}  {col:<28s}  {shap_imp[i]:<12.6f}  {pct:<8.2f}  {corr:<+12.4f}'
            )

        lines.extend([
            '',
            f'  Gain importance:',
            f'  {"#":>3s}  {"Feature":<28s}  {"Gain":<12s}',
            f'  {"-"*3}  {"-"*28}  {"-"*12}',
        ])
        for rank, i in enumerate(gain_order, 1):
            lines.append(
                f'  {rank:3d}  {X.columns[i]:<28s}  {gain_imp[i]:<12.6f}'
            )

        lines.extend([
            '',
            f'  SHAP direction (continuous variables):',
        ])
        for var in CORE_FEATURES:
            if var in X.columns:
                idx = list(X.columns).index(var)
                vals = X[var].values
                sv = shap_values[:, idx]
                med = np.median(vals)
                high = sv[vals > med]
                low = sv[vals <= med]
                corr = np.corrcoef(vals, sv)[0, 1]
                delta = high.mean() - low.mean()
                direction = '+' if delta > 0 else '-'
                lines.append(
                    f'    {var:<28s}  r={corr:+.4f}  '
                    f'high_SHAP={high.mean():+.4f}  low_SHAP={low.mean():+.4f}  '
                    f'delta={delta:+.4f}  -> {direction}'
                )
    else:
        lines.append('  SHAP SKIPPED: AUC below threshold')
        lines.append('')
        lines.append("  3'UTR classification AUC < 0.65 -- no predictive power")
        lines.append('  Features cannot distinguish up/down regulation of TE')
        lines.append('  Recommendation: use gene-level stats (v3.2 approach)')

    write_report(os.path.join(OUTPUT_DIR, 'utr3_classification.txt'), lines)
    print(f'  -> utr3_classification.txt')

    # ---- 7. Feature importance CSV ----
    if do_shap:
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'shap_mean_abs': shap_imp,
            'gain_importance': gain_imp,
        }).sort_values('shap_mean_abs', ascending=False)
        imp_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
        print(f'  -> feature_importance.csv')

    print(f'\n{"=" * 60}')
    if do_shap:
        print(f'Done! Results saved to: {OUTPUT_DIR}')
    else:
        print(f"3'UTR classification AUC={mean_auc:.4f} < {AUC_THRESHOLD}, no predictive power")
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
