"""
============================================================================
UTR 双模型 Split-Model 策略
5'UTR 和 3'UTR 分别训练独立的 XGBoost 模型 + SHAP 解释
============================================================================
输入: output/ml/micro_ml_features_final_integrated.csv
输出: output/ml/ml_split/  (结果文本 + SHAP 图)
============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ============================================================================
# 路径
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
INPUT_CSV = os.path.join(PROJECT_ROOT, 'output', 'ml',
                         'micro_ml_features_final_integrated.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'ml', 'ml_split')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 字体
# ============================================================================
_ZH_FONTS = ['Microsoft YaHei', 'SimHei', 'DengXian', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = _ZH_FONTS
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 共享配置
# ============================================================================
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbosity=0,
)

DROP_COLS = [
    'transcript_id', 'gene_id', 'gene_name', 'alu_id',
    'alu_unique_id', 'alu_chr_start', 'alu_chr_end', 'alu_strand',
    'strand', 'chrom', 'log2_dist_to_aug', 'log2_dist_to_stop',
]

# 核心理化特征（用于 SHAP 方向检验）
DEP_VARS = [
    'gc_rich_stem_density', 'overlap_bp', 'sl1_terminal_mfe',
    'spliced_utr_dist', 'alu_full_mfe', 'alu_gc_content', 'rrna_18s_score',
]


def write_report(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def train_and_explain(X, y, label):
    """训练单个区域的 XGBoost 模型 + SHAP 解释"""
    print(f'\n{"="*60}')
    print(f'{label} 模型')
    print(f'{"="*60}')
    print(f'样本量: {len(X)}')
    print(f'特征数: {X.shape[1]}')

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}')

    # 训练
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'\n评估指标:')
    print(f'  MSE  = {mse:.6f}')
    print(f'  RMSE = {rmse:.6f}')
    print(f'  MAE  = {mae:.6f}')
    print(f'  R2   = {r2:.4f}')

    # SHAP
    print(f'\n计算 SHAP ...')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(shap_imp)[::-1]

    # Gain 重要性 Top 15
    gain_imp = model.feature_importances_
    gain_order = np.argsort(gain_imp)[::-1]

    # 构建结果文本
    result_lines = [
        f'{"="*70}',
        f'  {label} 模型结果',
        f'{"="*70}',
        f'  样本量: {len(X)}',
        f'  特征数: {X.shape[1]}',
        f'  训练集: {X_train.shape[0]}',
        f'  测试集: {X_test.shape[0]}',
        '',
        f'  MSE  = {mse:.6f}',
        f'  RMSE = {rmse:.6f}',
        f'  MAE  = {mae:.6f}',
        f'  R2   = {r2:.4f}',
        '',
        f'{"=":-^70}',
        f'  SHAP 全局重要性 (Top 15)',
        f'{"=":-^70}',
        f'  {"#":>3s}  {"特征":<38s}  {"mean|SHAP|":<12s} {"SHAP%":<8s}  {"r(SHAP,val)":<12s}',
        f'  {"-"*3}  {"-"*38}  {"-"*12} {"-"*8}  {"-"*12}',
    ]

    total_imp = shap_imp.sum()
    for rank, i in enumerate(order[:15], 1):
        col = X.columns[i]
        vals = X_test[col].values
        s = shap_values[:, i]
        corr = np.corrcoef(vals, s)[0, 1] if np.std(vals) * np.std(s) > 0 else 0
        pct = shap_imp[i] / total_imp * 100
        result_lines.append(
            f'  {rank:3d}  {col:<38s}  {shap_imp[i]:<12.6f} {pct:<8.2f} {corr:<+12.4f}'
        )

    result_lines.extend([
        '',
        f'{"=":-^70}',
        f'  Gain 重要性 Top 15 (仅供对比, 不作主要解读)',
        f'{"=":-^70}',
    ])
    for rank, i in enumerate(gain_order[:15], 1):
        result_lines.append(
            f'  {rank:3d}  {X.columns[i]:<38s}  {gain_imp[i]:<12.6f}'
        )

    result_lines.extend([
        '',
        f'{"=":-^70}',
        f'  关键连续变量 SHAP 方向检验',
        f'{"=":-^70}',
    ])

    for var in DEP_VARS:
        if var and var in X_test.columns:
            idx = list(X.columns).index(var)
            vals = X_test[var].values
            s = shap_values[:, idx]
            med = np.median(vals)
            high = s[vals > med]
            low = s[vals <= med]
            corr = np.corrcoef(vals, s)[0, 1]
            delta = high.mean() - low.mean()
            direction = '+' if delta > 0 else '-'
            result_lines.append(
                f'  {var:<35s}  r={corr:+.4f}  '
                f'高值SHAP={high.mean():+.6f} 低值SHAP={low.mean():+.6f}  '
                f'Δ={delta:+.6f}  → {direction}'
            )

    write_report(os.path.join(OUTPUT_DIR, f'{label}_results.txt'), result_lines)

    # ---- 画图 ----
    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(
        os.path.join(OUTPUT_DIR, f'{label}_shap_beeswarm.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print(f'  -> {label}_shap_beeswarm.png')

    # Dependence plots
    for var in DEP_VARS:
        if var not in X_test.columns:
            continue
        idx = list(X.columns).index(var)
        vals = X_test[var].values
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
        ax.set_title(f'{label} — {var}  (r={corr:+.3f})', fontsize=11)
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(
            os.path.join(OUTPUT_DIR, f'{label}_dep_{var}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f'  -> {label}_dep_{var}.png')

    # ---- SHAP 交互作用分析 ----
    print(f'\n计算 SHAP interaction values ...')
    interaction_values = explainer.shap_interaction_values(X_test)
    n_feat = X.shape[1]

    # 计算每对特征的平均绝对交互效应（上三角）
    pair_interactions = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            mean_abs_inter = np.abs(interaction_values[:, i, j]).mean()
            pair_interactions.append((mean_abs_inter, i, j))

    pair_interactions.sort(reverse=True)
    top5 = pair_interactions[:5]

    result_lines.extend([
        '',
        f'{"=":-^70}',
        f'  SHAP 交互作用分析 — Top-5 特征对',
        f'{"=":-^70}',
        f'  {"#":>3s}  {"特征 A":<30s}  {"特征 B":<30s}  {"mean|交互|":<12s}',
        f'  {"-"*3}  {"-"*30}  {"-"*30}  {"-"*12}',
    ])
    for rank, (val, i, j) in enumerate(top5, 1):
        result_lines.append(
            f'  {rank:3d}  {X.columns[i]:<30s}  {X.columns[j]:<30s}  {val:<12.6f}'
        )
    write_report(os.path.join(OUTPUT_DIR, f'{label}_results.txt'), result_lines)

    # 交互依赖图：对 top-3 交互对 + 距离特征 × 自动交互
    print(f'  绘制交互依赖图 ...')

    # 收集要画的特征列表：DEP_VARS 中每个特征
    plot_features = [var for var in DEP_VARS if var in X_test.columns]

    for var in plot_features:
        if var not in X_test.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        shap.dependence_plot(
            var, shap_values, X_test,
            interaction_index='auto',
            ax=ax, show=False,
        )
        ax.set_title(f'{label} — {var} (auto interaction)', fontsize=10)
        fig.tight_layout()
        fig.savefig(
            os.path.join(OUTPUT_DIR, f'{label}_interact_{var}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f'    -> {label}_interact_{var}.png')

    return model, X.columns.tolist(), shap_imp, order


def main():
    print('=' * 60)
    print('UTR 双模型 Split-Model 策略')
    print('=' * 60)

    # ---- 1. 加载数据 ----
    print(f'\n[1] 加载数据 ...')
    df = pd.read_csv(INPUT_CSV)
    df = df[df['region_inserted'] != 'CDS'].dropna(subset=['l2fc'])
    print(f'  UTR 子集: {df.shape}')

    n5 = len(df[df["region_inserted"].str.contains("5'")])
    n3 = len(df[df['region_inserted'] == '3UTR'])
    print(f'    5\'UTR: {n5}')
    print(f'    3UTR:  {n3}')

    # ---- 2. 特征工程 ----
    print(f'\n[2] 特征工程 ...')

    def prepare_data(sub_df):
        y = sub_df['l2fc'].copy()
        X = sub_df.drop(columns=DROP_COLS, errors='ignore')
        # 排除目标变量本身
        X = X.drop(columns=['l2fc'], errors='ignore')
        # 独热编码
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)
        # 删除 chrom
        X = X.drop(columns=[c for c in X.columns if c.startswith('chrom_')], errors='ignore')
        assert X.shape[1] == X.select_dtypes(include=[np.number]).shape[1]
        return X, y

    # 5'UTR 子集
    sub5 = df[df["region_inserted"].str.contains("5'")]
    X5, y5 = prepare_data(sub5)
    print(f'  5\'UTR X: {X5.shape}')

    # 3'UTR 子集
    sub3 = df[df['region_inserted'] == '3UTR']
    X3, y3 = prepare_data(sub3)
    print(f'  3UTR X:  {X3.shape}')

    # ---- 3. 分别训练 + SHAP ----
    print(f'\n[3] 训练双模型 ...')
    model5, cols5, imp5, ord5 = train_and_explain(
        X5, y5, 'utr5'
    )
    model3, cols3, imp3, ord3 = train_and_explain(
        X3, y3, 'utr3'
    )

    # ---- 4. 对比表 ----
    print(f'\n[4] 生成对比表 ...')

    # 构建共享特征在两个模型中的 SHAP 对比
    common_features = set(cols5) & set(cols3)
    comp_lines = [
        f'{"="*70}',
        f'  5\'UTR vs 3UTR: 共享特征的 SHAP 重要性对比',
        f'{"="*70}',
        f'',
        f'  {"特征":<38s}  {"5UTR SHAP":<12s}  {"3UTR SHAP":<12s}  {"5UTR排名":<10s}  {"3UTR排名":<10s}',
        f'  {"-"*38}  {"-"*12}  {"-"*12}  {"-"*10}  {"-"*10}',
    ]

    # 按 5'UTR 重要性排序
    feat5_imp = {cols5[i]: imp5[i] for i in range(len(cols5))}
    feat5_rank = {cols5[i]: r + 1 for r, i in enumerate(ord5)}
    feat3_imp = {cols3[i]: imp3[i] for i in range(len(cols3))}
    feat3_rank = {cols3[i]: r + 1 for r, i in enumerate(ord3)}

    for feat in sorted(common_features, key=lambda f: feat5_imp.get(f, 0), reverse=True):
        s5 = feat5_imp.get(feat, 0)
        s3 = feat3_imp.get(feat, 0)
        r5 = feat5_rank.get(feat, '-')
        r3 = feat3_rank.get(feat, '-')
        rank_5 = f'{r5:>3d}' if isinstance(r5, int) else r5
        rank_3 = f'{r3:>3d}' if isinstance(r3, int) else r3
        comp_lines.append(f'  {feat:<38s}  {s5:<12.6f}  {s3:<12.6f}  {rank_5:<10s}  {rank_3:<10s}')

    write_report(os.path.join(OUTPUT_DIR, 'comparison_table.txt'), comp_lines)
    print(f'  -> comparison_table.txt')

    print(f'\n{"="*60}')
    print(f'完成！所有结果已保存至: {OUTPUT_DIR}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
