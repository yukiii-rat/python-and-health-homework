"""
============================================================================
XGBoost 回归结果 PDF 报告生成器
============================================================================
输入: output/ml/micro_ml_features_final_integrated.csv
输出: output/ml/xgboost_report.pdf
============================================================================
"""

import os
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# ============================================================================
# 路径
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
INPUT_CSV = os.path.join(PROJECT_ROOT, 'output', 'ml',
                         'micro_ml_features_final_integrated.csv')
OUTPUT_PDF = os.path.join(PROJECT_ROOT, 'output', 'ml',
                          'xgboost_report.pdf')

# ============================================================================
# 中文字体设置
# ============================================================================
# 尝试几个常见的中文字体
_ZH_FONTS = [
    'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi',
    'FangSong', 'DengXian', 'Source Han Sans SC',
    'Noto Sans CJK SC',
]
_FONT = None
for f in _ZH_FONTS:
    try:
        plt.rcParams['font.sans-serif'] = [f, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        # 测试该字体能否渲染中文
        fig_t = plt.figure()
        ax_t = fig_t.add_subplot(111)
        ax_t.set_title('测试中文')
        plt.close(fig_t)
        _FONT = f
        break
    except Exception:
        continue

if _FONT:
    plt.rcParams['font.sans-serif'] = [_FONT, 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def save_figure_to_pdf(fig, pdf):
    """将 matplotlib figure 写入 PDF 页面"""
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def create_title_page(summary_data):
    """封面页"""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.axis('off')

    lines = [
        ("", 24),
        ("XGBoost 回归分析报告", 20),
        ("", 12),
        (f"数据文件: micro_ml_features_final_integrated.csv", 10),
        (f"生成日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", 10),
        ("", 16),
        ("=" * 50, 8),
        ("", 8),
    ]

    for key, val in summary_data.items():
        lines.append((f"{key}:  {val}", 10))

    texts = []
    for text, size in lines:
        t = ax.text(0.5, 0.5, text, fontsize=size,
                    ha='center', va='center', transform=ax.transAxes)
        texts.append(t)

    # 用垂直堆叠的方式重新布局
    ax.clear()
    ax.axis('off')
    y0 = 0.92
    dy = 0.035
    for text, size in lines:
        ax.text(0.1, y0, text, fontsize=size,
                transform=ax.transAxes, verticalalignment='top',
                fontfamily='monospace')
        y0 -= dy * (size / 8)

    return fig


def create_preprocessing_page(df_info, X_shape, y_shape, X_cols, cat_info, na_info):
    """数据预处理信息页"""
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    # 上：预处理流程
    ax = axes[0]
    ax.axis('off')
    ax.set_title('数据预处理流程', fontsize=14, fontweight='bold')

    info_lines = [
        f"原始数据: {df_info}",
        f"删除标识符列: transcript_id, gene_id, gene_name, alu_id, alu_chr_start, alu_chr_end, layout",
        f"类别特征独热编码: {cat_info}",
        f"缺失值处理: {na_info}",
        f"",
        f"最终 X 形状: {X_shape}",
        f"最终 y 形状: {y_shape}",
        f"特征总数: {X_cols}",
    ]
    for i, line in enumerate(info_lines):
        ax.text(0.02, 0.88 - i * 0.08, line, fontsize=9,
                transform=ax.transAxes, verticalalignment='top',
                fontfamily='monospace')

    # 下：特征列名（分栏显示）
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('特征列清单', fontsize=14, fontweight='bold')
    # 这里不重复列名，留给后续页面

    return fig


def create_feature_importance_plot(model, feature_names):
    """Top-10 特征重要性柱状图"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:10]

    fig, ax = plt.subplots(figsize=(8.27, 6))
    fig.patch.set_facecolor('white')

    names = [feature_names[i] for i in indices][::-1]
    vals = [importance[i] for i in indices][::-1]

    bars = ax.barh(range(len(names)), vals, color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Top-10 特征重要性', fontsize=14, fontweight='bold')

    # 在柱状条右侧标注数值
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=7)

    # 标注主导特征
    if vals:
        ax.text(0.5, -0.12,
                f'主导特征: {names[-1]} ({vals[-1]*100:.2f}%)',
                transform=ax.transAxes, ha='center', fontsize=9,
                style='italic', color='gray')

    fig.tight_layout()
    return fig


def create_prediction_scatter(y_test, y_pred):
    """真实值 vs 预测值散点图"""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('white')

    ax.scatter(y_test, y_pred, alpha=0.3, s=5, c='steelblue', edgecolors='none')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, alpha=0.7, label='y = x (ideal)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual l2fc', fontsize=11)
    ax.set_ylabel('Predicted l2fc', fontsize=11)
    ax.set_title('真实值 vs 预测值 (测试集)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def create_residuals_plot(y_test, y_pred):
    """残差分布图"""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(8.27, 4))
    fig.patch.set_facecolor('white')

    # 左：残差直方图
    ax = axes[0]
    ax.hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('残差分布', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 标注均值 + 标准差
    ax.text(0.95, 0.95,
            f'Mean: {residuals.mean():.4f}\nStd:  {residuals.std():.4f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 右：残差 vs 预测值
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=5, c='steelblue', edgecolors='none')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Predicted l2fc', fontsize=10)
    ax.set_ylabel('Residual', fontsize=10)
    ax.set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def create_metrics_table(mse, rmse, mae, r2):
    """评估指标表格页"""
    fig, ax = plt.subplots(figsize=(8.27, 3))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    metrics = [
        ('MSE (Mean Squared Error)', f'{mse:.6f}'),
        ('RMSE (Root Mean Squared Error)', f'{rmse:.6f}'),
        ('MAE (Mean Absolute Error)', f'{mae:.6f}'),
        ('R-squared (R²)', f'{r2:.6f}'),
    ]

    col_labels = ['Metric', 'Value']
    table_data = [[m, v] for m, v in metrics]
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 着色
    for j in range(2):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(4):
        table[i + 1, 0].set_facecolor('#D6E4F0')
        table[i + 1, 1].set_facecolor('#E8F0FE')

    ax.set_title('模型评估指标 (测试集)', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("XGBoost 回归报告 PDF 生成")
    print("=" * 70)

    # ---- Step 1: 加载数据 ----
    print("\n[1/5] 加载数据 ...")
    df = pd.read_csv(INPUT_CSV)
    original_shape = df.shape
    print(f"  原始形状: {original_shape}")

    # ---- Step 2: 预处理 ----
    print("[2/5] 数据预处理 ...")

    # 分离 y
    y = df['l2fc'].copy()
    X = df.drop(columns=[
        'transcript_id', 'gene_id', 'gene_name', 'alu_id',
        'alu_chr_start', 'alu_chr_end', 'layout', 'l2fc'
    ], errors='ignore')

    # 独热编码
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_info = f"{len(cat_cols)} 列: {', '.join(cat_cols)}"
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)

    # 缺失值
    na_y = y.isna().sum()
    na_info = f"X 无缺失, y 有 {na_y} 个 NaN → 删除对应行"
    valid = y.notna()
    y = y[valid]
    X = X.loc[valid]

    # 数值检查
    assert X.shape[1] == X.select_dtypes(include=[np.number]).shape[1]

    X_shape = X.shape
    y_shape = y.shape
    n_features = X.shape[1]

    print(f"  X: {X_shape}, y: {y_shape}, 特征数: {n_features}")

    # ---- Step 3: 训练集/测试集划分 ----
    print("[3/5] 划分训练集/测试集 (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # ---- Step 4: 训练 XGBoost ----
    print("[4/5] 训练 XGBoost 模型 ...")
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)
    print("  训练完成")

    # 预测与评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")

    # ---- Step 5: 生成 PDF ----
    print("[5/5] 生成 PDF 报告 ...")

    with PdfPages(OUTPUT_PDF) as pdf:
        # Page 1: 封面
        print("  页面 1/5: 封面")
        summary = {
            '原始数据行数': str(original_shape[0]),
            '原始特征数': str(original_shape[1]),
            '预处理后特征数': str(n_features),
            '训练集样本数': str(X_train.shape[0]),
            '测试集样本数': str(X_test.shape[0]),
            '模型': 'XGBoost Regressor',
            'n_estimators': '300',
            'max_depth': '6',
            'learning_rate': '0.05',
            'subsample': '0.8',
            'MSE': f'{mse:.6f}',
            'RMSE': f'{rmse:.6f}',
            'MAE': f'{mae:.6f}',
            'R-squared': f'{r2:.6f}',
        }
        fig = create_title_page(summary)
        save_figure_to_pdf(fig, pdf)

        # Page 2: 预处理信息
        print("  页面 2/5: 数据预处理信息")
        fig = create_preprocessing_page(
            df_info=str(original_shape),
            X_shape=str(X_shape),
            y_shape=str(y_shape),
            X_cols=str(n_features),
            cat_info=cat_info,
            na_info=na_info,
        )
        save_figure_to_pdf(fig, pdf)

        # Page 3: 评估指标
        print("  页面 3/5: 评估指标")
        fig = create_metrics_table(mse, rmse, mae, r2)
        save_figure_to_pdf(fig, pdf)

        # Page 4: 特征重要性
        print("  页面 4/5: 特征重要性")
        fig = create_feature_importance_plot(model, X.columns)
        save_figure_to_pdf(fig, pdf)

        # Page 5: 预测值 vs 真实值 + 残差
        print("  页面 5/5: 预测值与残差分析")
        fig1 = create_prediction_scatter(y_test, y_pred)
        save_figure_to_pdf(fig1, pdf)
        fig2 = create_residuals_plot(y_test, y_pred)
        save_figure_to_pdf(fig2, pdf)

    print(f"\n  PDF 报告已生成: {OUTPUT_PDF}")
    print("=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
