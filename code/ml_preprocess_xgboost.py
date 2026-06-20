"""
============================================================================
数据预处理 + XGBoost 回归训练与评估
============================================================================
输入: output/ml/micro_ml_features_final_integrated.csv (52500 rows, 28 cols)
输出: 控制台打印评估指标与 Top-10 特征重要性
============================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
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

# ============================================================================
# Step 1: 加载数据
# ============================================================================
print("=" * 70)
print("数据预处理 + XGBoost 回归")
print("=" * 70)

print("\n[1] 加载数据 ...")
df = pd.read_csv(INPUT_CSV)
print(f"  原始形状: {df.shape}")

# ============================================================================
# Step 2: 分离目标变量，删除无用列
# ============================================================================
print("\n[2] 分离目标变量 y = l2fc，删除标识符列 ...")

y = df['l2fc'].copy()

# 删除标识符和派生列（模型不应看到的）
cols_to_drop = [
    'transcript_id',    # 纯标识符
    'gene_id',          # 纯标识符
    'gene_name',        # 纯标识符
    'alu_id',           # 纯标识符
    'alu_chr_start',    # 坐标参数，非通用特征
    'alu_chr_end',      # 坐标参数，非通用特征
    'layout',           # 由其他列派生的复合编码
]
X = df.drop(columns=cols_to_drop + ['l2fc'], errors='ignore')

print(f"  删除 {len(cols_to_drop)} 列后 X 形状: {X.shape}")

# ============================================================================
# Step 3: 类别特征编码
# ============================================================================
print("\n[3] 类别特征编码 ...")

# 识别所有非数值列（object / category dtype）
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"  发现 {len(cat_cols)} 个类别列: {cat_cols}")

# 对 region_inserted 和 subfamily 进行独热编码（用户指定）
# 同时对 chrom / strand / alu_strand 也做独热编码保证全数值
X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)

print(f"  独热编码后 X 形状: {X.shape}")

# ============================================================================
# Step 4: 缺失值处理
# ============================================================================
print("\n[4] 缺失值检查 ...")

na_counts = X.isna().sum()
na_cols = na_counts[na_counts > 0]
if len(na_cols) == 0:
    print("  X 中无缺失值")
else:
    print(f"  X 中 {len(na_cols)} 列有缺失值:")
    for col, cnt in na_cols.items():
        print(f"    {col}: {cnt} NaN")
    # 数值列用均值填充
    for col in na_cols.index:
        X[col] = X[col].fillna(X[col].mean())
    print("  已用均值填充")

na_y = y.isna().sum()
if na_y > 0:
    print(f"  y 中有 {na_y} 个 NaN，删除对应行")
    valid_mask = y.notna()
    y = y[valid_mask]
    X = X.loc[valid_mask]

print(f"  最终 X 形状: {X.shape}")
print(f"  最终 y 形状: {y.shape}")

# 确认全部为数值型
assert X.shape[1] == X.select_dtypes(include=[np.number]).shape[1], \
    "X 中存在非数值列！"
print("\n  [OK] X 全部为数值型数据")
print(f"  特征列数: {X.shape[1]}")
print(f"  列名: {list(X.columns)}")

# ============================================================================
# Step 5: 数据集划分
# ============================================================================
print("\n" + "=" * 70)
print("XGBoost 回归训练与评估")
print("=" * 70)

print(f"\n[5] 数据集划分 (80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  训练集: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"  测试集: X_test {X_test.shape}, y_test {y_test.shape}")

# ============================================================================
# Step 6: 模型构建与训练
# ============================================================================
print("\n[6] 构建 XGBoost 回归模型 ...")

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbosity=0,
)

print(f"  参数: n_estimators=300, max_depth=6, "
      f"learning_rate=0.05, subsample=0.8")
print("\n  训练中 ...")
model.fit(X_train, y_train)
print("  训练完成")

# ============================================================================
# Step 7: 性能评估
# ============================================================================
print("\n[7] 测试集评估 ...")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"  MSE:  {mse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  R2:   {r2:.6f}")

# ============================================================================
# Step 8: 特征重要性
# ============================================================================
print("\n[8] Top-10 特征重要性 ...")

importance = model.feature_importances_
feat_names = X.columns

# 排序并取前 10
top10_idx = np.argsort(importance)[::-1][:10]
print(f"  {'特征':<40} {'重要性':<10}")
print(f"  {'-'*40} {'-'*10}")
for idx in top10_idx:
    print(f"  {feat_names[idx]:<40} {importance[idx]:.6f}")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
