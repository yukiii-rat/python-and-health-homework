# Alu 元件在 mRNA 不同区域对翻译效率的影响

**The Impact of Alu Element Localization on Translational Efficiency: A Machine Learning and SHAP-Based Analysis**

---

## 概述 / Overview

SINE/Alu 元件是人类基因组中丰度最高的转座元件之一，近年来的研究发现其能够通过 SINEUP 机制调控 mRNA 翻译效率。本研究系统评估了 Alu 元件在 mRNA 不同区域（5'UTR、CDS、3'UTR）对翻译效率（Translational Efficiency, TE）的影响及其关键决定因素。

**核心发现：**
- Alu 对翻译效率的影响呈 **位置依赖性**：CDS 插入 → 翻译抑制，UTR 插入 → 翻译促进
- 3'UTR 内 Alu 的精细定位和二级结构稳定性是 TE 的核心预测因子
- XGBoost 模型 + SHAP 可解释性分析定量揭示了调控机制

---

## 项目结构 / Project Structure

```
├── code/                          # Python 分析脚本
│   ├── Alu_mRNA.py               #   Alu 元件在 mRNA 上的定位分析
│   ├── build_feature_matrix.py     #   特征矩阵构建
│   ├── alu_te_within_gene_v3.py   #   配对统计分析 v3
│   ├── alu_rna_features.py        #   RNA 二级结构特征提取
│   ├── rnafold_batch_processor.py #   RNAfold 批量折叠
│   ├── ml_pipeline_human.py       #   人 ML 主流程
│   └── ... (共 35 个脚本)
├── output/
│   ├── v3_2/                      # 配对统计分析结果
│   ├── v4/                        # 布局指纹与剂量效应分析
│   ├── ml_full_region/            # 全区域 XGBoost 模型结果
│   └── ml_3UTR_regression_tune/   # 3'UTR 亚组优化模型结果
├── docs/                          # GitHub Pages 网站源文件
├── requirements.txt               # Python 依赖
└── README.md
```

---

## 方法 / Methods

### 数据来源
| 数据 | 来源 |
|------|------|
| 转录组注释 | GENCODE v49 (human) |
| Alu 坐标 | RepeatMasker (hg38) |
| 翻译效率 | TEDD Database #00137 (HEK293T) |
| 基因组序列 | hg38.2bit |

### 分析流程

1. **特征提取**：位置特征、序列基序、GC 含量、RNA 二级结构（MFE）
2. **配对统计检验**：Wilcoxon 符号秩检验，基因内 Alu+ vs Alu- 配对比较
3. **布局指纹分析**：`u{N_5'UTR}_c{N_CDS}_t{N_3'UTR}` 编码，剂量效应，协同效应
4. **机器学习**：XGBoost 回归，5 折交叉验证，RandomizedSearchCV 超参数优化
5. **可解释性**：SHAP TreeExplainer，特征重要性 + 依赖图

---

## 关键结果 / Key Results

### 1. 配对统计分析
| 区域 | 配对 | l2fc | p-value | 方向 |
|------|------|------|---------|------|
| All | 3,362 | +0.162 | 4.2e-67 | 促进 |
| CDS | 512 | **-0.295** | 1.0e-24 | 抑制 |
| 5'UTR | 703 | **+0.244** | 7.5e-40 | 促进 |
| 3'UTR | 2,576 | **+0.200** | 2.1e-76 | 促进 |

### 2. 机器学习性能
- **全区域模型** (N=11,704): R² = 0.252, RMSE = 0.826
- **3'UTR 亚组模型** (N=9,442): R² = 0.268, RMSE = 0.813

### 3. SHAP 特征重要性 (3'UTR 模型)
1. `spliced_utr_dist` — 12.7% — Alu 到 CDS 边界的距离
2. `region_relative_pos` — 10.7% — 在 3'UTR 内的相对位置
3. `alu_full_mfe` — 9.0% — Alu 全长 RNA 折叠自由能

---

## 引用 / Citation

```bibtex
@misc{gao2025alu,
  author = {Gao, Dengrong},
  title = {基于机器学习与SHAP探究Alu元件在mRNA不同区域对翻译效率的影响},
  year = {2025},
  school = {Peking University}
}
```

---

## 依赖 / Dependencies

```
numpy, pandas, scipy, scikit-learn, xgboost, shap, matplotlib, seaborn
```

详见 `requirements.txt`。
