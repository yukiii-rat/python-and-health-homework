"""
Generate PDF report for UTR subset XGBoost model with region_inserted included.
"""
import os, warnings
import pandas as pd, numpy as np
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import uniform_filter1d

OUTPUT_DIR = r'D:\pycharm\data\health and medicine of basic python coding\python and health homework\output\ml'

# Font
for f in ['Microsoft YaHei', 'SimHei', 'DengXian']:
    try:
        plt.rcParams['font.sans-serif'] = [f, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure().gca().set_title('t')
        plt.close()
        break
    except:
        pass

# 1. Load + filter
df = pd.read_csv(os.path.join(OUTPUT_DIR, 'micro_ml_features_final_integrated.csv'))
df = df[df['region_inserted'] != 'CDS'].dropna(subset=['l2fc'])
n5 = len(df[df['region_inserted'].str.contains("5'")])
n3 = len(df[df['region_inserted'] == '3UTR'])
print(f'UTR subset: {df.shape}  (5UTR={n5}, 3UTR={n3})')

# 2. Feature engineering
y = df['l2fc'].copy()
X = df.drop(columns=['transcript_id','gene_id','gene_name','alu_id',
                      'alu_chr_start','alu_chr_end','layout','l2fc'], errors='ignore')
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)
X = X.drop(columns=[c for c in X.columns if c.startswith('chrom_')])

# 3. Split + train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, random_state=42, verbosity=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE={mse:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}  R2={r2:.4f}')

# 4. SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap_imp = np.abs(shap_values).mean(axis=0)
order = np.argsort(shap_imp)[::-1]

# 5. PDF
def sf(fig, pdf):
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

pdf_path = os.path.join(OUTPUT_DIR, 'xgboost_report_utr.pdf')
with PdfPages(pdf_path) as pdf:
    # ----- Page 1: Cover -----
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111); ax.axis('off')
    info = [
        ('XGBoost Regression Report  (UTR Subset)', 15),
        ('', 6),
        (f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 9),
        ('', 10),
        ('-'*50, 7),
        (f'Samples (UTR):           {len(df)}', 9),
        (f'  5\'UTR:                 {n5}  ({n5/len(df)*100:.1f}%)', 9),
        (f'  3UTR:                  {n3}  ({n3/len(df)*100:.1f}%)', 9),
        (f'Features:                {X.shape[1]}', 9),
        (f'Train / Test:            {X_train.shape[0]} / {X_test.shape[0]}', 9),
        (f'', 6),
        (f'Model: XGBoost Regressor', 9),
        (f'  n_estimators=300, max_depth=6, lr=0.05, subsample=0.8', 8),
        (f'', 6),
        (f'MSE:  {mse:.6f}', 9),
        (f'RMSE: {rmse:.6f}', 9),
        (f'MAE:  {mae:.6f}', 9),
        (f'R2:   {r2:.4f}', 10),
        (f'', 6),
        (f'region_inserted:        included (5UTR / 3UTR)', 9),
        (f'chrom dummies:          excluded', 9),
    ]
    y0 = 0.88
    for text, size in info:
        ax.text(0.1, y0, text, fontsize=size, transform=ax.transAxes,
                fontfamily='monospace', verticalalignment='top')
        y0 -= 0.028 * (size / 8)
    sf(fig, pdf)

    # ----- Page 2: SHAP importance table -----
    fig = plt.figure(figsize=(8.27, 7))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111); ax.axis('off')
    ax.set_title('SHAP Global Importance (Top 15)', fontsize=13, fontweight='bold', pad=10)
    rows = []
    for rank, i in enumerate(order[:15], 1):
        col = X.columns[i]
        s = shap_values[:, i]
        corr = np.corrcoef(X_test[col], s)[0, 1] if np.std(X_test[col])*np.std(s) > 0 else 0
        pct = shap_imp[i] / shap_imp.sum() * 100
        rows.append([rank, col[:38], f'{shap_imp[i]:.6f}', f'{pct:.1f}%', f'{corr:+.3f}'])
    tbl = ax.table(cellText=rows, colLabels=['#', 'Feature', 'mean|SHAP|', '%', 'SHAP corr'],
                   loc='center', cellLoc='center', colWidths=[0.04, 0.44, 0.13, 0.07, 0.10])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.7)
    for j in range(5):
        tbl[0,j].set_facecolor('#4472C4')
        tbl[0,j].set_text_props(color='white', fontweight='bold')
    for i, r in enumerate(rows):
        if 'region_inserted' in r[1]:
            for j in range(5):
                tbl[i+1,j].set_facecolor('#FFD699')
    sf(fig, pdf)

    # ----- Page 3: Beeswarm -----
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    beeswarm_path = os.path.join(OUTPUT_DIR, '_beeswarm_temp.png')
    plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
    plt.close()
    fig = plt.figure(figsize=(8.27, 7))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111); ax.axis('off')
    ax.set_title('SHAP Beeswarm Summary', fontsize=13, fontweight='bold')
    ax.imshow(plt.imread(beeswarm_path), aspect='auto')
    sf(fig, pdf)
    os.unlink(beeswarm_path)

    # ----- Page 4: Dependence plots -----
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 8))
    fig.patch.set_facecolor('white')
    fig.suptitle('SHAP Dependence Plots', fontsize=13, fontweight='bold', y=0.98)
    for ax_v, var in zip(axes.flatten(), ['gc_rich_stem_density', 'overlap_bp',
                                           'sl1_terminal_mfe', 'rrna_18s_score']):
        if var not in X_test.columns:
            ax_v.text(0.5, 0.5, f'{var} not found', ha='center', transform=ax_v.transAxes); continue
        idx = list(X_test.columns).index(var)
        vals = X_test[var].values
        sv = shap_values[:, idx]
        ax_v.scatter(vals, sv, alpha=0.12, s=3, c='steelblue', edgecolors='none')
        o = np.argsort(vals)
        ax_v.plot(np.sort(vals), uniform_filter1d(sv[o], size=200), 'r-', lw=2.5, alpha=0.9)
        ax_v.axhline(0, color='gray', ls='--', lw=0.5)
        corr = np.corrcoef(vals, sv)[0, 1]
        ax_v.set_xlabel(var, fontsize=8); ax_v.set_ylabel('SHAP value', fontsize=8)
        ax_v.set_title(f'{var}  (r={corr:+.3f})', fontsize=10)
        ax_v.tick_params(labelsize=7); ax_v.grid(True, alpha=0.15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    sf(fig, pdf)

    # ----- Page 5: Pred vs Actual + Residuals -----
    fig, axes = plt.subplots(1, 2, figsize=(8.27, 4))
    fig.patch.set_facecolor('white')
    # scatter
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.15, s=3, c='steelblue', edgecolors='none')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual l2fc', fontsize=9); ax.set_ylabel('Predicted l2fc', fontsize=9)
    ax.set_title(f'Actual vs Predicted  (R2={r2:.4f})', fontsize=10, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    # residuals
    ax = axes[1]
    res = y_test - y_pred
    ax.hist(res, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Residual', fontsize=9); ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title(f'Residuals  (mean={res.mean():.4f}, std={res.std():.4f})', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    sf(fig, pdf)

print(f'PDF generated: {pdf_path}')
