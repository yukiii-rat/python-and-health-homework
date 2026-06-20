"""
为 micro_ml_features_final_integrated.csv 追加 spliced_utr_dist 列。

从 GTF 解析 UTR 外显子坐标，计算 Alu 到 CDS 边界在剪接后 mRNA 上的
外显子水平距离（而非基因组距离，后者包含大量内含子）。

坐标系统：
  GTF: 1-based closed (UTR, CDS 特征)
  BED: 0-based half-open (alu_chr_start/alu_chr_end)

  BED alu_end = 0-based exclusive = 1-based last position（与 GTF 1-based 对齐）
  BED alu_start = 0-based inclusive → 1-based: alu_start + 1

原理：
  GTF 中的 UTR 特征是外显子级的（不含内含子）。
  剪接后，各 UTR 外显子在 mRNA 上首尾相接，无需添加 gap_to_cds。
"""

import os, re, warnings, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# 路径
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ML_DIR = os.path.join(PROJECT_ROOT, 'output', 'ml')

GTF_PATH = os.path.join(DATA_DIR, 'gencode.v49.primary_assembly.annotation.gtf')
CSV_PATH = os.path.join(ML_DIR, 'micro_ml_features_final_integrated.csv')

# ============================================================================
# 1. 解析 GTF
# ============================================================================
print('=' * 60)
print('Step 1: 解析 GTF — CDS 坐标 + UTR 外显子')
print('=' * 60)

GTF_ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')

tx_info = {}                        # tx_id → {chrom, strand}
tx_cds = defaultdict(list)          # tx_id → [(start, end)]
tx_utr = defaultdict(list)          # tx_id → [(start, end, chrom, strand)]

line_count = 0
with open(GTF_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line_count += 1
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        feature = parts[2]
        if feature not in ('transcript', 'CDS', 'UTR'):
            continue
        start, end = int(parts[3]), int(parts[4])
        chrom = parts[0]
        strand = parts[6]
        attr_str = parts[8]

        tx_id = ''
        for m in GTF_ATTR_RE.finditer(attr_str):
            if m.group(1) == 'transcript_id':
                tx_id = m.group(2).split('.')[0]
                break
        if not tx_id:
            continue

        if feature == 'transcript' and tx_id not in tx_info:
            tx_info[tx_id] = {'chrom': chrom, 'strand': strand}
        elif feature == 'CDS':
            tx_cds[tx_id].append((start, end))
        elif feature == 'UTR':
            tx_utr[tx_id].append((start, end, chrom, strand))

print(f'  GTF 行数: {line_count / 1e6:.1f}M')
print(f'  总转录本: {len(tx_info):,}')
print(f'  有 CDS:   {len(tx_cds):,}')
print(f'  有 UTR:   {len(tx_utr):,}')


def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_iv[0])]
    for s, e in sorted_iv[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# 为每个编码转录本构建 CDS 坐标 + UTR 外显子列表
tx_regions = {}

for tx_id in tqdm(set(tx_cds.keys()) & set(tx_info.keys()), desc='  处理转录本'):
    info = tx_info[tx_id]
    chrom = info['chrom']
    strand = info['strand']

    cds_merged = merge_intervals(tx_cds[tx_id])
    if not cds_merged:
        continue
    cds_min = min(s for s, e in cds_merged)
    cds_max = max(e for s, e in cds_merged)

    utr5_exons, utr3_exons = [], []
    for s, e, c, st in tx_utr[tx_id]:
        if c != chrom or st != strand:
            continue
        if strand == '+':
            if e <= cds_min:
                utr5_exons.append((s, e))
            elif s >= cds_max:
                utr3_exons.append((s, e))
        else:
            if s >= cds_max:
                utr5_exons.append((s, e))
            elif e <= cds_min:
                utr3_exons.append((s, e))

    tx_regions[tx_id] = {
        'cds_min': cds_min,
        'cds_max': cds_max,
        'utr5_exons': sorted(utr5_exons, key=lambda x: x[0]),
        'utr3_exons': sorted(utr3_exons, key=lambda x: x[0]),
        'strand': strand,
        'chrom': chrom,
    }

print(f'  成功提取的编码转录本: {len(tx_regions):,}')


def calc_spliced_dist(tx_id, region, alu_start, alu_end):
    """
    计算 Alu 到 CDS 边界在剪接后 mRNA 上的外显子水平距离。

    四种情况：
      5'UTR (+): 外显子升序，向 CDS(右侧)=正向, remaining=e-alu_edge
      5'UTR (-): 外显子降序，向 CDS(左侧)=正向, remaining=alu_edge-s
      3UTR  (+): 外显子升序，向 CDS(左侧)=反向, remaining=alu_edge-s
      3UTR  (-): 外显子降序，向 CDS(右侧)=反向, remaining=e-alu_edge
    """
    reg = tx_regions.get(tx_id)
    if reg is None:
        return np.nan

    strand = reg['strand']

    # Alu 的 CDS 侧边缘（在剪接后 mRNA 上靠 CDS 的那一侧）
    if region == "5'UTR":
        alu_edge = alu_end if strand == '+' else (alu_start + 1)
        exons = reg['utr5_exons']
        is_rev = (strand == '-')
        forward = True          # 向 CDS = mRNA 正向
    elif region == '3UTR':
        alu_edge = (alu_start + 1) if strand == '+' else alu_end
        exons = reg['utr3_exons']
        is_rev = (strand == '-')
        forward = False         # 向 CDS = mRNA 反向
    else:
        return np.nan

    if not exons:
        return np.nan

    # 按 mRNA 5'→3' 排序
    exons_sorted = sorted(exons, key=lambda x: x[0], reverse=is_rev)

    # 找到 Alu 边缘落在哪个外显子
    target = -1
    for idx, (s, e) in enumerate(exons_sorted):
        if s <= alu_edge <= e:
            target = idx
            break

    if target == -1:
        # 不在任何外显子内 → 紧邻 CDS, 距离为 0
        return 0

    s_ex, e_ex = exons_sorted[target]

    if forward:
        # 5'UTR: 向 CDS = 沿 mRNA 方向前进
        remain = (e_ex - alu_edge) if not is_rev else (alu_edge - s_ex)
        remain = max(0, remain)
        downstream = sum(e - s for s, e in exons_sorted[target + 1:])
        return remain + downstream
    else:
        # 3UTR: 向 CDS = 沿 mRNA 方向后退
        remain = (alu_edge - s_ex) if not is_rev else (e_ex - alu_edge)
        remain = max(0, remain)
        upstream = sum(e - s for s, e in exons_sorted[:target])
        return remain + upstream


# ============================================================================
# 2. 读取 CSV + 合并 + 计算
# ============================================================================
print(f'\n{"=" * 60}')
print('Step 2: 读取特征表')
print('=' * 60)

df = pd.read_csv(CSV_PATH)
print(f'  原始行数: {len(df):,}')

# 合并 CDS 坐标（仅用于验证）
cds_df = pd.DataFrame([
    {'transcript_id': tx_id, 'cds_min': v['cds_min'], 'cds_max': v['cds_max']}
    for tx_id, v in tx_regions.items()
])
df = df.merge(cds_df, on='transcript_id', how='left')
missing = df['cds_min'].isna().sum()
print(f'  缺失 CDS 坐标的行数: {missing} ({missing / len(df) * 100:.2f}%)')

# ============================================================================
# 3. 计算剪接距离
# ============================================================================
print(f'\n{"=" * 60}')
print('Step 3: 计算 spliced_utr_dist')
print('=' * 60)

t0 = time.time()
dists = []
for _, row in tqdm(df.iterrows(), total=len(df), desc='  计算剪接距离'):
    region = row['region_inserted']
    if region not in ("5'UTR", '3UTR'):
        dists.append(np.nan)
        continue
    dists.append(calc_spliced_dist(
        row['transcript_id'], region,
        row['alu_chr_start'], row['alu_chr_end']
    ))

df['spliced_utr_dist'] = dists
print(f'  耗时: {time.time() - t0:.0f}s')

n_valid = df['spliced_utr_dist'].notna().sum()
print(f'\n  spliced_utr_dist 有效值: {n_valid:,} / {len(df):,}')

print(f'\n  各区域分布:')
for region in ["5'UTR", 'CDS', '3UTR']:
    sub = df[(df['region_inserted'] == region) & df['spliced_utr_dist'].notna()]['spliced_utr_dist']
    if len(sub) == 0:
        print(f'    {region}: (无有效值)')
        continue
    print(f'    {region}: n={len(sub):,}, min={sub.min():.0f}, '
          f'median={sub.median():.0f}, mean={sub.mean():.0f}, max={sub.max():.0f}')

# ============================================================================
# 4. 验证相关性
# ============================================================================
print(f'\n{"=" * 60}')
print('Step 4: 验证 spliced_utr_dist vs overlap_bp')
print('=' * 60)

for region in ["5'UTR", '3UTR']:
    sub = df[(df['region_inserted'] == region) & df['spliced_utr_dist'].notna()]
    if len(sub) < 10:
        continue
    r_new = sub['spliced_utr_dist'].corr(sub['overlap_bp'])
    r_old_aug = sub['log2_dist_to_aug'].corr(sub['overlap_bp'])
    r_old_stop = sub['log2_dist_to_stop'].corr(sub['overlap_bp'])
    print(f'  {region}: n={len(sub):,}')
    print(f'    spliced_utr_dist  vs overlap_bp: r={r_new:.4f}')
    print(f'    log2_dist_to_aug  vs overlap_bp: r={r_old_aug:.4f} (旧)')
    print(f'    log2_dist_to_stop vs overlap_bp: r={r_old_stop:.4f} (旧)')

# ============================================================================
# 5. 保存
# ============================================================================
print(f'\n{"=" * 60}')
print('Step 5: 保存')
print('=' * 60)

for col in ['cds_min', 'cds_max']:
    if col in df.columns:
        df = df.drop(columns=[col])
if 'utr_local_dist' in df.columns:
    df = df.drop(columns=['utr_local_dist'])

df.to_csv(CSV_PATH, index=False)
print(f'  已保存至: {CSV_PATH}')
print(f'  列数: {len(df.columns)}')
print(f'  新增列: spliced_utr_dist')
print(f'\n完成！')
