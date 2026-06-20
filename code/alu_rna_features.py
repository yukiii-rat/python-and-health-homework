"""
============================================================================
RNA 热力学二级结构特征追加脚本
读取 micro_ml_features_integrated.csv → 提取 Alu 序列 → 计算热力学特征 → 输出最终大表
============================================================================

计算引擎：subprocess 调用本地 RNAfold.exe（批量模式）

计算 3 个特征：
  1. alu_full_mfe          : 全长 Alu 最小自由能 (kcal/mol)
  2. sl1_terminal_mfe      : 末端 80nt 发夹 MFE (kcal/mol)
  3. alu_gc_content        : GC 含量 (比例)

输出：micro_ml_features_final_integrated.csv
"""

import os
import warnings
import subprocess

import pandas as pd
import numpy as np
from tqdm import tqdm
from twobitreader import TwoBitFile

# 导入 RNAfold 批量处理器
from rnafold_batch_processor import RNAfoldBatchProcessor

warnings.filterwarnings('ignore')

# ============================================================================
# 路径配置
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'ml')

INPUT_CSV = os.path.join(OUTPUT_DIR, 'micro_ml_features_integrated.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'micro_ml_features_final_integrated.csv')
TWB_PATH = os.path.join(DATA_DIR, 'hg38.2bit')

# ============================================================================
# RNAfold 路径配置（用户可修改）
# ============================================================================
RNAFOLD_PATH = r"C:\Program Files (x86)\ViennaRNA Package\RNAfold.exe"


def reverse_complement(seq: str) -> str:
    """DNA 反向互补"""
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
            'N': 'N', 'n': 'n'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


def main():
    print("=" * 70)
    print("RNA 热力学特征追加脚本")
    print(f"RNAfold: {RNAFOLD_PATH}")
    print("=" * 70)

    # ---- Step 1: 读取特征表 ----
    print(f"\n[1/5] 读取特征表: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)}")

    # ---- Step 2: 初始化 RNAfold 批量处理器 ----
    print(f"\n[2/5] 初始化 RNAfold 批量处理器 ...")
    try:
        processor = RNAfoldBatchProcessor(RNAFOLD_PATH)
        print(f"  RNAfold 可用: {RNAFOLD_PATH}")
    except FileNotFoundError as e:
        print(f"  [FAIL] {e}")
        print(f"  请检查 RNAFOLD_PATH 配置。")
        return

    # ---- Step 3: 打开 hg38.2bit 并提取所有序列 ----
    print(f"\n[3/5] 提取 Alu 序列 (hg38.2bit) ...")
    twb = TwoBitFile(TWB_PATH)

    seq_records = []       # (idx, seq) 用于 full-length 折叠
    sl1_records = []       # (idx, seq) 用于 sl1 折叠
    gc_contents = []
    row_index_map = []     # 存储有效行的原始索引，用于后续回填
    skipped = 0

    for idx in tqdm(range(len(df)), desc="提取序列"):
        row = df.iloc[idx]
        chrom = str(row['chrom'])
        start = int(row['alu_chr_start'])
        end = int(row['alu_chr_end'])
        alu_strand = str(row['alu_strand']).strip()

        # ---- 从 2bit 提取原始序列 ----
        try:
            raw_seq = twb[chrom][start:end].upper()
        except Exception:
            skipped += 1
            continue

        # ---- 负链反向互补矫正 ----
        if alu_strand == '-':
            seq = reverse_complement(raw_seq)
        else:
            seq = raw_seq

        if len(seq) < 2:
            skipped += 1
            continue

        # ---- GC 含量（立即计算，不需要 RNAfold） ----
        gc = (seq.upper().count('G') + seq.upper().count('C')) / len(seq)
        gc_contents.append(round(gc, 6))

        # 转 RNA（T→U）
        rna_seq = seq.upper().replace('T', 'U')

        # 收集待折叠序列（使用行索引作为 ID）
        record_id = str(idx)
        seq_records.append((record_id, rna_seq))

        # 收集 sl1（末端 80nt）
        sl1_seq = rna_seq[-80:] if len(rna_seq) >= 80 else rna_seq
        sl1_id = f"sl1_{idx}"
        sl1_records.append((sl1_id, sl1_seq))

        row_index_map.append(idx)

    n_valid = len(row_index_map)
    print(f"  有效序列: {n_valid}, 跳过: {skipped}")

    if n_valid == 0:
        print("  [FAIL] 无有效序列，退出")
        return

    # ---- Step 4a: 批量折叠全长 MFE ----
    print(f"\n[4a/5] 批量折叠全长序列 (n={n_valid}) ...")
    try:
        full_df = processor.fold_sequences(seq_records)
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] RNAfold 执行失败: {e}")
        print(f"  stderr: {e.stderr}")
        return

    # 按 idx 排序并提取 MFE
    full_df['idx'] = full_df['id'].astype(int)
    full_df = full_df.sort_values('idx')
    full_mfes = full_df['mfe'].tolist()

    print(f"  [OK] 全长 MFE 计算完成")
    print(f"     MFE 范围: [{min(full_mfes):.2f}, {max(full_mfes):.2f}]")

    # ---- Step 4b: 批量折叠 sl1（末端 80nt） ----
    print(f"\n[4b/5] 批量折叠末端 80nt (n={n_valid}) ...")
    try:
        sl1_df = processor.fold_sequences(sl1_records)
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] RNAfold 执行失败: {e}")
        print(f"  stderr: {e.stderr}")
        return

    sl1_df['idx'] = sl1_df['id'].str.replace('sl1_', '').astype(int)
    sl1_df = sl1_df.sort_values('idx')
    sl1_mfes = sl1_df['mfe'].tolist()

    print(f"  [OK] sl1 MFE 计算完成")
    print(f"     MFE 范围: [{min(sl1_mfes):.2f}, {max(sl1_mfes):.2f}]")

    # ---- Step 5: 追加新列并导出 ----
    print(f"\n[5/5] 追加新列到 DataFrame ...")

    # 先在 df 中创建列，全部初始化为 NaN
    df['alu_gc_content'] = np.nan
    df['alu_full_mfe'] = np.nan
    df['sl1_terminal_mfe'] = np.nan

    # 按 row_index_map 回填有效行的值
    for i, orig_idx in enumerate(row_index_map):
        df.at[orig_idx, 'alu_gc_content'] = gc_contents[i]
        df.at[orig_idx, 'alu_full_mfe'] = full_mfes[i]
        df.at[orig_idx, 'sl1_terminal_mfe'] = sl1_mfes[i]

    print(f"  跳过/失败的序列: {skipped}")
    print(f"  各特征统计:")
    print(f"    alu_gc_content:    mean={df['alu_gc_content'].mean():.4f}, "
          f"na={df['alu_gc_content'].isna().sum()}")
    print(f"    alu_full_mfe:      mean={df['alu_full_mfe'].mean():.4f}, "
          f"na={df['alu_full_mfe'].isna().sum()}")
    print(f"    sl1_terminal_mfe:  mean={df['sl1_terminal_mfe'].mean():.4f}, "
          f"na={df['sl1_terminal_mfe'].isna().sum()}")

    # ---- 导出 ----
    print(f"\n  导出最终大表: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)}")
    print(f"\n{'=' * 70}")
    print("完成！")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
