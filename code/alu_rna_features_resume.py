"""
============================================================================
利用已有的 RNAfold out 文件 + 补跑 sl1 批次，生成最终特征表
============================================================================
"""

import os
import re
import subprocess
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
from twobitreader import TwoBitFile

warnings.filterwarnings('ignore')

# ============================================================================
# 路径
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'ml')

INPUT_CSV = os.path.join(OUTPUT_DIR, 'micro_ml_features_integrated.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'micro_ml_features_final_integrated.csv')
TWB_PATH = os.path.join(DATA_DIR, 'hg38.2bit')

RNAFOLD_PATH = r"C:\Program Files (x86)\ViennaRNA Package\RNAfold.exe"

# 已有 RNAfold 输出文件（全长）
FULL_OUT = r"C:\Users\Dengrong Gao\AppData\Local\Temp\tmpakzzblgy.out"


def reverse_complement(seq: str) -> str:
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
            'a': 't', 't': 'a', 'c': 'g', 'g': 'c',
            'N': 'N', 'n': 'n'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


def parse_rnafold_output(out_path: str, n_expected: int):
    """
    解析 RNAfold 3行/条的输出格式：
    >id
    SEQUENCE
    structure  (-X.XX)
    """
    ids, seqs, structs, mfes = [], [], [], []
    buf = []
    with open(out_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n\r')
            buf.append(line)
            if len(buf) == 3:
                h = buf[0].lstrip('>').strip()
                s = buf[1].strip()
                parts = buf[2].strip().split()
                st = parts[0] if parts else ''
                mfe = np.nan
                if len(parts) >= 2:
                    try:
                        mfe = round(float(parts[-1].strip('()')), 4)
                    except ValueError:
                        pass
                ids.append(h)
                seqs.append(s)
                structs.append(st)
                mfes.append(mfe)
                buf = []
    if buf:
        warnings.warn(f"残留 {len(buf)} 行未解析")

    if len(ids) != n_expected:
        warnings.warn(f"期望 {n_expected} 条，解析到 {len(ids)} 条")

    return ids, seqs, structs, mfes


def main():
    print("=" * 70)
    print("RNA 热力学特征追加脚本（利用已有 RNAfold out 文件）")
    print("=" * 70)

    # ---- Step 1: 读取特征表 ----
    print(f"\n[1] 读取特征表: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  形状: {df.shape}")

    # ---- Step 2: 解析已有的全长 out 文件 ----
    print(f"\n[2] 解析全长 RNAfold out 文件: {FULL_OUT}")
    full_ids, _, _, full_mfes = parse_rnafold_output(FULL_OUT, len(df))
    print(f"  解析到 {len(full_ids)} 条全长 MFE")
    print(f"  MFE 范围: [{min(full_mfes):.2f}, {max(full_mfes):.2f}]")

    # ---- Step 3: 从 hg38.2bit 提取序列 + GC 含量 ----
    print(f"\n[3] 提取序列 + 计算 GC 含量 (hg38.2bit) ...")
    twb = TwoBitFile(TWB_PATH)

    gc_list = []
    sl1_list = []
    row_index_map = []
    skipped = 0

    for idx in tqdm(range(len(df)), desc="提取"):
        row = df.iloc[idx]
        chrom = str(row['chrom'])
        start = int(row['alu_chr_start'])
        end = int(row['alu_chr_end'])
        alu_strand = str(row['alu_strand']).strip()

        try:
            raw = twb[chrom][start:end].upper()
        except Exception:
            skipped += 1
            continue

        if alu_strand == '-':
            seq = reverse_complement(raw)
        else:
            seq = raw

        if len(seq) < 2:
            skipped += 1
            continue

        gc = (seq.upper().count('G') + seq.upper().count('C')) / len(seq)
        gc_list.append(round(gc, 6))

        rna_seq = seq.upper().replace('T', 'U')
        sl1_seq = rna_seq[-80:] if len(rna_seq) >= 80 else rna_seq
        sl1_list.append((str(idx), sl1_seq))

        row_index_map.append(idx)

    n_valid = len(row_index_map)
    print(f"  有效: {n_valid}, 跳过: {skipped}")

    # ---- Step 4: 批量折叠 sl1 ----
    print(f"\n[4] 批量折叠 sl1 (末端 80nt, n={n_valid}) ...")

    sl1_fasta = os.path.join(os.environ.get('TEMP', '/tmp'), 'sl1_input.fasta')
    sl1_out = os.path.join(os.environ.get('TEMP', '/tmp'), 'sl1_output.out')

    # 写 FASTA
    with open(sl1_fasta, 'w', encoding='ascii') as f:
        for sid, sseq in sl1_list:
            f.write(f">{sid}\n{sseq}\n")

    # 执行 RNAfold
    cmd = [RNAFOLD_PATH, '-i', sl1_fasta, '--noPS']
    with open(sl1_out, 'w', encoding='utf-8') as of:
        result = subprocess.run(cmd, stdout=of, stderr=subprocess.DEVNULL,
                                timeout=3600, check=False)

    if result.returncode != 0:
        print(f"  RNAfold 失败 (returncode={result.returncode})")
        return

    # 解析 sl1 结果
    sl1_ids, _, _, sl1_mfes = parse_rnafold_output(sl1_out, n_valid)
    print(f"  解析到 {len(sl1_ids)} 条 sl1 MFE")
    print(f"  MFE 范围: [{min(sl1_mfes):.2f}, {max(sl1_mfes):.2f}]")

    # ---- Step 5: 回填 DataFrame ----
    print(f"\n[5] 追加新列 + 导出 ...")
    df['alu_gc_content'] = np.nan
    df['alu_full_mfe'] = np.nan
    df['sl1_terminal_mfe'] = np.nan

    for i, orig_idx in enumerate(row_index_map):
        df.at[orig_idx, 'alu_gc_content'] = gc_list[i]
        df.at[orig_idx, 'alu_full_mfe'] = full_mfes[i]
        df.at[orig_idx, 'sl1_terminal_mfe'] = sl1_mfes[i]

    print(f"  统计:")
    print(f"    alu_gc_content:    mean={df['alu_gc_content'].mean():.4f}, na={df['alu_gc_content'].isna().sum()}")
    print(f"    alu_full_mfe:      mean={df['alu_full_mfe'].mean():.4f}, na={df['alu_full_mfe'].isna().sum()}")
    print(f"    sl1_terminal_mfe:  mean={df['sl1_terminal_mfe'].mean():.4f}, na={df['sl1_terminal_mfe'].isna().sum()}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  导出: {OUTPUT_CSV}")
    print(f"  形状: {df.shape}")
    print("完成！")


if __name__ == '__main__':
    main()
