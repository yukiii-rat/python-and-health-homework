"""
============================================================================
RNAfold 批量处理器 — subprocess 调用本地 RNAfold.exe
============================================================================

功能：
  1. 将多条 RNA 序列写入临时 FASTA 文件
  2. 单次调用 RNAfold.exe 批量折叠
  3. 解析输出文件提取结构 + MFE
  4. 返回 pandas.DataFrame

用法：
  from rnafold_batch_processor import RNAfoldBatchProcessor

  processor = RNAfoldBatchProcessor()
  df = processor.fold_sequences([("id1", "GGGAAACCC"), ("id2", "CCCUUUGGG")])
"""

import os
import re
import subprocess
import tempfile
import warnings
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

# ============================================================================
# 全局配置：用户需根据本地安装路径修改
# ============================================================================
RNAFOLD_PATH: str = r"C:\Program Files (x86)\ViennaRNA Package\RNAfold.exe"


# ============================================================================
# RNAfold 批量处理器
# ============================================================================
class RNAfoldBatchProcessor:
    """
    RNAfold.exe 批量子进程封装

    将多条序列写入临时 FASTA，单次调用 RNAfold，
    解析结果后清理临时文件。
    """

    # 编译正则：匹配 " -XX.XX kcal/mol" 形式的 MFE
    _MFE_RE = re.compile(r'\s+([-+]?\d+\.?\d*)\s*kcal/mol')

    def __init__(self, rnafold_path: str = RNAFOLD_PATH):
        """
        参数
        ----
        rnafold_path : str
            RNAfold.exe 的完整路径
        """
        self._rnafold_path = rnafold_path

        # 启动时验证可执行文件存在
        if not os.path.isfile(self._rnafold_path):
            raise FileNotFoundError(
                f"RNAfold 可执行文件未找到: {self._rnafold_path}\n"
                f"请检查 RNAFOLD_PATH 配置是否正确。"
            )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def fold_sequences(
        self,
        sequences: List[Tuple[str, str]],
        temp_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        批量折叠 RNA 序列

        参数
        ----
        sequences : list of (id, seq)
            每条元组为 (序列标识符, RNA 序列字符串)
        temp_dir : str or None
            临时文件存放目录，默认使用系统临时目录

        返回
        ----
        pd.DataFrame
            列: ['id', 'sequence', 'structure', 'mfe']
        """
        if not sequences:
            return pd.DataFrame(columns=['id', 'sequence', 'structure', 'mfe'])

        # ---- Step 1: 写入临时 FASTA ----
        fasta_path, out_path = self._make_temp_paths(temp_dir)
        self._write_fasta(fasta_path, sequences)

        try:
            # ---- Step 2: 单次调用 RNAfold.exe ----
            self._run_rnafold(fasta_path, out_path)

            # ---- Step 3: 解析输出 ----
            results = self._parse_output(out_path, len(sequences))

        finally:
            # ---- Step 4: 清理 ----
            self._cleanup(fasta_path, out_path)

        return pd.DataFrame(results, columns=['id', 'sequence', 'structure', 'mfe'])

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _make_temp_paths(self, temp_dir: Optional[str] = None) -> Tuple[str, str]:
        """生成临时输入/输出文件路径"""
        base = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix='')
        fasta_path = base.name + '.fasta'
        out_path = base.name + '.out'
        base.close()
        os.unlink(base.name)  # 只保留 .fasta 和 .out
        return fasta_path, out_path

    @staticmethod
    def _write_fasta(fasta_path: str, sequences: List[Tuple[str, str]]) -> None:
        """将序列列表写入 FASTA 文件"""
        with open(fasta_path, 'w', encoding='ascii') as f:
            for seq_id, seq in sequences:
                f.write(f">{seq_id}\n{seq}\n")

    def _run_rnafold(self, fasta_path: str, out_path: str) -> None:
        """
        执行 RNAfold.exe 批处理

        命令: RNAfold.exe -i temp_input.fasta --noPS
        --noPS 禁止生成 PostScript 文件，减少磁盘 IO
        """
        cmd = [
            self._rnafold_path,
            '-i', fasta_path,
            '--noPS',
        ]

        with open(out_path, 'w', encoding='utf-8') as out_f:
            result = subprocess.run(
                cmd,
                stdout=out_f,
                stderr=subprocess.DEVNULL,  # 避免 PIPE 缓冲区阻塞
                timeout=3600,
                check=False,
            )

        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd,
                stderr=f"RNAfold 返回值 {result.returncode}，请检查输入文件",
            )

    def _parse_output(
        self,
        out_path: str,
        expected_count: int,
    ) -> List[List]:
        """
        解析 RNAfold 输出文件

        RNAfold 每条序列输出 3 行：
          >seq_id
          AGCGUAUCG...
          ((...))   (-3.40 kcal/mol)

        第 3 行格式：结构字符串 + 空格 + (MFE kcal/mol)
        """
        results: List[List] = []
        line_buf: List[str] = []

        with open(out_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n\r')
                line_buf.append(line)

                # 每 3 行为一条记录
                if len(line_buf) == 3:
                    header, seq, struct_line = line_buf

                    # 解析 ID（去掉 >）
                    seq_id = header.lstrip('>').strip() if header.startswith('>') else header.strip()

                    # 解析序列
                    sequence = seq.strip()

                    # 解析结构 + MFE
                    # 结构在第 3 行的开头部分（连续的非空格字符）
                    # MFE 在末尾括号中
                    parts = struct_line.strip().split()
                    structure = parts[0] if parts else ''
                    mfe = np.nan

                    if len(parts) >= 2:
                        # 尝试从 "(-3.40)" 或 "-3.40" 提取
                        mfe_str = parts[-1].strip('()')
                        try:
                            mfe = round(float(mfe_str), 4)
                        except ValueError:
                            mfe = np.nan

                    results.append([seq_id, sequence, structure, mfe])
                    line_buf = []

        # 处理不完整的最后一条记录（理论上不会发生）
        if line_buf:
            warnings.warn(f"输出文件中存在 {len(line_buf)} 行未配对的残留数据")

        # 如果解析结果数量不等于期望数量，给出警告
        if len(results) != expected_count:
            warnings.warn(
                f"期望解析 {expected_count} 条，实际解析 {len(results)} 条。"
                f"可能存在解析错误。"
            )

        return results

    @staticmethod
    def _cleanup(fasta_path: str, out_path: str) -> None:
        """删除临时文件"""
        for p in (fasta_path, out_path):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except PermissionError:
                warnings.warn(f"无法删除临时文件: {p}")


# ============================================================================
# 便捷函数（单次调用）
# ============================================================================
def batch_fold(
    sequences: List[Tuple[str, str]],
    rnafold_path: str = RNAFOLD_PATH,
) -> pd.DataFrame:
    """
    便捷函数：单次调用批量折叠

    参数
    ----
    sequences : list of (id, seq)
    rnafold_path : str

    返回
    ----
    pd.DataFrame
    """
    processor = RNAfoldBatchProcessor(rnafold_path)
    return processor.fold_sequences(sequences)


# ============================================================================
# 简单自测
# ============================================================================
if __name__ == '__main__':
    # 测试用例
    test_seq = [
        ("test1", "GGGAAACCC"),
        ("test2", "CCCAUUGGG"),
        ("test3", "AAAAAAAACCCCCCCCGGGGGGGGUUUUUUUU"),
    ]

    try:
        df = batch_fold(test_seq)
        print(df.to_string(index=False))
        print(f"\n测试完成，共 {len(df)} 条")
    except FileNotFoundError as e:
        print(f"RNAfold 未找到: {e}")
    except subprocess.CalledProcessError as e:
        print(f"RNAfold 执行失败 (返回值 {e.returncode}): {e.stderr}")
    except Exception as e:
        print(f"未知错误: {e}")
