

# -*- coding: utf-8 -*-
"""
利用 pandas 和 mygene 批量查询基因的官方 symbol 和 Ensembl ID
处理 mmc1.CSV 中的基因标识符，映射为最新官方名称和 Ensembl ID
最后导出为 mmc1_updated.csv

修改日期：2026-05-30
"""

import pandas as pd
import mygene
import time
import random

# ===================== 1. 读取 CSV 文件 =====================
# 修改日期：2026-05-30
df = pd.read_csv("mmc1.CSV")
print(f"原始数据行数: {len(df)}")
print(f"列名: {list(df.columns)}")

# ===================== 2. 自动识别目标列 =====================
# 修改日期：2026-05-30
# 自动识别包含 "UCSC ID" 或 "ucsc" 或 "uc" 的列作为 UCSC ID 列
ucsc_col = None
gene_col = None

for col in df.columns:
    col_lower = col.lower()
    # 识别 UCSC ID 列：列名包含 ucsc 或 id，或首行值以 uc 开头
    if "ucsc" in col_lower or ("id" in col_lower and "uc" in col_lower):
        ucsc_col = col
    # 识别 Gene 列：列名为 "Gene"
    elif col_lower == "gene":
        gene_col = col

# 如果上面的逻辑没有找到，用更宽松的规则
if gene_col is None:
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ["gene", "gene_name", "symbol", "gene_symbol"]:
            gene_col = col
            break

if ucsc_col is None:
    for col in df.columns:
        col_lower = col.lower()
        if "ucsc" in col_lower:
            ucsc_col = col
            break

print(f"识别到的 UCSC ID 列: {ucsc_col}")
print(f"识别到的 Gene 列: {gene_col}")

# ===================== 3. 初始化 mygene =====================
# 修改日期：2026-05-30
mg = mygene.MyGeneInfo()

# ===================== 4. 分批查询基因 =====================
# 修改日期：2026-05-30
# 获取所有基因 ID 列表，去除 NaN 和空值
gene_ids_all = df[gene_col].dropna().astype(str).str.strip().tolist()
gene_ids_all = [g for g in gene_ids_all if g and g.lower() != "nan"]
# 去重以提高效率
unique_gene_ids = list(set(gene_ids_all))
print(f"待查询的唯一基因数量: {len(unique_gene_ids)}")

# 分批查询（每批 50 个，避免服务器 500 错误）
# 使用 querymany 批量查询
# scopes: 在多个字段中搜索（symbol, alias, accession, entrezgene），最大化召回率
# fields: 需要的返回字段（官方 symbol、ensembl gene ID）
# species: 小鼠
old_to_new_symbol = {}
old_to_ensembl = {}
failed_queries = []
batch_size = 50

for i in range(0, len(unique_gene_ids), batch_size):
    batch = unique_gene_ids[i:i + batch_size]
    # 重试当前批次最多 3 次
    for attempt in range(3):
        try:
            batch_results = mg.querymany(
                batch,
                scopes='symbol,alias,accession,entrezgene',
                fields='symbol,ensembl.gene',
                species='mouse'
            )
            # 解析当前批次结果
            for item in batch_results:
                query = item['query']

                # 提取官方 symbol
                if 'symbol' in item and item['symbol']:
                    old_to_new_symbol[query] = item['symbol']

                # 提取 Ensembl ID（可能为字符串、字典或列表）
                if 'ensembl' in item and item['ensembl']:
                    ensembl_data = item['ensembl']
                    ensembl_id = None

                    if isinstance(ensembl_data, dict):
                        ensembl_id = ensembl_data.get('gene')
                    elif isinstance(ensembl_data, list):
                        if len(ensembl_data) > 0:
                            first = ensembl_data[0]
                            if isinstance(first, dict):
                                ensembl_id = first.get('gene')
                            else:
                                ensembl_id = str(first)
                    else:
                        ensembl_id = str(ensembl_data)

                    if ensembl_id:
                        old_to_ensembl[query] = str(ensembl_id)

            # 当前批次成功，跳出重试循环
            break
        except Exception as e:
            if attempt < 2:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                # 3 次都失败，记录这批 ID
                failed_queries.extend(batch)
                print(f"  批次 {i // batch_size + 1} 失败，跳过 {len(batch)} 个 ID: {e}")

    # 进度提示
    if (i // batch_size + 1) % 20 == 0:
        print(f"  已处理 {i + len(batch)} / {len(unique_gene_ids)} 个基因...")

print(f"成功映射 symbol 的数量: {len(old_to_new_symbol)}")
print(f"成功映射 Ensembl ID 的数量: {len(old_to_ensembl)}")
if failed_queries:
    print(f"查询失败的基因数: {len(failed_queries)} (已跳过)")
    # 记录失败的 ID 以便后续排查
    pd.Series(failed_queries).to_csv("failed_queries.csv", index=False, header=["Gene"])

# ===================== 6. 新增列并映射 =====================
# 修改日期：2026-05-30
# 新增 New_Official_Symbol 列
# 如果查到了新 symbol 就用新 symbol，否则用旧 ID 兜底
df['New_Official_Symbol'] = df[gene_col].astype(str).map(
    lambda x: old_to_new_symbol.get(x, x)
)

# 新增 Ensembl_ID 列
# 如果没查到 Ensembl ID 就留空
df['Ensembl_ID'] = df[gene_col].astype(str).map(
    lambda x: old_to_ensembl.get(x, "")
)

# ===================== 7. 导出结果 =====================
# 修改日期：2026-05-30
df.to_csv("mmc1_updated.csv", index=False)
print(f"处理完成！已导出为 mmc1_updated.csv")
print(f"最终数据行数: {len(df)}")
print(f"最终列名: {list(df.columns)}")