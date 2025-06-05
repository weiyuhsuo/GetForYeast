import zarr
import numpy as np
import pandas as pd
import os
from collections import Counter

# 文件路径
EXPRESSION_CSV_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/20250601_data/WT_gene_filled.csv"
GENE_INFO_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/mapping/Saccharomyces_cerevisiae.gene_info/Saccharomyces_cerevisiae.gene_info"
MAPPING_FILE = "/home/rhyswei/Code/aiyeast/aiyeast-514/5_ver2association/gene_peak_mapping.csv"
ZARR_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/20250601_data/yeast_data_with_conditions_original_peaks.zarr"

def assign_expression():
    print("开始分配基因表达值到 Peak...")

    # 1. 加载基因表达数据
    print("加载基因表达数据...")
    try:
        # 从CSV文件读取，假设第一列是基因名 (LocusTag)
        expression_df = pd.read_csv(EXPRESSION_CSV_FILE, index_col=0)
        expression_df.index.name = 'gene_name'
        # 清理表达数据中的基因名
        expression_df.index = expression_df.index.astype(str).str.strip()

        # 确保所有表达值列都是数值类型
        for col in expression_df.columns:
            # 尝试将列转换为数值，错误的值设为NaN，然后用0填充NaN
            expression_df[col] = pd.to_numeric(expression_df[col], errors='coerce').fillna(0)

        num_samples_exp = expression_df.shape[1]
        num_genes_exp = expression_df.shape[0]
        print(f"基因表达数据形状: {expression_df.shape}")
        print(f"表达数据前5个基因名: {expression_df.index[:5].tolist()}")
        print(f"表达数据前5个样本列名: {expression_df.columns[:5].tolist()}")

    except FileNotFoundError:
        print(f"错误：未找到基因表达文件 {EXPRESSION_CSV_FILE}")
        return
    except Exception as e:
        print(f"读取或处理基因表达文件时发生错误: {e}")
        return

    # 2. 加载和处理基因信息文件以创建全面的基因名映射
    print("加载基因信息文件并创建全面的基因名映射...")
    # mapping: 各种命名 -> Symbol (标准名)
    all_names_to_symbol_map = {}
    # mapping: Symbol (标准名) -> LocusTag (用于表达数据查找)
    symbol_to_locus_tag_map = {}
    try:
        gene_info_columns = [
            'tax_id', 'GeneID', 'Symbol', 'LocusTag', 'Synonyms', 'dbXrefs', 'chromosome', 'map_location',
            'description', 'type_of_gene', 'Symbol_from_nomenclature_authority',
            'Full_name_from_nomenclature_authority', 'Nomenclature_status', 'Other_designations',
            'Modification_date', 'Feature_type'
        ]
        gene_info_df = pd.read_csv(GENE_INFO_FILE, sep='\t', skiprows=1, names=gene_info_columns, header=None)
        # 清理列名：去除开头的#和空格 (虽然指定了names，这里保留以防万一)
        gene_info_df.columns = gene_info_df.columns.str.lstrip('# ').str.strip()
        # 确保必需的列存在
        required_cols = ['Symbol', 'LocusTag', 'Synonyms']
        if not all(col in gene_info_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in gene_info_df.columns]
             raise ValueError(f"基因信息文件中未找到必需的列 {missing}。找到的列: {list(gene_info_df.columns)}")

        # 创建全面的映射：各种命名 -> Symbol
        for index, row in gene_info_df.iterrows():
            symbol = str(row['Symbol']).strip()
            locus_tag = str(row['LocusTag']).strip()
            synonyms_str = str(row.get('Synonyms', '')).strip()

            if symbol:
                # 将Symbol映射到自身
                all_names_to_symbol_map[symbol] = symbol
                # 如果Symbol不是'-'，建立Symbol到LocusTag的映射（用于表达数据查找）
                if symbol != '-' and locus_tag and locus_tag != '-':
                    # 注意：这里可能存在一个Symbol对应多个LocusTag的情况，简单取第一个
                    if symbol not in symbol_to_locus_tag_map:
                         symbol_to_locus_tag_map[symbol] = locus_tag

                # 将LocusTag映射到Symbol
                if locus_tag and locus_tag != '-':
                    all_names_to_symbol_map[locus_tag] = symbol

                # 将Synonyms映射到Symbol
                if synonyms_str and synonyms_str != '-':
                    # 分割同义词，支持逗号、分号或空格分割
                    synonyms = [s.strip() for s in synonyms_str.replace(';', ',').replace(' ', ',').split(',') if s.strip()]
                    for syn in synonyms:
                        if syn:
                            all_names_to_symbol_map[syn] = symbol

    except FileNotFoundError:
        print(f"错误：未找到基因信息文件 {GENE_INFO_FILE}")
        return
    except ValueError as e:
        print(f"基因信息文件列错误: {e}")
        return
    except Exception as e:
        print(f"读取或处理基因信息文件时发生错误: {e}")
        return
    print(f"创建全面基因名映射数量: {len(all_names_to_symbol_map)}")
    print(f"创建Symbol到LocusTag映射数量: {len(symbol_to_locus_tag_map)}")


    # 3. 加载 Peak-Gene 映射数据并标准化基因名
    print("加载 Peak-Gene 映射数据并标准化基因名...")
    # mapping: peak_id -> list of standardized gene names (Symbol)
    peak_to_standardized_genes = {}
    mapped_gene_names_in_mapping_std = set()
    try:
        # 假设映射文件有两列：gene_name 和 peak_id
        mapping_df = pd.read_csv(MAPPING_FILE)
        if 'gene_name' not in mapping_df.columns or 'peak_id' not in mapping_df.columns:
             raise ValueError(f"Peak-Gene 映射文件中未找到必需的列 'gene_name' 或 'peak_id'。找到的列: {list(mapping_df.columns)}")

        # 清理映射文件中的基因名和Peak ID
        mapping_df['gene_name'] = mapping_df['gene_name'].astype(str).str.strip()
        mapping_df['peak_id'] = mapping_df['peak_id'].astype(str).str.strip()

        # 构建 peak_id 到相关联的标准化基因名 (Symbol) 的映射
        for index, row in mapping_df.iterrows():
            original_gene_name = row['gene_name'] # 已经清理过空格
            peak_id = row['peak_id'] # 已经清理过空格

            # 使用全面基因名映射将基因名标准化为 Symbol
            standardized_gene_name = all_names_to_symbol_map.get(original_gene_name, None) # 如果找不到映射，设为None

            if peak_id and standardized_gene_name:
                 if peak_id not in peak_to_standardized_genes:
                     peak_to_standardized_genes[peak_id] = []
                 # 确保同一个 Peak 不重复关联同一个标准化基因名
                 if standardized_gene_name not in peak_to_standardized_genes[peak_id]:
                    peak_to_standardized_genes[peak_id].append(standardized_gene_name)
                 mapped_gene_names_in_mapping_std.add(standardized_gene_name)

        print(f"从映射文件构建 Peak 到标准化基因映射数量: {len(peak_to_standardized_genes)}")
        print(f"映射文件中涉及的标准化基因名数量: {len(mapped_gene_names_in_mapping_std)}")

        # 统计映射情况 (基于Symbol)
        exp_genes_locus_tags = set(expression_df.index.tolist())
        # 将表达数据中的LocusTag转换为Symbol进行比较
        exp_genes_symbols = set()
        unmapped_exp_genes = set() # 统计在gene_info中找不到Symbol的表达基因
        for locus_tag in exp_genes_locus_tags:
            symbol = all_names_to_symbol_map.get(locus_tag, None)
            if symbol:
                exp_genes_symbols.add(symbol)
            else:
                unmapped_exp_genes.add(locus_tag)

        mapping_genes_symbols = set(mapped_gene_names_in_mapping_std)

        common_genes_symbols = list(exp_genes_symbols.intersection(mapping_genes_symbols))
        genes_in_exp_not_in_mapping_symbols = list(exp_genes_symbols - mapping_genes_symbols)
        genes_in_mapping_not_in_exp_symbols = list(mapping_genes_symbols - exp_genes_symbols)

        print(f"\n=== 基因映射统计 (基于Symbol) ===")
        print(f"表达数据中的基因总数 (LocusTag): {len(exp_genes_locus_tags)}")
        print(f"在基因信息文件中找到对应Symbol的表达基因数: {len(exp_genes_symbols)}")
        print(f"在基因信息文件中未找到对应Symbol的表达基因数: {len(unmapped_exp_genes)}")
        print(f"未在基因信息文件中找到对应Symbol的前10个表达基因 (LocusTag): {list(unmapped_exp_genes)[:10]}")

        print(f"Peak-Gene映射文件中涉及的标准化基因总数 (Symbol): {len(mapping_genes_symbols)}")
        print(f"同时存在于表达基因Symbol列表和映射文件Symbol列表中的基因数 (有效基因): {len(common_genes_symbols)}")
        print(f"表达基因Symbol列表中有，但在映射文件Symbol列表中没有的基因数: {len(genes_in_exp_not_in_mapping_symbols)}")
        print(f"表达基因Symbol列表中有，但在映射文件Symbol列表中没有的前10个基因: {genes_in_exp_not_in_mapping_symbols[:10]}")
        print(f"映射文件Symbol列表中有，但在表达基因Symbol列表中没有的基因数: {len(genes_in_mapping_not_in_exp_symbols)}")
        print(f"映射文件Symbol列表中有，但在表达基因Symbol列表中没有的前10个基因: {genes_in_mapping_not_in_exp_symbols[:10]}")

        # 检查 Zarr 中的 Peak 关联基因在表达数据 Symbol 列表中是否找到
        peaks_with_no_effective_genes = 0 # Zarr中的Peak，但在映射文件/表达数据中找不到有效关联基因Symbol
        peaks_with_effective_genes_count = 0 # 统计成功关联到有效基因的 Peak 数量

        # 打开 Zarr 文件 (只读模式用于获取 peak_ids)
        try:
            zarr_root_r = zarr.open(ZARR_FILE, mode='r')
            peak_ids_in_zarr = zarr_root_r['peak_ids'][:].tolist() # 获取 Zarr 中的 Peak IDs
            print(f"在 Zarr 中的 Peak 总数: {len(peak_ids_in_zarr)}")
        except Exception as e:
             print(f"读取 Zarr 文件中的 peak_ids 时发生错误: {e}")
             return

        # 遍历 Zarr 文件中的每个 Peak ID
        for peak_id_in_zarr in peak_ids_in_zarr:
            # 获取与当前 Zarr Peak 关联的标准化基因名 (Symbol) 列表 (来自映射文件)
            standardized_genes_for_peak = peak_to_standardized_genes.get(peak_id_in_zarr, [])

            # 检查这些关联的 Symbol 是否在表达基因的 Symbol 列表中
            effective_genes_for_peak_symbols = [gene_symbol for gene_symbol in standardized_genes_for_peak if gene_symbol in exp_genes_symbols]

            if len(effective_genes_for_peak_symbols) > 0:
                 peaks_with_effective_genes_count += 1
            else:
                 peaks_with_no_effective_genes += 1

        print(f"Zarr 中的 Peak，成功关联到表达基因 Symbol 的数量: {peaks_with_effective_genes_count}")
        print(f"Zarr 中的 Peak，没有找到有效关联基因 Symbol 的数量: {peaks_with_no_effective_genes}")

    except FileNotFoundError:
        print(f"错误：未找到 Peak-Gene 映射文件 {MAPPING_FILE}")
        return
    except ValueError as e:
         print(f"读取或处理 Peak-Gene 映射文件时发生错误: {e}")
         return
    except Exception as e:
        print(f"处理 Peak-Gene 映射数据时发生错误: {e}")
        return

    # 4. 打开 Zarr 文件 (读写模式用于写入 exp_label)
    print("打开 Zarr 文件 (读写模式) ...")
    try:
        root = zarr.open(ZARR_FILE, mode='a') # 使用 'a' 模式以便修改现有数据集
        if 'peak_ids' not in root or 'exp_label' not in root or 'region_motif' not in root:
             print(f"错误: Zarr 文件 {ZARR_FILE} 中未找到必需的数据集 ('peak_ids', 'exp_label', 或 'region_motif')。")
             return
        # 重新获取 peak_ids_in_zarr，确保与读写模式打开的一致
        peak_ids_in_zarr = root['peak_ids'][:].tolist()
        exp_label_zarr = root['exp_label'] # 获取 exp_label 数据集引用
        region_motif_zarr = root['region_motif'] # 获取 region_motif 数据集引用

        # 创建 Zarr 中的 peak_id 到其索引的映射
        zarr_peak_id_to_index = {peak_id: i for i, peak_id in enumerate(peak_ids_in_zarr)}

        print(f"Zarr 文件中的 Peak 数量: {len(peak_ids_in_zarr)}")
        print(f"Zarr 文件中 exp_label 数据集形状: {exp_label_zarr.shape}")
        print(f"Zarr 文件中 region_motif 数据集形状: {region_motif_zarr.shape}")

        # 警告: Zarr 文件中 exp_label 和 region_motif 的样本数和 Peak 数应一致
        if exp_label_zarr.shape[0] != num_samples_exp or exp_label_zarr.shape[1] != len(peak_ids_in_zarr):
             print("警告: Zarr 文件中 exp_label 的样本数或 Peak 数与 Zarr 的 Peak IDs 不符。")
             # 为了安全，这里打印警告并终止执行
             return
        if region_motif_zarr.shape[0] != num_samples_exp or region_motif_zarr.shape[1] != len(peak_ids_in_zarr):
             print("警告: Zarr 文件中 region_motif 的样本数或 Peak 数与 Zarr 的 Peak IDs 不符。")
             # 为了安全，这里打印警告并终止执行
             return

    except FileNotFoundError:
        print(f"错误：未找到 Zarr 文件 {ZARR_FILE}")
        return
    except Exception as e:
        print(f"打开或处理 Zarr 文件时发生错误: {e}")
        return

    # 5. 分配基因表达值到 Peak
    print("开始分配表达值...")

    # 假设表达数据的样本列顺序与 Zarr 文件中的样本顺序一致 (即行顺序一致)
    sample_names = expression_df.columns.tolist()

    filled_count = 0
    peaks_with_no_effective_exp_value = 0 # 统计Zarr中的Peak，虽然关联到有效基因Symbol，但在表达数据中该样本表达值为0或NaN的Peak样本对数量
    peaks_with_no_associated_genes_in_mapping = 0 # 统计Zarr中的Peak，在mapping文件里完全没有关联基因的 (这个统计现在在第3步完成)
    peaks_associated_genes_not_in_exp = 0 # 统计Zarr中的Peak，关联的基因Symbol不在表达数据Symbol列表里的Peak数量 (这个统计现在在第3步完成)

    # 遍历 Zarr 文件中的每个 Peak ID及其索引
    for zarr_peak_index, peak_id in enumerate(peak_ids_in_zarr):

        # 获取与当前 Zarr Peak 关联的标准化基因名 (Symbol) 列表 (来自映射文件)
        standardized_genes_for_peak = peak_to_standardized_genes.get(peak_id, [])

        # 检查这些关联的 Symbol 是否在表达基因的 Symbol 列表中
        effective_genes_for_peak_symbols = [gene_symbol for gene_symbol in standardized_genes_for_peak if gene_symbol in exp_genes_symbols]

        if not effective_genes_for_peak_symbols:
            # 如果 Peak 关联的 Symbol 都不在表达基因 Symbol 列表中，或者在mapping文件里就没有关联基因
            # 这两种情况在步骤3已经统计过，这里只需跳过分配
            continue

        # 遍历每个样本
        for sample_index, sample_name in enumerate(sample_names):
            gene_expression_values = []

            # 收集当前样本中与当前 Peak 关联的、且在表达数据中的基因的表达值
            for gene_symbol in effective_genes_for_peak_symbols:
                # 使用 Symbol 查找对应的 LocusTag (用于在表达数据中查找) - 注意这里可能一个Symbol对应多个LocusTag，我们简单取第一个
                locus_tag = symbol_to_locus_tag_map.get(gene_symbol)

                if locus_tag and locus_tag in expression_df.index:
                    expr_value = expression_df.loc[locus_tag, sample_name]
                    if pd.notna(expr_value):
                         gene_expression_values.append(expr_value)
                    # else: 表达值为NaN，用0填充了，所以不会进入这里

            if gene_expression_values:
                # 计算 Peak 在当前样本的表达值 (求和)
                peak_expression = sum(gene_expression_values)

                # 将计算出的表达值写入 Zarr 的 exp_label 数据集
                try:
                     exp_label_zarr[sample_index, zarr_peak_index] = float(peak_expression)
                     filled_count += 1
                except Exception as e:
                     print(f"写入 Zarr 文件时发生错误: {e}")
                     print(f"样本索引: {sample_index}, Zarr Peak 索引: {zarr_peak_index}, Peak ID: {peak_id}, 表达值: {peak_expression}")
            # else: gene_expression_values 为空，说明关联到的有效基因在该样本表达值为0或NaN，exp_label保持为0

    print(f"分配完成。共填充了 {filled_count} 个 exp_label 值。")
    # 在步骤3已经打印过相关统计，这里可以省略或打印汇总
    # print(f"Zarr 中的 Peak，在mapping文件里完全没有关联基因的数量: {peaks_with_no_associated_genes_in_mapping}")
    # print(f"Zarr 中的 Peak，关联的基因 Symbol 不在表达数据 Symbol 列表里的数量: {peaks_associated_genes_not_in_exp}")
    print(f"Zarr 文件 {ZARR_FILE} 中的 exp_label 数据集已更新。")

if __name__ == "__main__":
    assign_expression() 