import pandas as pd
import numpy as np
import zarr
import os

data_wt_path = 'aiyeast/get_model/input/mapping/data_WT_filled.xlsx'
matrix_path = 'aiyeast/get_model/input/mapping/combined_matrix.csv'
zarr_output_path = 'aiyeast/get_model/input/mapped_data_with_conditions/yeast_data_with_conditions.zarr' # Zarr文件输出路径
gene_annotation_path = 'aiyeast/get_model/input/mapping/ncbiRefSeqCurated.txt'
peaks_bed_path = 'aiyeast/get_model/input/mapping/peaks.bed'
expression_data_path = 'aiyeast/get_model/input/mapping/WT_gene_filled.xlsx' # 基因表达数据
gene_info_path = 'aiyeast/get_model/input/mapping/Saccharomyces_cerevisiae.gene_info/Saccharomyces_cerevisiae.gene_info'

# 创建输出目录（如果不存在）
output_dir = os.path.dirname(zarr_output_path)
os.makedirs(output_dir, exist_ok=True)

try:
    # --- 1. 加载所有原始数据文件 ---
    print("正在加载原始数据文件...")
    df_conditions = pd.read_excel(data_wt_path)
    df_matrix = pd.read_csv(matrix_path) # 加载完整的 Peak*Motif 矩阵
    df_genes_raw = pd.read_csv(gene_annotation_path, sep='\t', header=None) # 原始基因注释文件
    df_peaks_bed_raw = pd.read_csv(peaks_bed_path, sep='\t', header=None) # 原始 Peak bed 文件
    df_expression = pd.read_excel(expression_data_path) # 基因表达数据
    
    # 加载基因信息文件并建立映射关系
    print("正在加载基因信息文件...")
    df_gene_info = pd.read_csv(gene_info_path, sep='\t', header=0)
    # 修正首列名
    if '#tax_id' in df_gene_info.columns:
        df_gene_info = df_gene_info.rename(columns={'#tax_id': 'tax_id'})
    # 建立基因名映射字典（Symbol <-> Symbol_from_nomenclature_authority）
    gene_name_mapping = {}
    for _, row in df_gene_info.iterrows():
        symbol = str(row['Symbol']) if pd.notna(row['Symbol']) else None
        standard_name = str(row['Symbol_from_nomenclature_authority']) if pd.notna(row['Symbol_from_nomenclature_authority']) else None
        if symbol and standard_name and standard_name != '-':
            gene_name_mapping[symbol] = standard_name
            gene_name_mapping[standard_name] = symbol
    print(f"已建立 {len(gene_name_mapping)} 个基因名映射关系")
    print("原始数据文件加载完成。")

    # --- 2. 解析和处理基因注释和 Peak 位置文件 ---
    print("正在解析基因注释文件...")
    gene_cols = ['bin', 'name', 'chrom', 'strand', 'txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'exonCount', 'exonStarts', 'exonEnds', 'score', 'name2', 'cdsStartStat', 'cdsEndStat', 'exonFrames']
    df_genes = df_genes_raw.copy()
    df_genes.columns = gene_cols
    df_genes['tss'] = df_genes.apply(lambda row: row['txStart'] if row['strand'] == '+' else row['txEnd'], axis=1)
    print("基因注释文件解析完成。")
    # print("前几行基因注释数据：")
    # print(df_genes.head())

    print("正在解析Peak位置文件...")
    peak_bed_cols = ['chrom', 'chromStart', 'chromEnd', 'name']
    df_peaks_bed = df_peaks_bed_raw.copy()
    df_peaks_bed.columns = peak_bed_cols
    df_peaks_bed['peak_center'] = (df_peaks_bed['chromStart'] + df_peaks_bed['chromEnd']) // 2
    print("Peak位置文件解析完成。")
    # print("前几行Peak位置数据：")
    # print(df_peaks_bed.head())
    
    # 将Peak bed信息与 combined_matrix.csv 中的 peak_id 对齐
    df_matrix_peak_ids = df_matrix['peak_id'].to_frame() # 从matrix文件获取的peak_id
    
    # 将 peak_id 与 peak bed 信息合并
    df_peaks = pd.merge(df_matrix_peak_ids, df_peaks_bed, left_on='peak_id', right_on='name', how='left')
    df_peaks = df_peaks.drop(columns=['name']).rename(columns={'peak_id': 'name'}) # 重命名回 name 列，并保留 peak_id 的值
    
    if df_peaks.isnull().any().any():
        print("警告：部分 Peak ID 未在 peaks.bed 文件中找到匹配项。这部分Peak可能无法关联到基因。")
        # TODO: 处理未匹配的 Peak

    print("Peak信息准备完成。")
    # print("前几行Peak信息：")
    # print(df_peaks.head())
    
    # --- 3. 选择和处理实验条件列 ---
    print("正在处理实验条件...")
    exclude_condition_columns = ['Unnamed: 0', 'GSE', 'GSM', '菌株', '基因组文件名', '基因表达文件名', 'Mark']
    condition_columns_to_use = [col for col in df_conditions.columns if col not in exclude_condition_columns]
    
    if not condition_columns_to_use:
        print("警告：没有找到符合条件的实验条件列用于整合。")
        condition_features = np.zeros((df_conditions.shape[0], 0)) # 创建一个空的条件特征数组
        num_conditions = 0
    else:
        df_conditions_subset = df_conditions[condition_columns_to_use]
        categorical_cols = df_conditions_subset.select_dtypes(include=['object', 'category']).columns.tolist()
        df_conditions_processed = pd.get_dummies(df_conditions_subset, columns=categorical_cols, dummy_na=False)
        condition_features = df_conditions_processed.values
        num_conditions = condition_features.shape[1]
        print(f"将使用的实验条件列 ({num_conditions} 列): {df_conditions_processed.columns.tolist()}")

    # --- 4. 准备 Peak*Motif 数据 ---
    print("正在准备 Peak*Motif 数据...")
    peak_ids_matrix = df_matrix['peak_id'].values.astype(str) # 从matrix文件获取的peak_id
    peak_motif_features_base = df_matrix.drop(columns=['peak_id']).values
    
    num_peaks = peak_motif_features_base.shape[0]
    num_motif_features_base = peak_motif_features_base.shape[1]
    print(f"Peak*Motif 矩阵形状: ({num_peaks}, {num_motif_features_base})")
    
    # 确保 peak_ids_matrix 和 df_peaks 中的 Peak 顺序一致
    # 这里假设 combined_matrix.csv 中的 peak_id 顺序与 peaks.bed 和 df_peaks 一致
    # 如果不一致，需要根据 peak_id 对齐 peak_motif_features_base

    # --- 5. 整合特征并构建三维数组 ---
    print("正在整合特征...")
    total_features_per_peak = num_motif_features_base + num_conditions
    all_gsm_features = np.zeros((66, num_peaks, total_features_per_peak), dtype=np.float32)

    for i in range(66):
        if num_conditions > 0:
             condition_matrix_i = np.tile(condition_features[i, :], (num_peaks, 1))
             all_gsm_features[i, :, :] = np.concatenate((peak_motif_features_base, condition_matrix_i), axis=1)
        else:
             all_gsm_features[i, :, :] = peak_motif_features_base
    print("特征整合完成。")

    # --- 6. 计算 exp_label 数组（在此处添加分配逻辑）---
    print("正在计算 Peak 的 exp_label...")
    
    # 加载基因表达数据 (已加载 df_expression)
    # 假设第一列是基因ID，其他列是GSM样本的表达值
    expression_genes = df_expression.iloc[:, 0].tolist() # 假设第一列是基因ID/名称
    expression_gsm_columns = df_expression.columns[1:].tolist() # GSM样本列名
    expression_values = df_expression.iloc[:, 1:].values # 基因表达值 (Genes x GSMs)
    
    # --- 输出GSM样本列表进行比对 ---
    gsm_order_conditions = df_conditions['GSM'].tolist()
    print(f"实验条件文件中的GSM样本 ({len(gsm_order_conditions)}): {gsm_order_conditions}")
    print(f"基因表达文件中的GSM样本 ({len(expression_gsm_columns)}): {expression_gsm_columns}")
    # ------------------------------

    # 确保基因表达数据的GSM列顺序与条件数据的GSM顺序一致
    gsm_order = df_conditions['GSM'].tolist()
    
    # 从基因表达文件的列名中提取GSM编号
    expression_gsm_columns = df_expression.columns[1:].tolist()
    expression_gsm_ids = []
    for col in expression_gsm_columns:
        if '_' in col:
            # 如果包含下划线，取第一部分作为GSM编号
            gsm_id = col.split('_')[0]
        else:
            # 如果不包含下划线，直接使用列名
            gsm_id = col
        expression_gsm_ids.append(gsm_id)
    
    # 检查所有gsm_order中的GSM是否都在expression_gsm_ids中
    missing_gsms = [gsm for gsm in gsm_order if gsm not in expression_gsm_ids]
    if missing_gsms:
        print(f"警告：以下GSM样本在基因表达文件中未找到，将被移除：{missing_gsms}")
        # 从gsm_order中移除不匹配的GSM
        gsm_order = [gsm for gsm in gsm_order if gsm in expression_gsm_ids]
        # 更新条件数据
        df_conditions = df_conditions[df_conditions['GSM'].isin(gsm_order)]
        condition_features = condition_features[df_conditions.index]
        print(f"更新后的GSM样本数量：{len(gsm_order)}")
    
    # 获取GSM样本在表达数据中的索引
    expression_gsm_indices = [expression_gsm_ids.index(gsm) for gsm in gsm_order]
    expression_values_ordered = expression_values[:, expression_gsm_indices]  # 形状 (Genes, len(gsm_order))
    
    # 构建基因表达值的字典，方便查找
    gene_expression_dict = {}
    for i, gene in enumerate(expression_genes):
        # 尝试获取基因的所有可能名称
        gene_names = {gene}
        if gene in gene_name_mapping:
            gene_names.add(gene_name_mapping[gene])
        
        # 为每个可能的基因名都存储表达值
        for gene_name in gene_names:
            gene_expression_dict[gene_name] = expression_values_ordered[i, :]

    # 构建 Peak 到基因的映射表 和 基因到 Peak 的映射表
    peak_to_genes = {} # {peak_index: [(gene_index, distance), ...]}
    gene_to_peaks = {} # {gene_index: [(peak_index, distance), ...]}
    tss_window = 4000 # TSS 上下游 +/- 2000 bp 窗口
    
    # 创建基因位置和TSS的查找结构 (例如 interval tree 或简单的按染色体分组)
    genes_by_chrom = {}
    for i, row in df_genes.iterrows():
        chrom = row['chrom']
        tss = row['tss']
        gene_name = row['name2'] # 使用name2作为基因ID
        if pd.isna(chrom) or pd.isna(tss) or pd.isna(gene_name):
             continue # 跳过无效基因记录
        chrom = str(chrom) # 确保染色体是字符串
        if chrom not in genes_by_chrom:
            genes_by_chrom[chrom] = []
        genes_by_chrom[chrom].append({'index': i, 'tss': tss, 'name': gene_name, 'strand': row['strand']})

    print("正在建立 Peak-Gene 关联...")
    # 遍历 Peak
    for peak_index, row in df_peaks.iterrows():
        peak_chrom = row['chrom']
        peak_center = row['peak_center']
        peak_id = row['name'] # 使用peaks.bed中的name作为peak_id
        
        if pd.isna(peak_chrom) or pd.isna(peak_center):
            continue # 跳过没有位置信息的Peak
        peak_chrom = str(peak_chrom) # 确保染色体是字符串
            
        if peak_chrom not in genes_by_chrom:
             continue # 跳过没有基因注释的染色体
             
        associated_genes = []
        # 遍历该染色体上的基因
        for gene_info in genes_by_chrom[peak_chrom]:
            gene_tss = gene_info['tss']
            # 检查 Peak 中心是否在 TSS 窗口内
            if peak_center >= gene_tss - tss_window // 2 and peak_center <= gene_tss + tss_window // 2:
                distance = abs(peak_center - gene_tss)
                associated_genes.append({'gene_index': gene_info['index'], 'distance': distance, 'name': gene_info['name']})

        if associated_genes:
             peak_to_genes[peak_index] = associated_genes
             for gene in associated_genes:
                  gene_index = gene['gene_index']
                  distance = gene['distance']
                  if gene_index not in gene_to_peaks:
                       gene_to_peaks[gene_index] = []
                  gene_to_peaks[gene_index].append({'peak_index': peak_index, 'distance': distance})

    print("Peak-Gene 关联建立完成。")

    # 计算 Peak 的 exp_label (形状 66 x num_peaks x 1)
    calculated_exp_label_data = np.zeros((66, num_peaks, 1), dtype=np.float32)

    print("正在分配表达值到 Peak...")
    # 应用分配规则
    # 首先处理基因到 Peak 的映射 (一对多规则)，找到每个基因最近的 Peak
    gene_nearest_peak = {} # {gene_index: (nearest_peak_index, distance)}
    for gene_index, peaks_list in gene_to_peaks.items():
         if peaks_list:
              # 找到距离最近的 Peak
              nearest_peak = min(peaks_list, key=lambda x: x['distance'])
              gene_nearest_peak[gene_index] = nearest_peak['peak_index']
              
    # 然后处理 Peak 到基因的映射 (多对一规则)，并结合一对多规则
    for peak_index in range(num_peaks):
         total_expression_per_gsm = np.zeros(66) # 存储该 Peak 在每个GSM下的总表达值
         
         if peak_index in peak_to_genes:
              associated_genes = peak_to_genes[peak_index]
              
              for gene_info in associated_genes:
                  gene_index = gene_info['gene_index']
                  gene_name = gene_info['name']
                  
                  # 应用一对多规则：只考虑距离该基因最近的那个 Peak
                  if gene_index in gene_nearest_peak and gene_nearest_peak[gene_index] == peak_index:
                      # 尝试使用原始基因名和映射后的基因名
                      if gene_name in gene_expression_dict:
                          total_expression_per_gsm += gene_expression_dict[gene_name]
                      elif gene_name in gene_name_mapping and gene_name_mapping[gene_name] in gene_expression_dict:
                          total_expression_per_gsm += gene_expression_dict[gene_name_mapping[gene_name]]
                      # else: 警告：基因表达文件中未找到该基因的任何形式
              
         calculated_exp_label_data[:, peak_index, 0] = total_expression_per_gsm # 分配给该 Peak 的 exp_label

    print("表达值分配完成。")

    # --- 7. 保存为 Zarr 文件 (更新 exp_label) ---
    print(f"正在更新 Zarr 文件中的 exp_label: {zarr_output_path}")
    root = zarr.open(zarr_output_path, mode='r+') # 以读写模式打开现有Zarr文件

    # 更新 exp_label 数组
    root['exp_label'][:] = calculated_exp_label_data
    print("exp_label 数组更新完成。")
    
    print("Zarr 文件更新成功！")

except FileNotFoundError as e:
    print(f"错误：文件未找到: {e}")
except Exception as e:
    print(f"处理文件时发生错误：{e}") 