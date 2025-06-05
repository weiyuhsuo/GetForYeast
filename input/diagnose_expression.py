import pandas as pd
import numpy as np
from collections import defaultdict
import os

# 文件路径
data_wt_path = 'aiyeast/get_model/input/mapping/data_WT_filled.xlsx'
matrix_path = 'aiyeast/get_model/input/mapping/peak_motif_matrix.csv'
gene_annotation_path = 'aiyeast/get_model/input/mapping/ncbiRefSeqCurated.txt'
peaks_bed_path = 'aiyeast/get_model/input/mapping/peaks.bed'
expression_data_path = 'aiyeast/get_model/input/mapping/WT_gene_filled.xlsx'
gene_info_path = 'aiyeast/get_model/input/mapping/Saccharomyces_cerevisiae.gene_info/Saccharomyces_cerevisiae.gene_info'

print("=== 1. 检查基因表达数据分布 ===")
# 读取基因表达数据
expression_data = pd.read_excel(expression_data_path)
print(f"表达数据形状: {expression_data.shape}")
print(f"基因数量: {len(expression_data)}")
print(f"GSM样本数量: {len(expression_data.columns) - 1}")  # 减去基因名列

# 统计非零表达值的基因数量
non_zero_genes = (expression_data.iloc[:, 1:] != 0).any(axis=1).sum()
print(f"在至少一个GSM样本下有非零表达值的基因数量: {non_zero_genes}")
print(f"非零表达值基因比例: {(non_zero_genes/len(expression_data))*100:.2f}%")

print("\n=== 2. 检查基因名映射 ===")
# 使用tab分隔符读取gene_info文件
gene_info = pd.read_csv(gene_info_path, sep='\t', header=0)
print(f"基因信息文件中的基因数量: {len(gene_info)}")
print(f"gene_info.columns: {gene_info.columns.tolist()}")

print("\n检查基因信息文件中的基因名样本：")
print("前5行基因信息：")
for _, row in gene_info.head().iterrows():
    print(f"Symbol: {row['Symbol']}")
    print(f"LocusTag: {row['LocusTag']}")
    print(f"Synonyms: {row['Synonyms']}")
    print("-" * 50)

# 创建基因名映射字典
gene_name_to_id = {}
for _, row in gene_info.iterrows():
    # 主要使用LocusTag字段（系统命名）
    if pd.notna(row['LocusTag']) and row['LocusTag'] != '-':
        gene_name_to_id[row['LocusTag']] = row['GeneID']
    # Symbol字段（标准名）
    if pd.notna(row['Symbol']) and row['Symbol'] != '-':
        gene_name_to_id[row['Symbol']] = row['GeneID']
    # Synonyms字段
    if pd.notna(row['Synonyms']) and row['Synonyms'] != '-':
        for synonym in str(row['Synonyms']).split('|'):
            s = synonym.strip()
            if s and s != '-':
                gene_name_to_id[s] = row['GeneID']

print(f"\n基因名映射字典大小: {len(gene_name_to_id)}")

# 检查表达数据中的基因名样本
print("\n检查表达数据中的基因名样本：")
print("前5个基因名：")
print(expression_data.iloc[:5, 0].tolist())

# 检查表达数据中的基因名是否能在映射中找到
expression_genes = set(expression_data.iloc[:, 0])
mapped_genes = set(gene_name_to_id.keys())
found_genes = expression_genes.intersection(mapped_genes)
print(f"\n表达数据中的基因数量: {len(expression_genes)}")
print(f"能在映射中找到的基因数量: {len(found_genes)}")
print(f"基因映射覆盖率: {(len(found_genes)/len(expression_genes))*100:.2f}%")

# 打印一些未找到映射的基因名样本
if len(found_genes) < len(expression_genes):
    print("\n未找到映射的基因名样本（前5个）：")
    unmapped_genes = expression_genes - mapped_genes
    print(list(unmapped_genes)[:5])

print("\n=== 3. 检查Peak与基因TSS窗口的关联 ===")
# 读取Peak位置信息
peaks_df = pd.read_csv(peaks_bed_path, sep='\t', header=None, 
                      names=['chrom', 'start', 'end', 'peak_id'])
peaks_df['center'] = (peaks_df['start'] + peaks_df['end']) // 2

# 读取基因注释信息
gene_annotations = []
with open(gene_annotation_path, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) >= 9:
            chrom = fields[2]
            strand = fields[3]
            gene_name = fields[8]  # 基因名直接在第9列
            if strand == '+':
                tss = int(fields[4])  # 正链基因的TSS是起始位置
            else:
                tss = int(fields[5])  # 负链基因的TSS是终止位置
            gene_annotations.append({
                'chrom': chrom,
                'tss': tss,
                'gene_name': gene_name
            })

print(f"\n读取到的基因注释数量: {len(gene_annotations)}")
# 检查基因注释中的基因名有多少能在映射中找到
annotation_gene_names = set([g['gene_name'] for g in gene_annotations])
annotated_genes_in_mapping = annotation_gene_names.intersection(mapped_genes)
print(f"基因注释中的基因数量: {len(annotation_gene_names)}")
print(f"能在基因名映射中找到的注释基因数量: {len(annotated_genes_in_mapping)}")
print(f"基因注释到映射的覆盖率: {(len(annotated_genes_in_mapping)/len(annotation_gene_names))*100:.2f}%")

print("前5个基因注释示例：")
for gene in gene_annotations[:5]:
    print(f"染色体: {gene['chrom']}, TSS: {gene['tss']}, 基因名: {gene['gene_name']}")

# 统计Peak与基因TSS窗口的关联情况
window_size = 2000  # TSS窗口大小
peak_gene_distances = []
peaks_with_genes = set()

for peak in peaks_df.itertuples():
    peak_center = peak.center
    peak_chrom = peak.chrom
    
    # 查找在TSS窗口内的基因
    for gene in gene_annotations:
        if gene['chrom'] == peak_chrom:
            distance = abs(gene['tss'] - peak_center)
            if distance <= window_size:
                peak_gene_distances.append({
                    'peak_id': peak.peak_id,
                    'gene_name': gene['gene_name'],
                    'distance': distance
                })
                peaks_with_genes.add(peak.peak_id)

print(f"\n总Peak数量: {len(peaks_df)}")
print(f"关联到至少一个基因的Peak数量: {len(peaks_with_genes)}")
print(f"Peak-基因关联比例: {(len(peaks_with_genes)/len(peaks_df))*100:.2f}%")

if peak_gene_distances:
    distances = [d['distance'] for d in peak_gene_distances]
    print(f"Peak到关联基因TSS的距离统计:")
    print(f"  最小距离: {min(distances)}")
    print(f"  最大距离: {max(distances)}")
    print(f"  平均距离: {np.mean(distances):.2f}")
    print(f"  中位数距离: {np.median(distances):.2f}")

print("\n=== 4. 检查最近Peak分配规则 ===")
# 按基因分组，找出每个基因最近的Peak
gene_nearest_peaks = defaultdict(list)
for pg in peak_gene_distances:
    gene_nearest_peaks[pg['gene_name']].append((pg['peak_id'], pg['distance']))

print(f"至少关联到一个Peak的基因数量: {len(gene_nearest_peaks)}")

# 统计每个基因关联的Peak数量
associated_peak_counts = [len(peaks) for peaks in gene_nearest_peaks.values()]
if associated_peak_counts:
    print(f"每个基因关联的Peak数量统计：")
    print(f"  最小: {min(associated_peak_counts)}")
    print(f"  最大: {max(associated_peak_counts)}")
    print(f"  平均: {np.mean(associated_peak_counts):.2f}")
    print(f"  中位数: {np.median(associated_peak_counts):.2f}")

print("\n检查每个基因关联的Peak及其距离（前5个基因）：")
# 打印前5个基因的关联信息
for i, (gene, peaks) in enumerate(gene_nearest_peaks.items()):
    if i >= 5: break
    print(f"基因: {gene}")
    # 按距离排序后打印
    sorted_peaks = sorted(peaks, key=lambda x: x[1])
    for peak_id, distance in sorted_peaks:
        print(f"  Peak: {peak_id}, 距离TSS: {distance}")
    # 找出并打印最近的Peak
    if sorted_peaks:
        nearest_peak = sorted_peaks[0][0]
        print(f"  -> 最近Peak: {nearest_peak}")
    else:
        print("  -> 没有关联的Peak")
    print("-" * 50)

# 统计每个基因的最近Peak
nearest_peaks_ids = set()
genes_with_nearest_peak = 0
genes_with_nearest_peak_and_expression = 0

# 读取peak_motif_matrix文件以获取其中的Peak ID列表
df_matrix = pd.read_csv(matrix_path)
matrix_peak_ids = set(df_matrix['peak_id'].tolist())
print(f"\npeak_motif_matrix中的Peak数量: {len(matrix_peak_ids)}")

# 将表达数据转换为以基因名为索引的DataFrame，方便查找
expression_data_indexed = expression_data.set_index(expression_data.columns[0])

print("\n检查被选为最近Peak的基因和Peak的属性：")
# 检查每个基因的最近Peak及其属性
for gene, peaks in gene_nearest_peaks.items():
    if peaks:
        # 找出最近的Peak
        sorted_peaks = sorted(peaks, key=lambda x: x[1])
        nearest_peak_id = sorted_peaks[0][0]
        distance_to_tss = sorted_peaks[0][1]
        
        # 检查这个最近Peak是否在peak_motif_matrix中
        is_in_matrix = nearest_peak_id in matrix_peak_ids
        
        # 检查这个基因名是否在表达数据中
        gene_in_expression = gene in expression_data_indexed.index
        has_non_zero_expression = False
        if gene_in_expression:
             # 检查是否有非零表达值
             if (expression_data_indexed.loc[gene, :] != 0).any():
                  has_non_zero_expression = True
             
        # 如果基因名能在映射中找到，进一步检查其系统命名或标准命名是否在表达数据中
        mapped_gene_id = gene_name_to_id.get(gene)
        mapped_gene_names_in_expression = False
        if mapped_gene_id:
             # 查找映射到同一个GeneID的所有基因名
             names_for_id = [name for name, gid in gene_name_to_id.items() if gid == mapped_gene_id]
             # 检查这些名字是否在表达数据中
             if any(name in expression_data_indexed.index for name in names_for_id):
                  mapped_gene_names_in_expression = True

        # 统计成功找到最近Peak的基因数量（不考虑表达值）
        genes_with_nearest_peak += 1
        
        # 统计满足所有条件的基因数量：有最近Peak，最近Peak在matrix中，基因在表达数据中有非零表达
        # 注意：这里的gene变量是来自ncbiRefSeqCurated.txt的基因名
        # 我们需要检查的是这个基因对应的表达数据中的系统命名或标准命名是否有非零表达
        if is_in_matrix and mapped_gene_names_in_expression and has_non_zero_expression: # 这里的条件需要进一步精炼，确保检查的是正确的表达值
             genes_with_nearest_peak_and_expression += 1
             nearest_peaks_ids.add(nearest_peak_id) # 将符合条件的最近Peak ID加入集合
             # 打印符合条件的基因和Peak信息（前10个）
             if len(nearest_peaks_ids) <= 10:
                 print(f"符合条件（有最近Peak，在matrix中，基因在表达数据中有非零表达）的基因和Peak：")
                 print(f"  基因: {gene}, 最近Peak: {nearest_peak_id}, 距离TSS: {distance_to_tss}")
                 print(f"  基因在表达数据中: {gene_in_expression}, 有非零表达: {has_non_zero_expression}")
                 print(f"  mapped_gene_id: {mapped_gene_id}, 映射名在表达数据中: {mapped_gene_names_in_expression}")
                 print("-"*50)

print(f"\n至少关联到一个Peak的基因数量: {len(gene_nearest_peaks)}")
print(f"成功找到最近Peak的基因数量 (有Peak关联): {genes_with_nearest_peak}")
print(f"成功找到最近Peak且Peak在matrix中且基因在表达数据中有非零表达的基因数量: {genes_with_nearest_peak_and_expression}")
print(f"被选为最近Peak且符合所有条件的Peak数量 (去重后): {len(nearest_peaks_ids)}")
print(f"符合条件的最近Peak比例 (占总Peak): {(len(nearest_peaks_ids)/len(peaks_df))*100:.2f}%")

print("\n诊断完成。") 