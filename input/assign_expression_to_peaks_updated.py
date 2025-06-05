import pandas as pd
import zarr
import numpy as np
import os
from collections import Counter
import re

# 文件路径
EXPRESSION_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/mapping/WT_gene_filled.xlsx"
MAPPING_FILE = "/home/rhyswei/Code/aiyeast/aiyeast-514/5_ver2association/gene_peak_mapping.csv"
GENE_INFO_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/mapping/Saccharomyces_cerevisiae.gene_info/Saccharomyces_cerevisiae.gene_info"
ZARR_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/20250601_data/yeast_data_with_conditions.zarr"

def analyze_gene_name(gene_name):
    """分析基因名的特征"""
    features = {
        'length': len(gene_name),
        'has_underscore': '_' in gene_name,
        'has_dash': '-' in gene_name,
        'has_dot': '.' in gene_name,
        'is_uppercase': gene_name.isupper(),
        'starts_with_y': gene_name.startswith('Y'),
        'has_digit': bool(re.search(r'\d', gene_name)),
        'format': 'unknown'
    }
    
    # 判断基因名格式
    if re.match(r'^Y[A-Z]\d{3}[A-Z](-[A-Z])?$', gene_name):
        features['format'] = 'systematic'
    elif re.match(r'^[A-Z]{1,3}\d{1,2}$', gene_name):
        features['format'] = 'standard'
    elif gene_name.startswith('t') or gene_name.startswith('r'):
        features['format'] = 'RNA'
    else:
        features['format'] = 'other'
    
    return features

# 1. 加载数据
print("加载数据...")
expression_df = pd.read_excel(EXPRESSION_FILE, index_col=0)
mapping_df = pd.read_csv(MAPPING_FILE)

print("\n=== 数据加载统计 ===")
print(f"表达矩阵形状: {expression_df.shape}")
print(f"映射文件行数: {len(mapping_df)}")
print(f"映射文件列名: {mapping_df.columns.tolist()}")

# 分析表达数据中的基因名
print("\n=== 表达数据基因名分析 ===")
expression_genes = expression_df.index.tolist()
expression_features = [analyze_gene_name(gene) for gene in expression_genes]
expression_format_counts = Counter(f['format'] for f in expression_features)
print("表达数据基因名格式分布:")
for format_type, count in expression_format_counts.items():
    print(f"{format_type}: {count}")

# 分析映射文件中的基因名
print("\n=== 映射文件基因名分析 ===")
mapping_genes = mapping_df['gene_name'].unique().tolist()
mapping_features = [analyze_gene_name(gene) for gene in mapping_genes]
mapping_format_counts = Counter(f['format'] for f in mapping_features)
print("映射文件基因名格式分布:")
for format_type, count in mapping_format_counts.items():
    print(f"{format_type}: {count}")

# 自动查找基因信息文件的表头行
print("\n查找基因信息文件表头...")
header_line = None
with open(GENE_INFO_FILE, 'r') as f:
    for i, line in enumerate(f):
        if line.startswith('#tax_id'):
            header_line = i
            break
if header_line is None:
    raise ValueError('未找到基因信息文件的表头行(#tax_id)')

# 用正确的header行读取
print(f"用第{header_line}行为表头读取基因信息文件...")
gene_info_df = pd.read_csv(GENE_INFO_FILE, sep='\t', header=header_line)
# 清洗列名，去除#和空格
new_columns = [col.lstrip('#').strip() for col in gene_info_df.columns]
gene_info_df.columns = new_columns
print("基因信息文件列名:", gene_info_df.columns.tolist())

print("\n=== 基因信息文件统计 ===")
print(f"基因信息文件行数: {len(gene_info_df)}")
print(f"非空Symbol数量: {gene_info_df['Symbol'].notna().sum()}")
print(f"非空Synonyms数量: {gene_info_df['Synonyms'].notna().sum()}")
print(f"非空Symbol_from_nomenclature_authority数量: {gene_info_df['Symbol_from_nomenclature_authority'].notna().sum()}")

# 分析基因信息文件中的基因名
print("\n=== 基因信息文件基因名分析 ===")
info_genes = gene_info_df['Symbol'].tolist()
info_features = [analyze_gene_name(gene) for gene in info_genes]
info_format_counts = Counter(f['format'] for f in info_features)
print("基因信息文件基因名格式分布:")
for format_type, count in info_format_counts.items():
    print(f"{format_type}: {count}")

# 构建更完整的基因名映射
gene_name_map = {}
systematic_to_standard = {}  # 系统命名到标准名的映射

for _, row in gene_info_df.iterrows():
    symbol = str(row['Symbol']).strip()
    synonyms = str(row['Synonyms']).strip().split('|')
    systematic = str(row['Symbol_from_nomenclature_authority']).strip()
    
    # 跳过空值
    if symbol == '-' or symbol == 'nan':
        continue
    
    # 1. 标准名映射
    gene_name_map[symbol] = symbol
    for syn in synonyms:
        syn = syn.strip()
        if syn != '-' and syn != 'nan' and syn not in gene_name_map:
            gene_name_map[syn] = symbol
    
    # 2. 系统命名映射
    if systematic != '-' and systematic != 'nan':
        systematic_to_standard[systematic] = symbol
        gene_name_map[systematic] = symbol

print(f"\n建立了 {len(gene_name_map)} 个基因名映射")
print(f"其中系统命名到标准名的映射: {len(systematic_to_standard)} 个")

# 应用基因名映射到表达数据和映射文件
print("\n应用基因名映射...")
def apply_gene_map(gene_name, gene_map):
    return gene_map.get(gene_name, gene_name)

# 记录映射前后的基因名
expression_genes_before = set(expression_df.index)
mapping_genes_before = set(mapping_df['gene_name'])

expression_df.index = expression_df.index.map(lambda x: apply_gene_map(x, gene_name_map))
mapping_df['gene_name'] = mapping_df['gene_name'].map(lambda x: apply_gene_map(x, gene_name_map))

expression_genes_after = set(expression_df.index)
mapping_genes_after = set(mapping_df['gene_name'])

print("\n=== 基因名映射统计 ===")
print(f"表达数据映射前基因数: {len(expression_genes_before)}")
print(f"表达数据映射后基因数: {len(expression_genes_after)}")
print(f"映射文件映射前基因数: {len(mapping_genes_before)}")
print(f"映射文件映射后基因数: {len(mapping_genes_after)}")

# 2. 构建 Peak 到基因的映射字典
print("\n构建Peak到基因的映射字典...")
mapped_expression_genes = set(expression_df.index)
mapped_mapping_genes = set(mapping_df['gene_name'])
valid_genes = mapped_expression_genes.intersection(mapped_mapping_genes)

print(f"\n=== 基因交集统计 ===")
print(f"表达数据中的基因数: {len(mapped_expression_genes)}")
print(f"映射文件中的基因数: {len(mapped_mapping_genes)}")
print(f"有效基因数: {len(valid_genes)}")

# 统计未映射的基因
unmapped_expression_genes = mapped_expression_genes - valid_genes
unmapped_mapping_genes = mapped_mapping_genes - valid_genes

print(f"\n=== 未映射基因统计 ===")
print(f"表达数据中未映射的基因数: {len(unmapped_expression_genes)}")
print(f"映射文件中未映射的基因数: {len(unmapped_mapping_genes)}")

# 分析未映射基因的特征
if len(unmapped_expression_genes) > 0:
    print("\n表达数据中未映射的基因示例（前5个）及其特征:")
    for gene in list(unmapped_expression_genes)[:5]:
        features = analyze_gene_name(gene)
        print(f"\n基因: {gene}")
        for key, value in features.items():
            print(f"  {key}: {value}")

if len(unmapped_mapping_genes) > 0:
    print("\n映射文件中未映射的基因示例（前5个）及其特征:")
    for gene in list(unmapped_mapping_genes)[:5]:
        features = analyze_gene_name(gene)
        print(f"\n基因: {gene}")
        for key, value in features.items():
            print(f"  {key}: {value}")

peak_to_genes = {}
for _, row in mapping_df.iterrows():
    peak_id = row['peak_id']
    gene_name = row['gene_name']
    if gene_name in valid_genes:
        if peak_id not in peak_to_genes:
            peak_to_genes[peak_id] = []
        peak_to_genes[peak_id].append(gene_name)

print(f"\n建立了 {len(peak_to_genes)} 个Peak到基因的映射")

# 统计每个Peak关联的基因数
peak_gene_counts = Counter(len(genes) for genes in peak_to_genes.values())
print("\n=== Peak关联基因数统计 ===")
print("每个Peak关联的基因数分布:")
for count, num_peaks in sorted(peak_gene_counts.items()):
    print(f"关联{count}个基因的Peak数: {num_peaks}")

# 输出一些示例
print("\n=== 示例Peak及其关联基因 ===")
for peak_id, genes in list(peak_to_genes.items())[:5]:
    print(f"\nPeak {peak_id} 关联的基因:")
    for gene in genes:
        print(f"  - {gene}")

# 3. 加载 Zarr 文件并计算 exp_label
print(f"\n加载Zarr文件 {ZARR_FILE} ...")
try:
    root = zarr.open(ZARR_FILE, mode='a')
except FileNotFoundError:
    print(f"错误: Zarr文件 {ZARR_FILE} 未找到。请先创建Zarr文件。")
    exit()
except Exception as e:
    print(f"加载Zarr文件时出错: {e}")
    exit()

# 获取 Peak IDs
try:
    peak_ids_zarr = root['peak_ids']
    peak_ids = [str(pid) for pid in peak_ids_zarr]
except KeyError:
    print("错误: Zarr文件中未找到'peak_ids'数组。请确认Zarr文件结构。")
    exit()
except Exception as e:
    print(f"读取或处理peak_ids时出错: {e}")
    exit()

# 获取样本数量和 Peak 数量
try:
    num_samples = root['region_motif'].shape[0]
    num_peaks = root['region_motif'].shape[1]
    print(f"样本数量: {num_samples}, Peak数量: {num_peaks}")
except KeyError:
    print("错误: Zarr文件中未找到'region_motif'数组。请确认Zarr文件结构。")
    exit()
except Exception as e:
    print(f"获取Zarr数组形状时出错: {e}")
    exit()

# 创建 exp_label 数组 (如果不存在或需要覆盖)
if 'exp_label' not in root or root['exp_label'].shape != (num_samples, num_peaks):
    print("创建或重置exp_label数组...")
    root['exp_label'] = np.zeros((num_samples, num_peaks), dtype=np.float32)

# 填充 exp_label
print("填充exp_label...")
sample_names = expression_df.columns.tolist()

# 检查表达数据的样本数量是否与Zarr文件一致
if len(sample_names) != num_samples:
    print(f"错误: 表达数据中的样本数量 ({len(sample_names)}) 与Zarr文件中的样本数量 ({num_samples}) 不匹配。请检查输入数据。")
    exit()

# 遍历样本
for sample_idx, sample_name in enumerate(sample_names):
    if sample_idx % 10 == 0:
        print(f"处理样本: {sample_name} ({sample_idx + 1}/{num_samples})")
    
    # 遍历 Peak
    for peak_idx, peak_id in enumerate(peak_ids):
        # 找到与当前 Peak 相关联的基因
        associated_genes = peak_to_genes.get(peak_id, [])
        
        peak_expression = 0.0
        if associated_genes:
            # 获取关联基因在该样本下的表达值并求和
            try:
                # 过滤掉在表达数据中不存在的基因
                existing_genes = [gene for gene in associated_genes if gene in expression_df.index]
                if existing_genes:
                    gene_expressions = expression_df.loc[existing_genes, sample_name]
                    # 确保取出的表达值是数值类型，并处理可能的NaN
                    gene_expressions = pd.to_numeric(gene_expressions, errors='coerce').fillna(0).sum()
                    peak_expression = gene_expressions
            except Exception as e:
                print(f"处理样本 {sample_name}, Peak {peak_id} 的基因表达时出错: {e}")
                pass
        
        # 将计算得到的表达值赋给 exp_label
        root['exp_label'][sample_idx, peak_idx] = peak_expression

print("exp_label 填充完成。")
print(f"Zarr文件 {ZARR_FILE} 已更新。")

# 添加检查和输出
print("\n检查更新后的Zarr文件...")
try:
    updated_root = zarr.open(ZARR_FILE, mode='r')
    if 'exp_label' in updated_root:
        exp_label_array = updated_root['exp_label']
        print(f"'exp_label' 数组形状: {exp_label_array.shape}")
        # 打印前几个样本的前几个Peak的exp_label
        print("前5个样本的前5个Peak的exp_label:")
        print(exp_label_array[:5, :5])
        # 打印一些非零值示例 (如果存在)
        non_zero_count = np.count_nonzero(exp_label_array)
        print(f"'exp_label' 数组中的非零值数量: {non_zero_count}")
        if non_zero_count > 0:
            # 获取非零值的索引
            non_zero_indices = np.transpose(np.nonzero(exp_label_array))
            print("前5个非零exp_label值及其位置 (样本索引, Peak索引):")
            for i in range(min(5, non_zero_count)):
                sample_idx, peak_idx = non_zero_indices[i]
                print(f"样本 {sample_idx}, Peak {peak_idx}: {exp_label_array[sample_idx, peak_idx]}")
    else:
        print("错误: 更新后的Zarr文件中未找到'exp_label'数组。")
except Exception as e:
    print(f"检查更新后的Zarr文件时出错: {e}") 