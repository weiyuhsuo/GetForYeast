import pandas as pd
import numpy as np

file1_path = 'data_WT.xlsx'
file2_path = 'WT_gene.xlsx'

output_file1_path = 'data_WT_filled.xlsx'
output_file2_path = 'WT_gene_filled.xlsx'

try:
    # 读取文件
    df1 = pd.read_excel(file1_path)
    df2 = pd.read_excel(file2_path)

    # 填充缺失值
    df1_filled = df1.fillna(0)
    df2_filled = df2.fillna(0)

    # 保存填充后的文件
    df1_filled.to_excel(output_file1_path, index=False)
    df2_filled.to_excel(output_file2_path, index=False)

    print(f"文件 {file1_path} 的缺失值已填充并保存到 {output_file1_path}")
    print(f"文件 {file2_path} 的缺失值已填充并保存到 {output_file2_path}")

except FileNotFoundError:
    print("错误：文件未找到。请检查文件路径是否正确。")
except Exception as e:
    print("处理文件时发生错误：", e) 