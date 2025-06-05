import pandas as pd

data_wt_path = 'mapping/data_WT_filled.xlsx'
matrix_path = 'mapping/combined_matrix.csv'

try:
    df_conditions = pd.read_excel(data_wt_path)
    print("\n实验条件文件 (data_WT_filled.xlsx) 的基本信息：")
    print("="*50)
    print(f"数据形状: {df_conditions.shape}")
    print("\n前几行数据：")
    print(df_conditions.head())
    print("\n列信息和数据类型：")
    print(df_conditions.info())

except Exception as e:
    print("读取实验条件文件时出错：", e)

try:
    # 只读取前几行来推断结构
    df_matrix_head = pd.read_csv(matrix_path, nrows=5)
    print("\nPeak*Motif 矩阵文件 (combined_matrix.csv) 的基本信息 (前5行)：")
    print("="*50)
    print(f"推断数据列数 (Motifs数量): {df_matrix_head.shape[1]}")
    print("\n前几行数据：")
    print(df_matrix_head.head())
    print("\n列名 (前几列)：")
    print(df_matrix_head.columns.tolist()[:10]) # 显示前10个列名

except Exception as e:
    print("读取Peak*Motif矩阵文件时出错：", e) 