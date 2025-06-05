import pandas as pd
import numpy as np
import zarr
import os

data_wt_path = 'aiyeast/get_model/input/mapping/data_WT_filled.xlsx'
matrix_path = 'aiyeast/get_model/input/mapping/peak_motif_matrix.csv'
zarr_output_path = 'aiyeast/get_model/input/mapped_data_with_conditions/yeast_data_with_conditions.zarr' # Zarr文件输出路径

# 创建输出目录（如果不存在）
output_dir = os.path.dirname(zarr_output_path)
os.makedirs(output_dir, exist_ok=True)

try:
    # 1. 加载数据
    df_conditions = pd.read_excel(data_wt_path)
    df_matrix = pd.read_csv(matrix_path) # 可能需要优化读取大文件的方式

    # 2. 选择和处理实验条件列
    # 排除的列
    exclude_condition_columns = ['Unnamed: 0', 'GSE', 'GSM', '菌株', '基因组文件名', '基因表达文件名', 'Mark']
    # 确认实际存在的列
    condition_columns_to_use = [col for col in df_conditions.columns if col not in exclude_condition_columns]
    
    if not condition_columns_to_use:
        print("警告：没有找到符合条件的实验条件列用于整合。")
        condition_features = np.zeros((df_conditions.shape[0], 0)) # 创建一个空的条件特征数组
        num_conditions = 0
    else:
        df_conditions_subset = df_conditions[condition_columns_to_use]

        # 处理分类条件 (独热编码)
        # 推断分类列（非数值和非布尔类型）
        categorical_cols = df_conditions_subset.select_dtypes(include=['object', 'category']).columns.tolist()
        df_conditions_processed = pd.get_dummies(df_conditions_subset, columns=categorical_cols, dummy_na=False) # dummy_na=False 不为NaN创建列

        # 处理数值条件 (标准化，这里跳过标准化步骤，直接使用填充0后的值)
        # 注意：根据需要可以添加StandardScaler或MinMaxScaler

        # 创建条件特征矩阵 (转换为 numpy 数组)
        condition_features = df_conditions_processed.values # 形状 (66, Total_Condition_Features)
        num_conditions = condition_features.shape[1]
        print(f"将使用的实验条件列 ({num_conditions} 列): {df_conditions_processed.columns.tolist()}")

    # 3. 准备 Peak*Motif 数据
    # 排除 'peak_id' 列，保留其他列作为基础特征
    if 'peak_id' in df_matrix.columns:
        peak_ids = df_matrix['peak_id'].values.astype(str) # 保存peak_id并转换为字符串
        peak_motif_features_base = df_matrix.drop(columns=['peak_id']).values # 形状 (Peaks, Motif_Features + Accessibility)
    else:
        peak_ids = np.arange(df_matrix.shape[0]).astype(str) # 如果没有peak_id，用索引代替并转换为字符串
        peak_motif_features_base = df_matrix.values # 形状 (Peaks, Motif_Features + Accessibility)
        
    num_peaks = peak_motif_features_base.shape[0]
    num_motif_features_base = peak_motif_features_base.shape[1]
    print(f"Peak*Motif 矩阵形状: ({num_peaks}, {num_motif_features_base})")

    # 4. 整合特征并构建三维数组
    # 新的特征总数 = Peak*Motif基础特征数 + 条件特征数
    total_features_per_peak = num_motif_features_base + num_conditions
    
    # 创建一个空的 numpy 数组用于存放所有样本整合后的特征
    all_gsm_features = np.zeros((66, num_peaks, total_features_per_peak), dtype=np.float32)

    for i in range(66):
        # 广播条件向量到 Peak 维度
        if num_conditions > 0:
             condition_matrix_i = np.tile(condition_features[i, :], (num_peaks, 1)) # 形状 (Peaks, Total_Condition_Features)
             # 拼接 Peak*Motif 特征和条件特征
             all_gsm_features[i, :, :] = np.concatenate((peak_motif_features_base, condition_matrix_i), axis=1)
        else:
             # 如果没有条件特征，只使用 Peak*Motif 基础特征
             all_gsm_features[i, :, :] = peak_motif_features_base

    # 5. 创建 exp_label 数组 (暂时创建空的，形状与 region_motif 匹配，但维度可能需要调整)
    # 根据 GETRegionFinetune 的 before_loss 方法，exp_label 期望形状是 (Batch_size, Num_peaks, 1)
    # Zarr 文件存储时，每个样本对应一个 Batch_size=1
    exp_label_data = np.zeros((66, num_peaks, 1), dtype=np.float32) # 暂时用0填充，形状 (GSMs, Peaks, 1)
    print(f"创建空的 exp_label 数组，形状: {exp_label_data.shape}")

    # 6. 保存为 Zarr 文件
    print(f"正在保存数据到 Zarr 文件: {zarr_output_path}")
    root = zarr.open(zarr_output_path, mode='w')

    # 保存 region_motif 数组
    root.create_dataset('region_motif', data=all_gsm_features, chunks=(1, num_peaks, total_features_per_peak), dtype='float32')
    print("region_motif 数组保存完成。")

    # 保存 exp_label 数组
    root.create_dataset('exp_label', data=exp_label_data, chunks=(1, num_peaks, 1), dtype='float32')
    print("exp_label 数组保存完成。")
    
    # 保存 peak_ids 作为元数据，指定 object_codec
    root.create_dataset('peak_ids', data=peak_ids, dtype=peak_ids.dtype, object_codec=zarr.json)
    print("peak_ids 数组保存完成。")
    
    # 可以在这里添加一个空的 'condition' 数组，以兼容 YeastZarrDataset 的 __getitem__ 方法，尽管我们已经将条件融入 region_motif
    # root.create_dataset('condition', shape=(66, 0), dtype='float32') # 形状 (GSMs, 0)

    print("Zarr 文件创建成功！")

except FileNotFoundError:
    print("错误：文件未找到。请检查文件路径是否正确。")
except Exception as e:
    print("处理文件时发生错误：", e) 