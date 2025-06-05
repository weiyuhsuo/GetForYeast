import zarr
import numpy as np
import pandas as pd
import os
from numcodecs import get_codec, JSON

# 文件路径
MAPPING_FILE = "/home/rhyswei/Code/aiyeast/aiyeast-514/5_ver2association/gene_peak_mapping.csv" # 仅作参考，不用于确定peak数量
PEAK_MOTIF_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/mapping/peak_motif_matrix.csv"
CONDITION_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/20250601_data/data_WT_filled.csv"  # 修改为CSV文件
ZARR_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/20250601_data/yeast_data_with_conditions_original_peaks.zarr"

def create_zarr_file():
    print("开始创建Zarr文件 (使用原始 Peak 集合)...")
    
    # 1. 加载peak-motif矩阵
    print("加载peak-motif矩阵...")
    peak_motif_df = pd.read_csv(PEAK_MOTIF_FILE, index_col=0)
    print(f"peak-motif矩阵形状: {peak_motif_df.shape}")
    
    # 2. 获取原始 Peak IDs 和数量
    peak_ids = peak_motif_df.index.tolist()
    num_peaks = len(peak_ids)
    print(f"原始 Peak 数量: {num_peaks}")
    print(f"前5个原始 Peak ID: {peak_ids[:5]}")
    
    # 处理NaN值
    print("\n处理NaN值...")
    nan_count_before = peak_motif_df.isna().sum().sum()
    print(f"NaN值数量: {nan_count_before}")
    peak_motif_df = peak_motif_df.fillna(0)
    nan_count_after = peak_motif_df.isna().sum().sum()
    print(f"填充后NaN值数量: {nan_count_after}")
    
    # 3. 加载实验条件数据
    print("\n加载实验条件数据...")
    condition_df = pd.read_csv(CONDITION_FILE)
    print(f"条件数据形状: {condition_df.shape}")
    
    # 获取样本数量
    num_samples = len(condition_df)
    print(f"样本数量: {num_samples}")
    
    # 4. 处理并添加condition特征
    print("\n添加condition特征...")
    # 选择所有需要的条件列
    condition_cols = [
        '培养基', '碳源', '氮源', 
        '预培养时间', '预培养温度', '预培养终点',
        '药物', '浓度', 
        '加药培养温度', '加药培养时间', '加药培养终点'
    ]
    
    # 提取选定的条件列
    selected_condition_df = condition_df[condition_cols].copy()
    print("原始条件列数据示例:")
    print(selected_condition_df.head())

    # 处理类别型变量（One-Hot编码）
    categorical_cols = ['培养基', '碳源', '氮源', '药物']
    # 检查每个类别列的唯一值
    for col in categorical_cols:
        unique_values = selected_condition_df[col].unique()
        print(f"\n{col} 的唯一值: {unique_values}")
    
    # 对每个类别列单独进行One-Hot编码
    cond_categorical_list = []
    for col in categorical_cols:
        if selected_condition_df[col].nunique() > 1:  # 只有当列有多个唯一值时才进行编码
            dummies = pd.get_dummies(selected_condition_df[col], prefix=col)
            cond_categorical_list.append(dummies)
        else:  # 如果列只有一个值，创建一个全1的列
            value = selected_condition_df[col].iloc[0]
            dummies = pd.DataFrame({f"{col}_{value}": 1}, index=selected_condition_df.index)
            cond_categorical_list.append(dummies)
    
    cond_categorical = pd.concat(cond_categorical_list, axis=1)
    print("\n类别型变量One-Hot编码后的列:")
    print(cond_categorical.columns.tolist())

    # 处理数值型变量
    numerical_cols = ['预培养温度', '预培养终点', '浓度', '加药培养温度', '加药培养终点']
    cond_numerical = selected_condition_df[numerical_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    print("\n数值型变量示例:")
    print(cond_numerical.head())
    
    # 处理时间字符串（去掉h转float）
    time_cols = ['预培养时间', '加药培养时间']
    cond_time = selected_condition_df[time_cols].astype(str).replace('h', '', regex=False).apply(pd.to_numeric, errors='coerce').fillna(0)
    print("\n时间变量处理示例:")
    print(cond_time.head())

    # 拼接所有condition特征
    cond_features_df = pd.concat([cond_categorical, cond_numerical, cond_time], axis=1)
    cond_features = cond_features_df.values
    
    print(f"\n最终Condition特征shape: {cond_features.shape}")
    print(f"最终Condition特征列: {list(cond_features_df.columns)}")
    
    num_condition_features = cond_features.shape[1]
    num_motifs = peak_motif_df.shape[1]

    # 5. 创建Zarr文件
    print(f"\n创建Zarr文件: {ZARR_FILE}")
    root = zarr.open(ZARR_FILE, mode='w')
    
    # 6. 创建peak_ids数组
    print("创建peak_ids数组...")
    root.create_dataset('peak_ids', data=np.array(peak_ids, dtype='object'), object_codec=JSON())
    
    # 7. 创建region_motif数组
    print("创建region_motif数组...")
    
    # 将peak-motif矩阵转换为3D数组 (samples, peaks, motifs)
    region_motif_base = peak_motif_df.values

    # 创建一个空的3D数组来存放整合后的数据
    region_motif_with_condition = np.zeros((num_samples, num_peaks, num_motifs + num_condition_features), dtype=np.float32)
    
    # 对每个样本，复制motif特征，并添加condition特征
    for i in range(num_samples):
        region_motif_with_condition[i, :, :num_motifs] = region_motif_base  # 复制所有peak的motif特征
        for j in range(num_condition_features):
            region_motif_with_condition[i, :, num_motifs + j] = cond_features[i, j] # 添加该样本的condition特征

    root.create_dataset('region_motif', data=region_motif_with_condition, chunks=(1, num_peaks, num_motifs + num_condition_features))
    
    # 8. 创建exp_label数组
    print("创建exp_label数组...")
    exp_label = np.zeros((num_samples, num_peaks), dtype=np.float32)
    root.create_dataset('exp_label', data=exp_label, chunks=(1, num_peaks))
    
    print("Zarr文件创建完成！")
    
    # 9. 验证创建的文件
    print("\n验证Zarr文件结构...")
    print("Zarr文件中的数组:")
    for key in root.keys():
        print(f"- {key}: {root[key].shape}")
    
    # 10. 输出一些统计信息
    print("\n=== 数据统计 ===")
    print(f"Peak数量 (原始): {num_peaks}")
    print(f"样本数量: {num_samples}")
    print(f"Motif数量: {num_motifs}")
    print(f"Condition特征数量: {num_condition_features}")
    print(f"region_motif形状: {region_motif_with_condition.shape}")
    print(f"region_motif非零值数量: {np.count_nonzero(region_motif_with_condition)}")
    print(f"region_motif值范围: [{region_motif_with_condition.min()}, {region_motif_with_condition.max()}]")
    
    # 11. 输出一些示例数据
    print("\n=== 示例数据 ===")
    print("前5个原始 Peak 的 Motif 值（前5个Motif）:")
    print(peak_motif_df.iloc[:5, :5])
    print("\n前5个样本的 Condition 特征:")
    print(cond_features_df.head())

if __name__ == "__main__":
    create_zarr_file() 