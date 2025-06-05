import pandas as pd
import numpy as np
import zarr
import os

def convert_csv_to_zarr(csv_path, zarr_path):
    """Converts a CSV file (with motif, accessibility, conditions, target) to a zarr archive."""
    print(f"Reading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    # 分离特征、条件和目标
    condition_cols = ['media_SC', 'media_YPD', 'temperature', 'drug_H2O2', 'drug_none', 'drug_concentration']
    feature_cols = [col for col in df.columns if col not in condition_cols + ['target', 'peak_id']]
    target_col = 'target'

    # 提取特征、条件和目标
    region_motif_data = df[feature_cols].values.reshape(-1, 1, len(feature_cols))  # Shape (num_rows, 1, num_features)
    condition_data = df[condition_cols].values.reshape(-1, 1, len(condition_cols))  # Shape (num_rows, 1, num_conditions)
    exp_label_data = df[target_col].values.reshape(-1, 1)  # Shape (num_rows, 1)

    # 创建zarr存档
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store)

    # 保存数据为zarr数组
    root.create_dataset('region_motif', data=region_motif_data, chunks=(100, 1, len(feature_cols)), overwrite=True)
    root.create_dataset('condition', data=condition_data, chunks=(100, 1, len(condition_cols)), overwrite=True)
    root.create_dataset('exp_label', data=exp_label_data, chunks=(100, 1), overwrite=True)

    print(f"Successfully converted CSV to Zarr at {zarr_path}")

if __name__ == '__main__':
    # 定义输入CSV路径和输出Zarr路径
    csv_file = '../input/gsm_example_with_condition.csv'
    zarr_archive = '../input/gsm_example_with_condition.zarr'

    # 确保输入文件存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_csv_file = os.path.join(script_dir, csv_file)

    if not os.path.exists(abs_csv_file):
        print(f"Error: Input CSV file not found at {abs_csv_file}")
    else:
        convert_csv_to_zarr(abs_csv_file, os.path.join(script_dir, zarr_archive)) 