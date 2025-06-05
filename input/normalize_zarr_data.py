import zarr
import numpy as np

# Define the path to the Zarr store
zarr_path = '/home/rhyswei/Code/aiyeast/get_model/input/20250601_data/yeast_data_with_conditions_original_peaks.zarr'

try:
    # Open the Zarr store in read-write mode
    root = zarr.open(zarr_path, mode='a')

    # Access the region_motif dataset
    if 'region_motif' not in root:
        print(f"Error: 'region_motif' dataset not found in {zarr_path}")
    else:
        region_motif_data = root['region_motif']
        print(f"Loaded region_motif data with shape: {region_motif_data.shape}")
        num_samples, num_peaks, num_features = region_motif_data.shape
        num_motif_features = 283
        motif_data = region_motif_data[:, :, :num_motif_features]
        cond_data = region_motif_data[:, :, num_motif_features:]
        num_cond_features = cond_data.shape[2]
        print(f"motif特征数: {num_motif_features}, condition特征数: {num_cond_features}")

        # 归一化motif特征
        mean_motif = np.mean(motif_data, axis=(0, 1), keepdims=True)
        std_motif = np.std(motif_data, axis=(0, 1), keepdims=True)
        std_motif[std_motif == 0] = 1e-8
        normalized_motif = (motif_data - mean_motif) / std_motif
        print("motif特征已归一化")

        # 归一化condition特征
        normalized_cond = np.empty_like(cond_data)
        for i in range(num_cond_features):
            feat = cond_data[:, :, i]
            unique_vals = np.unique(feat)
            # 判断是否为one-hot（全是0/1或True/False）
            if np.all(np.isin(unique_vals, [0, 1])) or np.all(np.isin(unique_vals, [True, False])):
                normalized_cond[:, :, i] = feat  # 保持不变
                print(f"condition特征 {i+num_motif_features} (one-hot) 保持不变，唯一值: {unique_vals}")
            else:
                mean = np.mean(feat)
                std = np.std(feat)
                if std == 0:
                    normalized_cond[:, :, i] = feat - mean
                else:
                    normalized_cond[:, :, i] = (feat - mean) / std
                print(f"condition特征 {i+num_motif_features} (连续型) 已归一化，均值: {mean:.4f}，std: {std:.4f}")

        # 拼接回去
        normalized_region_motif = np.concatenate([normalized_motif, normalized_cond], axis=2)
        print(f"归一化后region_motif shape: {normalized_region_motif.shape}")
        if normalized_region_motif.shape != region_motif_data.shape:
            print(f"Error: Shape mismatch. 原始: {region_motif_data.shape}, 归一化: {normalized_region_motif.shape}")
        else:
            root['region_motif'][:] = normalized_region_motif
            print(f"region_motif归一化并写回Zarr文件: {zarr_path}")

except FileNotFoundError:
    print(f"Error: Zarr store not found at {zarr_path}")
except Exception as e:
    print(f"An error occurred: {e}")

print("Normalization script finished.") 