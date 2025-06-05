import zarr
import numpy as np

ZARR_FILE = "/home/rhyswei/Code/aiyeast/get_model/input/20250601_data/yeast_data_with_conditions_original_peaks.zarr"

def check_peak_ids():
    print(f"正在检查 Zarr 文件: {ZARR_FILE}")
    try:
        root = zarr.open(ZARR_FILE, mode='r')
        if 'peak_ids' in root:
            peak_ids = root['peak_ids'][:]
            print(f"Zarr 文件中 Peak ID 数量: {len(peak_ids)}")
            if len(peak_ids) > 0:
                print(f"前10个 Peak ID: {peak_ids[:10].tolist()}")
            else:
                print("Peak ID 数据集为空。")
        else:
            print("错误: Zarr 文件中未找到 'peak_ids' 数据集。")
            
    except FileNotFoundError:
        print(f"错误：未找到 Zarr 文件 {ZARR_FILE}")
    except Exception as e:
        print(f"读取 Zarr 文件时发生错误: {e}")

if __name__ == "__main__":
    check_peak_ids() 