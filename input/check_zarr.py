import zarr
import os
import numpy as np

zarr_path = 'aiyeast/get_model/input/mapped_data_with_conditions/yeast_data_with_conditions.zarr'

print(f"正在打开 Zarr 文件: {zarr_path}")

if not os.path.exists(zarr_path):
    print(f"错误：Zarr 路径不存在: {zarr_path}")
else:
    try:
        root = zarr.open(zarr_path, mode='r')
        print("Zarr 文件打开成功。")
        print("\n数据集信息：")
        print("="*20)
        
        # 遍历所有数据集并打印信息
        for name in root.keys():
            dataset = root[name]
            print(f"数据集名称: {name}")
            print(f"  形状: {dataset.shape}")
            print(f"  数据类型: {dataset.dtype}")
            
            if name == 'exp_label':
                # 读取整个 exp_label 数据集
                exp_label_data = dataset[:]
                
                # 找到至少在一个GSM样本下有非零表达值的Peak的索引
                # 检查每个Peak在所有GSM下的最大值，如果大于0，则认为是非零Peak
                peaks_with_expression_indices = np.where(np.max(exp_label_data[:, :, 0], axis=0) > 0)[0]
                
                num_peaks_with_expression = len(peaks_with_expression_indices)
                total_peaks = dataset.shape[1]
                percentage_with_expression = (num_peaks_with_expression / total_peaks) * 100 if total_peaks > 0 else 0
                
                print(f"  在至少一个GSM样本下有非零表达值的Peak数量: {num_peaks_with_expression} / {total_peaks} ({percentage_with_expression:.2f}%) ")

                # 打印前几个有表达值的Peak的数据示例
                num_examples_to_show = min(5, num_peaks_with_expression)
                if num_examples_to_show > 0:
                    print(f"  前 {num_examples_to_show} 个有表达值的Peak在所有GSM样本下的数据示例:")
                    # 获取这些Peak的全局索引
                    example_peak_indices = peaks_with_expression_indices[:num_examples_to_show]
                    # 打印这些Peak在所有GSM下的 exp_label
                    print(exp_label_data[:, example_peak_indices, :])

            elif dataset.ndim >= 2:
                 # 尝试读取并打印第一个 GSM 的前几个 Peak 的数据 (对于其他二维数据集)
                 try:
                      print(f"  前1个GSM，前2个Peak的数据示例:\n{dataset[0, :2]}")
                 except Exception as e:
                      print(f"  无法读取数据示例: {e}")
            elif dataset.ndim == 1:
                 # 尝试读取并打印前几个元素 (对于一维数据集)
                 try:
                      print(f"  前5个元素示例:\n{dataset[:5]}")
                 except Exception as e:
                      print(f"  无法读取数据示例: {e}")

            print("-"*20)

    except Exception as e:
        print(f"打开或读取 Zarr 文件时发生错误: {e}")

print("检查完成。") 