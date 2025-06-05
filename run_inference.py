import os
import pandas as pd
import torch
import logging
import numpy as np
from omegaconf import OmegaConf
from get_model.model.model import GETRegionFinetune

# 日志设置
log_dir = "output/log"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "inference.log"),
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# 输入输出路径
input_path = "input/combined_matrix.csv"
checkpoint_path = "checkpoints/fetal_pretrain_fetal_finetune/Astrocytes/checkpoint-best.pth"
output_path = "output/prediction_combined.csv"

# 读取输入
logging.info(f"读取输入矩阵: {input_path}")
df = pd.read_csv(input_path)

# 保存非数值列（用于后续结果匹配）
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
logging.info(f"非数值列: {list(non_numeric_cols)}")
logging.info(f"数值列: {list(numeric_cols)}")

# 只保留数值型列
df_numeric = df[numeric_cols]

# 确保数据类型为float32
logging.info("转换输入数据类型为float32")
input_data = df_numeric.astype(np.float32).values
logging.info(f"输入矩阵形状: {input_data.shape}")

# 加载模型配置
cfg = OmegaConf.load("get_model/config/model/GETRegionFinetune.yaml")

# 初始化模型
model = GETRegionFinetune(cfg.model.cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载checkpoint
logging.info(f"加载checkpoint: {checkpoint_path}")
try:
    # 尝试使用新的加载方式
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
except Exception as e:
    logging.warning(f"标准加载失败，尝试使用pickle加载: {str(e)}")
    import pickle
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)

# 提取模型权重
if isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

# 加载权重
load_result = model.load_state_dict(state_dict, strict=False)
logging.info(f"模型权重加载结果: {load_result}")
model.to(device)
model.eval()

# 推理
logging.info("开始推理...")
with torch.no_grad():
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(input_tensor)
    output_np = output.cpu().numpy().squeeze(0)

# 保存结果
result_df = pd.DataFrame(output_np)
# 如果有非数值列，将其添加回结果中
if len(non_numeric_cols) > 0:
    result_df = pd.concat([df[non_numeric_cols], result_df], axis=1)
result_df.to_csv(output_path, index=False)
logging.info(f"推理完成，结果已保存到: {output_path}")

print("推理完成，详细信息见 output/log/inference.log")