import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from get_model.model.yeast_model import create_model
from get_model.data.yeast_dataset import YeastDataset

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            # 将数据移动到设备
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device)
            
            # 前向传播
            predictions = model(batch_x)
            
            # 收集预测和目标
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # 合并所有批次的结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算评估指标
    metrics = {
        'mse': mean_squared_error(all_targets.flatten(), all_predictions.flatten()),
        'r2': r2_score(all_targets.flatten(), all_predictions.flatten())
    }
    
    return metrics, all_predictions, all_targets

def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_dir: Path
):
    """绘制预测结果"""
    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制散点图
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=targets.flatten(), y=predictions.flatten(), alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Values')
    plt.savefig(save_dir / 'predictions_scatter.png')
    plt.close()
    
    # 绘制残差图
    residuals = predictions.flatten() - targets.flatten()
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=targets.flatten(), y=residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs True Values')
    plt.savefig(save_dir / 'residuals_scatter.png')
    plt.close()

def main():
    # 设置日志
    logger = setup_logging()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model_path = "output/yeast_finetune/best_model.pth"
    model = create_model("get_model/config/yeast_training.yaml")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    logger.info(f"模型加载完成: {model_path}")
    
    # 加载数据
    test_dataset = YeastDataset("input/20250601_data/yeast_data_with_conditions_original_peaks.zarr")
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    logger.info("数据加载完成")
    
    # 评估模型
    metrics, predictions, targets = evaluate_model(model, test_loader, device)
    logger.info(f"评估指标: {metrics}")
    
    # 绘制结果
    plot_predictions(predictions, targets, Path("output/yeast_finetune/evaluation"))
    logger.info("评估结果已保存")

if __name__ == "__main__":
    main() 