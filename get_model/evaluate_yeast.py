import torch
import torch.nn as nn
import yaml
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import json

from model.yeast_model import create_model
from dataset.yeast_dataset import create_dataloader

def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'evaluate.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_features = []
    
    with torch.no_grad():
        for features, targets in tqdm(test_loader, desc='Evaluating'):
            # 将数据移到设备
            features = {k: v.to(device) for k, v in features.items()}
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # 更新统计信息
            total_loss += loss.item()
            
            # 收集预测结果
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 收集特征
            feature_dict = {k: v.cpu().numpy() for k, v in features.items()}
            all_features.append(pd.DataFrame(feature_dict))
    
    # 计算评估指标
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    metrics = {
        'mse': mean_squared_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'r2': r2_score(targets, predictions)
    }
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'target': targets,
        'prediction': predictions
    })
    
    # 合并特征
    features_df = pd.concat(all_features, ignore_index=True)
    results_df = pd.concat([results_df, features_df], axis=1)
    
    return total_loss / len(test_loader), metrics, results_df

def plot_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
):
    """绘制评估结果"""
    # 创建图表目录
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # 预测值vs真实值散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=results_df, x='target', y='prediction')
    plt.plot([results_df['target'].min(), results_df['target'].max()],
             [results_df['target'].min(), results_df['target'].max()],
             'r--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.savefig(plots_dir / 'predictions_vs_true.png')
    plt.close()
    
    # 残差图
    plt.figure(figsize=(10, 8))
    results_df['residual'] = results_df['prediction'] - results_df['target']
    sns.scatterplot(data=results_df, x='target', y='residual')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(plots_dir / 'residual_plot.png')
    plt.close()
    
    # 特征重要性分析（可选）
    if 'feature_importance' in results_df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=results_df, x='feature', y='importance')
        plt.xticks(rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_importance.png')
        plt.close()

def main(
    config_path: str,
    data_path: str,
    model_path: str,
    output_dir: str,
    device: Optional[str] = None
):
    """主函数"""
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir / 'logs')
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    test_loader = create_dataloader(
        data_path=data_path,
        config_path=config_path,
        split='test',
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # 创建模型
    model = create_model(config_path).to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if config['training'].get('use_lora', False):
        # 如果有LoRA参数，加载LoRA权重
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("加载了LoRA微调权重")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("加载了普通权重")
    
    # 创建损失函数
    criterion = nn.MSELoss()
    
    # 评估模型
    test_loss, metrics, results_df = evaluate(model, test_loader, criterion, device, logger)
    
    # 记录评估结果
    logger.info(f"测试损失: {test_loss:.4f}")
    logger.info("评估指标:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # 保存评估结果
    results_df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 绘制结果
    plot_results(results_df, output_dir, logger)
    
    logger.info("评估完成")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--device', type=str, help='设备（cuda/cpu）')
    args = parser.parse_args()
    
    main(args.config, args.data, args.model, args.output, args.device) 