import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import zarr
from torch.utils.data import Dataset, DataLoader
from get_model.model.yeast_model import YeastModel, create_model
import logging
from pathlib import Path
from typing import Dict, Any
import os
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import wandb
from torch.utils.tensorboard import SummaryWriter

class YeastDataset(Dataset):
    """
    酵母数据集类，用于加载Zarr格式的数据
    数据包含：
    - region_motif: 形状为 (66, 2268, 297) 的特征矩阵
        - 0-282: motif特征（已归一化）
        - 283-296: condition特征（部分归一化）
    - exp_label: 形状为 (66, 2268) 的表达值矩阵
    """
    def __init__(self, zarr_path):
        self.root = zarr.open(zarr_path, mode='r')
        self.region_motif = self.root['region_motif']  # 特征矩阵
        self.exp_label = self.root['exp_label']       # 表达值标签

    def __len__(self):
        return self.region_motif.shape[0]  # 返回样本数量

    def __getitem__(self, idx):
        # 转换为PyTorch张量
        x = torch.tensor(self.region_motif[idx], dtype=torch.float32)
        y = torch.tensor(self.exp_label[idx], dtype=torch.float32)
        
        # 提取特征
        features = {
            'media_type': x[:, 283].long(),  # 培养基类型
            'temperature': x[:, 284:285],    # 温度
            'pre_culture_time': x[:, 285:286],  # 预培养时间
            'pre_culture_od600': x[:, 286:287],  # 预培养OD600
            'drug_culture_time': x[:, 287:288],  # 药物培养时间
            'drug_name': x[:, 288].long(),   # 药物名称
            'concentration': x[:, 289:290],  # 药物浓度
            'carbon_source': x[:, 290].long(),  # 碳源
            'nitrogen_source': x[:, 291].long(),  # 氮源
            'motif_features': x[:, :283]  # motif特征
        }
        
        return features, y

def setup_logging(config):
    """设置日志和wandb"""
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.project_name,
            name=config.logging.run_name,
            config=config
        )
    
    if config.logging.tensorboard:
        writer = SummaryWriter(log_dir=config.logging.save_dir)
    else:
        writer = None
    
    return writer

@hydra.main(version_base=None, config_path="get_model/config", config_name="yeast_training")
def main(config: DictConfig):
    """
    主训练函数
    参数:
        config: 包含训练配置的字典
    """
    # 设置随机种子
    torch.manual_seed(config.experiment.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 设置日志
    writer = setup_logging(config)
    
    # 加载数据
    train_dataset = YeastDataset(config.data.data_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    logger.info(f"数据加载完成，批次大小: {config.training.batch_size}")
    
    # 初始化模型
    model = create_model("get_model/config/yeast_training.yaml")
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        logger.info(f"使用 {torch.cuda.device_count()} 个GPU训练")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # 初始化优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 初始化学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler.T_max,
        T_mult=2,
        eta_min=config.scheduler.eta_min
    )
    
    # 初始化混合精度训练
    scaler = GradScaler()
    
    # 创建输出目录
    output_dir = Path(config.logging.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    logger.info("开始训练...")
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.max_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # 将数据移动到设备
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device)
            
            # 使用混合精度训练
            with autocast():
                # 前向传播
                outputs = model(batch_x)
                # 计算损失
                loss = model.compute_loss(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if config.training.clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.clip_grad
                )
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率
            scheduler.step()
            
            total_loss += loss.item()
            
            # 打印训练进度
            if batch_idx % config.training.log_every_n_steps == 0:
                logger.info(f'Epoch {epoch+1}/{config.training.max_epochs}, '
                          f'Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'LR: {scheduler.get_last_lr()[0]:.6f}')
                
                if config.logging.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + 1,
                        'train/batch': batch_idx
                    })
                
                if config.logging.tensorboard:
                    writer.add_scalar('train/loss', loss.item(),
                                    epoch * len(train_loader) + batch_idx)
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0],
                                    epoch * len(train_loader) + batch_idx)
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{config.training.max_epochs}, '
                   f'Average Loss: {avg_loss:.4f}')
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_path = output_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'Best model saved to {best_model_path}')
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # 定期保存检查点
        if (epoch + 1) % config.training.save_every_n_epochs == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved to {checkpoint_path}')
    
    # 保存最终模型
    final_model_path = output_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'Final model saved to {final_model_path}')
    
    # 关闭wandb和tensorboard
    if config.logging.use_wandb:
        wandb.finish()
    if config.logging.tensorboard:
        writer.close()

if __name__ == "__main__":
    main() 