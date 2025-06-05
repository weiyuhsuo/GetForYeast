import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import zarr
from torch.utils.data import Dataset, DataLoader
from get_model.model.yeast_model import YeastModel
import logging
from pathlib import Path

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
        return x, y

def setup_logging(output_dir: Path) -> logging.Logger:
    """设置日志"""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'training.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="get_model/config", config_name="yeast_training")
def main(config: DictConfig):
    """
    主训练函数
    参数:
        config: 包含训练配置的字典
    """
    # 设置随机种子
    torch.manual_seed(config.experiment.seed)
    
    # 设置输出目录
    output_dir = Path(config.logging.save_dir)
    logger = setup_logging(output_dir)
    logger.info(f"开始训练，配置：\n{config}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    dataset = YeastDataset(config.data.data_path)
    batch_size = config.training.batch_size
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    logger.info(f"数据加载完成，批次大小: {batch_size}")
    
    # 初始化模型
    model = YeastModel(
        input_dim=297,  # region_motif的特征维度
        hidden_dim=config.model.cfg.embed_dim,
        num_layers=config.model.cfg.num_layers,
        use_lora=config.training.use_lora,
        lora_rank=config.training.lora_rank,
        lora_alpha=config.training.lora_alpha
    ).to(device)
    logger.info("模型初始化完成")
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 设置损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    logger.info("开始训练...")
    model.train()
    for epoch in range(config.training.max_epochs):
        epoch_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 打印训练进度
            if batch_idx % config.training.log_every_n_steps == 0:
                logger.info(f"Epoch {epoch+1}/{config.training.max_epochs}, "
                          f"Batch {batch_idx}/{len(dataloader)}, "
                          f"Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{config.training.max_epochs}, "
                   f"Average Loss: {avg_epoch_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % config.training.save_every_n_epochs == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            logger.info(f"保存检查点到: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"训练完成，最终模型保存到: {final_model_path}")

if __name__ == "__main__":
    main() 