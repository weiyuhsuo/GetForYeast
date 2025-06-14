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
from torch.amp import autocast, GradScaler
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from scipy.stats import pearsonr, linregress

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

logger = logging.getLogger(__name__)

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

def setup_logging(config, output_dir):
    """设置日志和wandb"""
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.project_name,
            name=config.logging.run_name,
            config=config
        )
    # 只设置console输出，不再生成train.log文件
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

@hydra.main(version_base=None, config_path="get_model/config", config_name="yeast_training")
def main(config: DictConfig):
    """
    主训练函数
    参数:
        config: 包含训练配置的字典
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config.logging.save_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(config, output_dir)
    
    # 设置随机种子
    torch.manual_seed(config.experiment.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
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
    logger.info(f"训练集样本数: {len(train_dataset)}")
    
    # 训练前可视化标签分布
    all_labels = []
    for i in range(len(train_dataset)):
        _, y = train_dataset[i]
        all_labels.extend(y.flatten().tolist())
    all_labels = np.array(all_labels)
    plt.figure()
    plt.hist(all_labels, bins=100, alpha=0.7)
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'label_distribution.png')
    plt.close()
    
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
    scaler = GradScaler('cuda')
    
    # tensorboard writer定义，必须在训练循环前
    if config.logging.tensorboard:
        writer = SummaryWriter(log_dir=output_dir)
    else:
        writer = None
    
    # 训练循环
    logger.info("开始训练...")
    best_loss = float('inf')
    patience_counter = 0
    train_loss_list = []
    lr_list = []
    train_start_time = time.time()
    
    def evaluate_and_log(preds, targets, output_dir, tag, md_path=None, log_time=None, filter_zero=False):
        # 评估并画图，tag为before/after
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr, linregress
        import numpy as np
        print(f"[DEBUG] evaluate_and_log: len(preds)={len(preds)}, len(targets)={len(targets)}")
        assert len(preds) == len(targets), f"[ERROR] preds和targets长度不一致: preds={len(preds)}, targets={len(targets)}"
        if filter_zero:
            mask = targets != 0
            preds = preds[mask]
            targets = targets[mask]
            tag = tag + '_filtered'
        plt.figure()
        idx = np.random.choice(len(targets), min(200, len(targets)), replace=False)
        plt.scatter(targets[idx], preds[idx], alpha=0.5)
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.title(f'Pred vs True ({tag})')
        fig_path = output_dir / f'pred_vs_true_{tag}.png'
        plt.savefig(fig_path)
        plt.close()
        # 计算指标
        r, p = pearsonr(targets, preds)
        slope, intercept, r_value, p_value, std_err = linregress(targets, preds)
        n = len(targets)
        # 写入log
        if md_path is not None:
            with open(md_path, 'a', encoding='utf-8') as f:
                if log_time:
                    f.write(f"\n## {tag} 评估时间：{log_time}\n")
                else:
                    f.write(f"\n## {tag} 评估\n")
                f.write(f"- Pearson r: {r:.4f}\n")
                f.write(f"- P值: {p:.2e}\n")
                f.write(f"- Slope: {slope:.4f}\n")
                f.write(f"- 样本数: {n}\n")
                f.write(f"- 预测均值: {np.mean(preds):.4f}\n")
                f.write(f"- 预测方差: {np.std(preds):.4f}\n")
                f.write(f"- 真实均值: {np.mean(targets):.4f}\n")
                f.write(f"- 真实方差: {np.std(targets):.4f}\n")
                f.write(f"- 结果可视化: {fig_path}\n")

    # ========== 训练前评估 ==========
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            # flatten后再extend，确保一一对应
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_targets.extend(batch_y.detach().cpu().numpy().flatten())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    print(f"[DEBUG] 训练前评估: len(all_preds)={len(all_preds)}, len(all_targets)={len(all_targets)}")
    md_path = output_dir / 'train_log.md'
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    evaluate_and_log(all_preds, all_targets, output_dir, tag='before', md_path=md_path, log_time=now)

    # ========== 训练过程 ==========
    for epoch in tqdm(range(80), desc='Epoch'):
        model.train()
        total_loss = 0
        batch_lr = []
        batch_losses = []
        all_preds = []
        all_targets = []
        
        for batch_idx, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Batch {epoch+1}'):
            # 将数据移动到设备
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device)
            
            # 使用混合精度训练
            with autocast('cuda'):
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
            
            # 记录loss和lr
            batch_losses.append(loss.item())
            batch_lr.append(scheduler.get_last_lr()[0])
            # 记录预测和真实值
            with torch.no_grad():
                pred = outputs.detach().cpu().numpy().flatten()
                target = batch_y.detach().cpu().numpy().flatten()
                all_preds.extend(pred.tolist())
                all_targets.extend(target.tolist())
            
            # 打印训练进度
            if batch_idx % config.training.log_every_n_steps == 0:
                logger.info(f'Epoch {epoch+1}/{80}, '
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
        train_loss_list.append(avg_loss)
        lr_list.append(np.mean(batch_lr))
        # TensorBoard记录
        if config.logging.tensorboard and writer is not None:
            writer.add_scalar('train/avg_loss', avg_loss, epoch)
            writer.add_scalar('train/lr', np.mean(batch_lr), epoch)
            # 参数分布
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.detach().cpu().numpy(), epoch)
        # 保存loss/lr曲线图片（每次覆盖）
        fig, ax1 = plt.subplots()
        ax1.plot(range(1, len(train_loss_list)+1), train_loss_list, label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(lr_list)+1), lr_list, color='orange', label='LR')
        ax2.set_ylabel('Learning Rate')
        fig.legend()
        plt.title('Loss & LR Curve')
        plt.savefig(output_dir / 'loss_lr_curve.png')
        plt.close(fig)
        # 保存预测vs真实值散点图（每次覆盖）
        idx = np.random.choice(len(all_preds), min(200, len(all_preds)), replace=False)
        fig2 = plt.figure()
        plt.scatter(np.array(all_targets)[idx], np.array(all_preds)[idx], alpha=0.5)
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.title(f'Pred vs True (epoch {epoch+1})')
        plt.savefig(output_dir / 'pred_vs_true.png')
        plt.close(fig2)
        # TensorBoard
        if config.logging.tensorboard and writer is not None:
            import io
            buf = io.BytesIO()
            fig2.savefig(buf, format='png')
            buf.seek(0)
            import PIL.Image
            image = PIL.Image.open(buf)
            image = np.array(image)
            writer.add_image('pred_vs_true', image.transpose(2,0,1), epoch)
            buf.close()
        
        # 早停检查
        if config.training.early_stopping_patience is not None:
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
        else:
            # 只保存最佳模型，不做早停判断
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = output_dir / 'best_model.pth'
                torch.save(model.state_dict(), best_model_path)
                logger.info(f'Best model saved to {best_model_path}')
        
        # epoch结束后输出统计信息
        logger.info(f'Epoch {epoch+1}: pred mean={np.mean(all_preds):.4f}, std={np.std(all_preds):.4f}, min={np.min(all_preds):.4f}, max={np.max(all_preds):.4f}; true mean={np.mean(all_targets):.4f}, std={np.std(all_targets):.4f}, min={np.min(all_targets):.4f}, max={np.max(all_targets):.4f}')
        # 保存部分预测和真实值到csv（每次覆盖）
        df_pred = pd.DataFrame({'pred': all_preds[:200], 'true': all_targets[:200]})
        df_pred.to_csv(output_dir / 'pred_true.csv', index=False)
    
    # ========== 训练后评估 ==========
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    evaluate_and_log(all_preds, all_targets, output_dir, tag='after', md_path=md_path, log_time=now)
    evaluate_and_log(all_preds, all_targets, output_dir, tag='after', md_path=md_path, log_time=now, filter_zero=True)

    # 训练结束后保存loss/lr为csv（每次覆盖）
    pd.DataFrame({'epoch': np.arange(1, len(train_loss_list)+1), 'loss': train_loss_list, 'lr': lr_list}).to_csv(output_dir / 'loss_lr.csv', index=False)
    
    # 训练总耗时
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    train_time_str = f"{train_time/60:.1f} 分钟" if train_time > 60 else f"{train_time:.1f} 秒"

    # 训练结束后自动记录到 Markdown 日志（输出到output_dir）
    md_path = output_dir / 'train_log.md'
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(md_path, 'a', encoding='utf-8') as f:
        f.write(f"\n## 实验时间：{now}\n")
        f.write(f"### 主要修改\n- （请在此补充本次实验的主要修改点）\n")
        f.write(f"\n### 训练参数\n")
        for k, v in dict(config).items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n### 训练总耗时\n- {train_time_str}\n")
        f.write(f"\n### 结果可视化\n")
        f.write(f"- Loss/LR曲线: {output_dir / 'loss_lr_curve.png'}\n")
        f.write(f"- 预测vs真实值散点图: {output_dir / 'pred_vs_true_after.png'}\n")
        f.write(f"\n### 备注\n- （可补充实验现象、问题、TODO等）\n")
    
    # 关闭wandb和tensorboard
    if config.logging.use_wandb:
        wandb.finish()
    if config.logging.tensorboard:
        writer.close()

if __name__ == "__main__":
    main() 