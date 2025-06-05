import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import yaml
import os
import logging

logger = logging.getLogger(__name__)

class FeatureEncoder(nn.Module):
    """特征编码器模块"""
    def __init__(self, config: Dict):
        super().__init__()
        self.encoders = nn.ModuleDict()
        
        # 培养基类型编码器
        self.encoders['media_type'] = nn.Embedding(
            num_embeddings=config['media_type']['num_embeddings'],
            embedding_dim=config['media_type']['embedding_dim'],
            padding_idx=config['media_type'].get('padding_idx')
        )
        
        # 温度编码器
        self.encoders['temperature'] = nn.Linear(
            in_features=config['temperature']['in_features'],
            out_features=config['temperature']['out_features']
        )
        
        # 预培养编码器
        self.encoders['pre_culture_time'] = nn.Linear(
            in_features=config['pre_culture']['time']['in_features'],
            out_features=config['pre_culture']['time']['out_features']
        )
        self.encoders['pre_culture_od600'] = nn.Linear(
            in_features=config['pre_culture']['od600']['in_features'],
            out_features=config['pre_culture']['od600']['out_features']
        )
        
        # 药物培养编码器
        self.encoders['drug_culture_time'] = nn.Linear(
            in_features=config['drug_culture']['time']['in_features'],
            out_features=config['drug_culture']['time']['out_features']
        )
        self.encoders['drug_name'] = nn.Embedding(
            num_embeddings=config['drug_culture']['drug_name']['num_embeddings'],
            embedding_dim=config['drug_culture']['drug_name']['embedding_dim'],
            padding_idx=config['drug_culture']['drug_name']['padding_idx']
        )
        self.encoders['concentration'] = nn.Linear(
            in_features=config['drug_culture']['concentration']['in_features'],
            out_features=config['drug_culture']['concentration']['out_features']
        )
        
        # 碳源编码器
        self.encoders['carbon_source'] = nn.Embedding(
            num_embeddings=config['carbon_source']['num_embeddings'],
            embedding_dim=config['carbon_source']['embedding_dim'],
            padding_idx=config['carbon_source']['padding_idx']
        )
        
        # 氮源编码器
        self.encoders['nitrogen_source'] = nn.Embedding(
            num_embeddings=config['nitrogen_source']['num_embeddings'],
            embedding_dim=config['nitrogen_source']['embedding_dim'],
            padding_idx=config['nitrogen_source']['padding_idx']
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded_features = {}
        for name, encoder in self.encoders.items():
            if name in features:
                encoded_features[name] = encoder(features[name])
        return encoded_features

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(in_features, out_features)
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B)
            self.scaling = alpha / r
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 1.0

    def forward(self, x):
        out = self.linear(x)
        if self.r > 0:
            out = out + self.dropout((x @ self.lora_A.t()) @ self.lora_B.t()) * self.scaling
        return out

class FeatureFusion(nn.Module):
    """特征融合模块，支持LoRA"""
    def __init__(self, config: Dict, use_lora=False, lora_rank=4, lora_alpha=16):
        super().__init__()
        self.fusion_type = config['type']
        self.fusion_dim = config['fusion_dim']
        self.hidden_dim = config['hidden_dim']
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        # 主线性层
        self.linear1 = nn.Linear(self.fusion_dim, self.hidden_dim)
        # LoRA分支
        if use_lora:
            self.lora1 = LoRALinear(self.fusion_dim, self.hidden_dim, r=lora_rank, alpha=lora_alpha, dropout=config['dropout'])
        else:
            self.lora1 = None
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(config['dropout'])
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        if use_lora:
            self.lora2 = LoRALinear(self.hidden_dim, self.hidden_dim, r=lora_rank, alpha=lora_alpha, dropout=config['dropout'])
        else:
            self.lora2 = None
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(config['dropout'])

    def forward(self, encoded_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        concat_features = torch.cat(list(encoded_features.values()), dim=-1)
        x = self.linear1(concat_features)
        if self.use_lora and self.lora1 is not None:
            x = x + self.lora1(concat_features)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x2 = self.linear2(x)
        if self.use_lora and self.lora2 is not None:
            x2 = x2 + self.lora2(x)
        x2 = self.norm2(x2)
        x2 = self.act2(x2)
        x2 = self.dropout2(x2)
        return x2

class YeastTransformer(nn.Module):
    """酵母Transformer模块，支持LoRA"""
    def __init__(self, config: Dict, use_lora=False, lora_rank=4, lora_alpha=16):
        super().__init__()
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation=config['activation'],
            layer_norm_eps=float(config['layer_norm_eps']),
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        # LoRA可选：可在主干的线性层插入LoRA分支（如有需要可进一步细化）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d]
        output = self.transformer(x)
        return output

class PredictionHead(nn.Module):
    """预测头模块"""
    def __init__(self, config: Dict):
        super().__init__()
        layers = []
        input_dim = config['hidden_dims'][0]
        
        # 构建MLP层
        for hidden_dim in config['hidden_dims'][1:]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout'])
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, config['output_dim']))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_regions, hidden_dim]
        x = self.mlp(x)  # [batch, num_regions, 1]
        return x.squeeze(-1)  # [batch, num_regions]

class YeastModel(nn.Module):
    def __init__(self, cfg: Dict, use_lora=False, lora_rank=4, lora_alpha=16):
        super().__init__()
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 特征编码器
        self.feature_encoder = FeatureEncoder(cfg['feature_encoders'])
        
        # 特征融合
        self.feature_fusion = FeatureFusion(
            cfg['feature_fusion'],
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )
        
        # Transformer
        self.transformer = YeastTransformer(
            cfg['transformer'],
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )
        
        # 预测头
        self.prediction_head = PredictionHead(cfg['prediction_head'])
        
        # 损失函数
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        # 加载预训练检查点（如果存在）
        if cfg.get('finetune', {}).get('pretrain_checkpoint', False):
            checkpoint_path = cfg['finetune'].get('checkpoint')
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"加载预训练检查点: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if cfg['finetune'].get('model_key') == 'state_dict':
                    self.load_state_dict(checkpoint, strict=cfg['finetune'].get('strict', False))
                else:
                    self.load_state_dict(checkpoint['model_state_dict'], strict=cfg['finetune'].get('strict', False))
                logger.info("预训练检查点加载完成")
            else:
                logger.warning(f"预训练检查点路径不存在: {checkpoint_path}")

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 特征编码
        encoded_features = self.feature_encoder(features)
        
        # 特征融合
        fused_features = self.feature_fusion(encoded_features)  # [batch, num_regions, hidden_dim]
        
        # Transformer处理
        transformed_features = self.transformer(fused_features)  # [batch, num_regions, hidden_dim]
        
        # 预测
        predictions = self.prediction_head(transformed_features)  # [batch, num_regions]
        
        return predictions

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 确保预测和目标的维度匹配
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)  # [batch, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)  # [batch, 1]
        
        # 计算损失
        loss = self.loss_fn(predictions, targets)
        return loss

def create_model(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 传入model下的model字段
    return YeastModel(
        cfg=config['model']['model'],
        use_lora=config['training'].get('use_lora', False),
        lora_rank=config['training'].get('lora_rank', 4),
        lora_alpha=config['training'].get('lora_alpha', 16)
    )

# YeastModel的cfg参数应为如下结构：
# cfg = {
#   'feature_encoders': {...},
#   'feature_fusion': {...},
#   'transformer': {...},
#   'prediction_head': {...},
#   ...
# } 