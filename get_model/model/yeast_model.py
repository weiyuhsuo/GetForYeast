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
            padding_idx=config['drug_culture']['drug_name'].get('padding_idx')
        )
        self.encoders['concentration'] = nn.Linear(
            in_features=config['drug_culture']['concentration']['in_features'],
            out_features=config['drug_culture']['concentration']['out_features']
        )
        
        # 碳源编码器
        self.encoders['carbon_source'] = nn.Embedding(
            num_embeddings=config['carbon_source']['num_embeddings'],
            embedding_dim=config['carbon_source']['embedding_dim'],
            padding_idx=config['carbon_source'].get('padding_idx')
        )
        
        # 氮源编码器
        self.encoders['nitrogen_source'] = nn.Embedding(
            num_embeddings=config['nitrogen_source']['num_embeddings'],
            embedding_dim=config['nitrogen_source']['embedding_dim'],
            padding_idx=config['nitrogen_source'].get('padding_idx')
        )
        
        # motif特征编码器
        self.encoders['motif_features'] = nn.Linear(
            in_features=config['motif_features']['in_features'],
            out_features=config['motif_features']['out_features']
        )

class FeatureFusion(nn.Module):
    """特征融合模块"""
    def __init__(self, config: Dict):
        super().__init__()
        self.fusion_type = config['type']
        self.fusion_dim = config['fusion_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # 特征融合层
        layers = []
        in_dim = self.fusion_dim
        for _ in range(self.num_layers):
            layers.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = self.hidden_dim
        
        self.fusion_net = nn.Sequential(*layers)

class YeastTransformer(nn.Module):
    """Transformer编码器模块"""
    def __init__(self, config: Dict):
        super().__init__()
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']
        self.activation = config['activation']
        self.layer_norm_eps = config['layer_norm_eps']
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_dim]
        Returns:
            输出张量，形状为 [batch_size, seq_len, hidden_dim]
        """
        return self.transformer(x)

class PredictionHead(nn.Module):
    """预测头模块"""
    def __init__(self, config: Dict):
        super().__init__()
        self.head_type = config['type']
        self.hidden_dims = config['hidden_dims']
        self.dropout = config['dropout']
        self.activation = config['activation']
        self.output_dim = config['output_dim']
        
        # 预测头网络
        layers = []
        in_dim = self.hidden_dims[0]
        for hidden_dim in self.hidden_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU() if self.activation == 'relu' else nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.pred_net = nn.Sequential(*layers)

class YeastModel(nn.Module):
    def __init__(self, cfg: Dict, use_lora=False, lora_rank=4, lora_alpha=16):
        super().__init__()
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 特征编码器
        self.feature_encoder = FeatureEncoder(cfg['feature_encoders'])
        
        # 特征融合
        self.feature_fusion = FeatureFusion(cfg['feature_fusion'])
        
        # Transformer
        self.transformer = YeastTransformer(cfg['transformer'])
        
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

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 特征编码
        encoded_features = {}
        for key, value in x.items():
            if key in self.feature_encoder.encoders:
                encoded_features[key] = self.feature_encoder.encoders[key](value)
        
        # 特征融合
        fused_features = torch.cat([encoded_features[key] for key in sorted(encoded_features.keys())], dim=-1)
        fused_features = self.feature_fusion.fusion_net(fused_features)
        
        # Transformer编码
        transformer_output = self.transformer(fused_features)
        
        # 预测
        predictions = self.prediction_head.pred_net(transformer_output)
        
        return predictions

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predictions, targets)

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