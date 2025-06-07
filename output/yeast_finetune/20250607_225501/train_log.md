
## before 评估时间：2025-06-07 22:55:04
- Pearson r: -0.0495
- P值: 8.33e-82
- Slope: -0.0017
- 样本数: 149688
- 预测均值: 0.2149
- 预测方差: 0.0699
- 真实均值: 4.1300
- 真实方差: 1.9793
- 结果可视化: output/yeast_finetune/20250607_225501/pred_vs_true_log1p_before.png

## after 评估时间：2025-06-07 22:58:39
- Pearson r: 0.1846
- P值: 0.00e+00
- Slope: 0.0440
- 样本数: 149688
- 预测均值: 4.0439
- 预测方差: 0.4720
- 真实均值: 4.1300
- 真实方差: 1.9793
- 结果可视化: output/yeast_finetune/20250607_225501/pred_vs_true_log1p_after.png

## 实验时间：2025-06-07 22:58:39
### 主要修改
- （请在此补充本次实验的主要修改点）

### 训练参数
- stage: fit
- type: region
- assembly: hg38
- eval_tss: False
- log_image: False
- experiment: {'name': 'yeast_lora_finetuning', 'seed': 42, 'debug': False}
- training: {'batch_size': 32, 'num_workers': 4, 'learning_rate': 0.0001, 'weight_decay': 0.0001, 'max_epochs': 100, 'early_stopping_patience': None, 'gradient_clip_val': 1.0, 'use_lora': True, 'lora_rank': 8, 'lora_alpha': 16, 'lora_layers': ['encoder', 'head_exp'], 'save_every_n_epochs': 1, 'accumulate_grad_batches': 2, 'clip_grad': 1.0, 'use_fp16': True, 'log_every_n_steps': 10, 'val_check_interval': 1.0, 'warmup_epochs': 2}
- data: {'data_path': 'input/20250601_data/yeast_data_with_conditions_original_peaks.zarr'}
- model: {'_target_': 'get_model.model.yeast_model.YeastModel', 'cfg': {'num_regions': 2268, 'num_motif': 283, 'embed_dim': 256, 'num_layers': 3, 'num_heads': 4, 'dropout': 0.1, 'output_dim': 1, 'flash_attn': False, 'pool_method': 'mean', 'region_embed': {'num_regions': '${model.cfg.num_regions}', 'num_features': '${model.cfg.num_motif}', 'embed_dim': '${model.cfg.embed_dim}'}, 'encoder': {'num_heads': '${model.cfg.num_heads}', 'embed_dim': '${model.cfg.embed_dim}', 'num_layers': '${model.cfg.num_layers}', 'drop_path_rate': '${model.cfg.dropout}', 'drop_rate': 0, 'attn_drop_rate': 0, 'use_mean_pooling': False, 'flash_attn': '${model.cfg.flash_attn}'}, 'head_exp': {'embed_dim': '${model.cfg.embed_dim}', 'output_dim': '${model.cfg.output_dim}', 'use_atac': False}, 'loss': {'components': {'exp': {'_target_': 'torch.nn.PoissonNLLLoss', 'reduction': 'mean', 'log_input': False}}, 'weights': {'exp': 1.0}}, 'metrics': {'components': {'exp': ['pearson', 'spearman', 'r2']}}}, 'model': {'name': 'yeast_model', 'type': 'transformer', 'feature_encoders': {'media_type': {'type': 'embedding', 'num_embeddings': 2, 'embedding_dim': 32, 'padding_idx': None}, 'temperature': {'type': 'linear', 'in_features': 1, 'out_features': 32}, 'pre_culture': {'time': {'type': 'linear', 'in_features': 1, 'out_features': 32}, 'od600': {'type': 'linear', 'in_features': 1, 'out_features': 32}}, 'drug_culture': {'time': {'type': 'linear', 'in_features': 1, 'out_features': 32}, 'drug_name': {'type': 'embedding', 'num_embeddings': 100, 'embedding_dim': 32, 'padding_idx': 0}, 'concentration': {'type': 'linear', 'in_features': 1, 'out_features': 32}}, 'carbon_source': {'type': 'embedding', 'num_embeddings': 10, 'embedding_dim': 32, 'padding_idx': 0}, 'nitrogen_source': {'type': 'embedding', 'num_embeddings': 10, 'embedding_dim': 32, 'padding_idx': 0}}, 'feature_fusion': {'type': 'concat', 'fusion_dim': 256, 'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.1}, 'transformer': {'num_layers': 3, 'num_heads': 4, 'hidden_dim': 128, 'dropout': 0.1, 'activation': 'gelu', 'layer_norm_eps': 1e-05}, 'prediction_head': {'type': 'mlp', 'hidden_dims': [128, 64], 'dropout': 0.1, 'activation': 'relu', 'output_dim': 1}, 'loss': {'type': 'mse', 'reduction': 'mean'}, 'optimizer': {'type': 'adam', 'lr': 0.001, 'weight_decay': 0.0001, 'betas': [0.9, 0.999], 'eps': 1e-08}, 'scheduler': {'type': 'cosine', 'T_max': 100, 'eta_min': 1e-06}}}
- finetune: {'pretrain_checkpoint': True, 'checkpoint': 'checkpoints/fetal_pretrain_fetal_finetune/Astrocytes/checkpoint-best.pth', 'strict': False, 'model_key': 'state_dict', 'use_lora': True, 'lora_checkpoint': None, 'layers_with_lora': ['encoder', 'head_exp'], 'patterns_to_freeze': [], 'patterns_to_drop': [], 'additional_checkpoints': []}
- logging: {'save_dir': 'output/yeast_finetune', 'tensorboard': True, 'wandb': True, 'project_name': 'yeast_finetune', 'run_name': 'lora_finetune_v1', 'use_wandb': False}
- machine: {'output_dir': 'output', 'num_devices': 1}
- optimizer: {'lr': 0.0001, 'min_lr': 1e-06, 'weight_decay': 0.05, 'opt': 'adamw', 'opt_eps': 1e-08, 'opt_betas': [0.9, 0.999]}
- scheduler: {'type': 'cosine', 'T_max': 50, 'eta_min': 1e-06, 'warmup_epochs': 2}
- task: {'test_mode': 'normal', 'interpret': False, 'save_predictions': False, 'save_embeddings': False}
- dataset: {'_target_': 'get_model.data.yeast_dataset.YeastZarrDataset', 'zarr_path': '/home/rhyswei/Code/aiyeast/get_model/input/20250601_data/yeast_data_with_conditions_original_peaks.zarr', 'split': 'train', 'batch_size': 8, 'shuffle': True, 'num_workers': 1}

### 训练总耗时
- 3.6 分钟

### 评估指标 (log1p空间)
- 最终训练loss: 3.6538
- 预测均值: 4.0439
- 预测方差: 0.4720
- 真实均值: 4.1300
- 真实方差: 1.9793
