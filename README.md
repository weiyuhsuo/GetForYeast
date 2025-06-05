# GET Model for Yeast

基于Transformer的酵母基因表达预测模型。

## 项目结构

```
get_model/
├── config/                 # 配置文件目录
│   ├── yeast_training.yaml # 训练配置
│   ├── yeast_model.yaml    # 模型配置
│   └── yeast_dataset.yaml  # 数据集配置
├── get_model/             # 核心代码
│   ├── data/             # 数据处理
│   ├── model/            # 模型定义
│   └── utils/            # 工具函数
├── input/                # 输入数据目录
├── checkpoints/          # 模型检查点
├── output/              # 输出目录
├── train_yeast.py       # 训练脚本
└── env.yml              # 环境配置
```

## 环境配置

```bash
conda env create -f env.yml
conda activate get
```

## 使用方法

1. 准备数据：
   - 将数据文件放在 `input/` 目录下
   - 确保数据格式符合要求

2. 训练模型：
```bash
python train_yeast.py
```

3. 配置说明：
   - 修改 `config/` 下的配置文件以调整训练参数
   - 主要参数包括：batch_size、learning_rate、epochs等

## 模型架构

- 基于Transformer的编码器-解码器结构
- 使用多头注意力机制处理序列数据
- 支持条件输入和表达量预测

## 数据格式

- 输入数据：Zarr格式
- 包含motif特征和表达量标签
- 支持训练/验证/测试集划分

## 注意事项

- 确保有足够的GPU内存
- 建议使用conda环境运行
- 数据预处理可能需要较长时间
