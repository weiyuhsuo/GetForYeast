# GET模型修改记录

## 原始代码结构

GET (General Expression Transformer) 是一个基于Transformer的转录预测模型，主要包含以下组件：

1. 核心模块 (`get_model/`):
   - `model/`: 模型定义
   - `dataset/`: 数据加载和处理
   - `config/`: 配置文件
   - `run.py`: 主要运行脚本
   - `utils.py`: 工具函数
   - `optim.py`: 优化器
   - `task.py`: 任务定义

2. 配置文件 (`get_model/config/`):
   - `config.py`: 基础配置
   - `model/*.yaml`: 模型配置
   - `dataset/*.yaml`: 数据集配置

3. 数据集处理 (`get_model/dataset/`):
   - `zarr_dataset.py`: 主要数据集类
   - `collate.py`: 数据批处理
   - `utils.py`: 数据集工具函数

## 计划修改

### 1. 数据特征扩展

基于 `info.md` 中的信息，我们需要添加以下特征：

#### 1.1 培养基信息
- 主要类型：sc, ypd
- 扩展：可以添加更多培养基类型
- 实现：使用分类编码

#### 1.2 培养条件
- 温度范围：20~45℃（主要使用30/25/42）
- 预培养：
  - 时间：3-4h
  - 终点：OD600值（0.n，不超过3）
- 加药培养：
  - 时间/终点
  - 药物名称
  - 药物浓度（mol浓度）
- 碳源/氮源（可选）

### 2. 代码修改计划

#### 2.1 数据集修改 (`get_model/dataset/`)
- 修改 `zarr_dataset.py`：
  - 添加新的特征处理逻辑
  - 实现特征编码和归一化
- 修改 `collate.py`：
  - 添加新特征的批处理逻辑

#### 2.2 模型修改 (`get_model/model/`)
- 添加新的特征输入层
- 修改特征融合逻辑
- 添加新的预测头

#### 2.3 配置修改 (`get_model/config/`)
- 创建新的配置文件 `yeast_training.yaml`
- 添加新特征的配置选项
- 修改训练参数

### 3. 具体实现步骤

1. 数据预处理
   - [ ] 实现培养基编码
   - [ ] 实现温度归一化
   - [ ] 实现预培养特征处理
   - [ ] 实现加药培养特征处理
   - [ ] 实现碳源/氮源编码（可选）

2. 模型修改
   - [ ] 添加特征输入层
   - [ ] 实现特征融合
   - [ ] 添加预测头

3. 配置更新
   - [x] 创建新配置文件
   - [ ] 更新训练参数
   - [ ] 添加特征配置

## 修改记录

### 2024-05-28
1. 创建酵母训练配置文件 `yeast_training.yaml`
   - 添加实验配置（名称、随机种子等）
   - 添加训练参数（批次大小、学习率等）
   - 添加数据配置（培养基类型、温度范围等）
   - 添加特征编码配置（分类编码、归一化等）
   - 添加模型配置（特征维度、隐藏层等）
   - 添加日志配置（保存目录、TensorBoard等）

### 2024-05-28
2. 创建酵母模型配置文件 `yeast_model.yaml`
   - 添加特征编码器配置
     * 培养基类型（SC/YPd）嵌入层
     * 温度线性层
     * 预培养特征（时间、OD600）线性层
     * 药物培养特征（时间、药物名称、浓度）混合编码
     * 碳源和氮源嵌入层
   - 添加特征融合配置
     * 拼接方式融合
     * 两层MLP特征转换
   - 添加Transformer架构配置
     * 3层Transformer
     * 4个注意力头
     * GELU激活函数
   - 添加预测头配置
     * 两层MLP
     * ReLU激活函数
   - 添加训练相关配置
     * MSE损失函数
     * Adam优化器
     * 余弦退火学习率调度器

### 2024-05-28
3. 实现模型代码 `yeast_model.py`
   - 实现特征编码器类 `FeatureEncoder`
     * 支持所有特征类型的编码
     * 使用ModuleDict管理多个编码器
   - 实现特征融合模块 `FeatureFusion`
     * 特征拼接
     * 两层MLP特征转换
   - 实现Transformer模块 `YeastTransformer`
     * 使用PyTorch的TransformerEncoder
     * 支持位置编码
   - 实现预测头模块 `PredictionHead`
     * 可配置的MLP层
     * 支持dropout和激活函数
   - 实现完整模型类 `YeastModel`
     * 模块化设计
     * 支持配置文件加载

4. 实现数据集处理 `yeast_dataset.py`
   - 实现数据集类 `YeastDataset`
     * 支持CSV和Excel格式
     * 特征预处理和归一化
     * 支持数据增强
   - 实现数据加载器创建函数
     * 支持训练/验证/测试集划分
     * 支持多进程加载
     * 支持批处理

5. 实现训练脚本 `train_yeast.py`
   - 实现训练循环
     * 支持GPU训练
     * 支持学习率调度
     * 支持模型检查点保存
   - 实现验证逻辑
     * 定期验证
     * 保存最佳模型
   - 实现日志记录
     * 文件日志
     * TensorBoard支持
   - 实现命令行接口

6. 实现评估脚本 `evaluate_yeast.py`
   - 实现模型评估
     * 计算多个评估指标
     * 支持批处理评估
   - 实现结果可视化
     * 预测值vs真实值散点图
     * 残差图
     * 特征重要性分析
   - 实现结果保存
     * CSV格式保存详细结果
     * JSON格式保存评估指标
   - 实现命令行接口

### 2024-05-30 LoRA微调与一键运行支持

1. 配置文件`yeast_training.yaml`支持LoRA参数、数据路径等直接配置。
2. `yeast_model.py`支持LoRA分支，自动根据配置启用。
3. `yeast_dataset.py`支持直接读取csv格式的motif+accessibility+实验条件+target数据。
4. 新增`train_yeast.py`脚本，实现一键训练。
5. `evaluate_yeast.py`支持LoRA权重加载，命令行参数化。
6. 操作流程梳理：
   - 准备数据：`input/gsm_example_with_condition.csv`，每行包含motif、accessibility、实验条件、target。
   - 配置文件：`get_model/config/yeast_training.yaml`，设置LoRA参数和数据路径。
   - 训练：
     ```bash
     python get_model/train_yeast.py \
       --config get_model/config/yeast_training.yaml \
       --data input/gsm_example_with_condition.csv \
       --output output/yeast_lora_finetune \
       --device cuda
     ```
   - 评估：
     ```bash
     python get_model/get_model/evaluate_yeast.py \
       --config get_model/config/yeast_training.yaml \
       --data input/gsm_example_with_condition.csv \
       --model output/yeast_lora_finetune/best_model.pth \
       --output output/yeast_lora_eval \
       --device cuda
     ```

### 2024-06-01

#### Debugging and Training Setup

本次修改主要集中在解决训练启动过程中遇到的配置错误和数据模块与模型交互的问题。经过一系列的调试，我们成功地运行了模型的单 epoch 训练。

主要解决的问题和修改如下：

1.  **配置文件缺失 `task` 键**：在 `yeast_training.yaml` 文件中添加了缺失的 `task` 配置部分，包括 `test_mode`、`interpret`、`save_predictions` 和 `save_embeddings` 等，并设置了合理的默认值。
2.  **配置文件中 `epochs` 与代码不匹配**：代码中的 `configure_optimizers` 和 `cosine_scheduler` 期望使用 `cfg.training.epochs`，但配置文件中使用的是 `max_epochs`。我们修改了 `get_model/get_model/run.py` 中 `configure_optimizers` 内对 `cosine_scheduler` 的调用，将 `epochs` 参数改为从 `self.cfg.training.max_epochs` 获取。
3.  **配置文件中 `warmup_epochs` 键缺失**：在 `yeast_training.yaml` 的 `training` 部分添加了 `warmup_epochs` 键，并将其设置为 0，以满足 `cosine_scheduler` 的参数要求。
4.  **配置文件中 `batch_size` 路径错误**：代码中的 `configure_optimizers`, `_shared_step`, `training_step`, 和 `validation_step` 方法错误地尝试从 `self.cfg.machine.batch_size` 获取 `batch_size`。我们修正了这些位置，使其从正确的路径 `self.cfg.training.batch_size` 获取。
5.  **数据模块 `dataset_train` 在 `configure_optimizers` 中未初始化**：由于 Lightning AI 的生命周期，`configure_optimizers` 在数据模块的 `setup` 方法之前运行，导致 `self.dm.dataset_train` 为 `None`。为了解决这个问题，我们在 `train_yeast.py` 脚本中，在创建 `RegionLitModel` 实例之前，手动调用了 `datamodule.setup(stage='fit')`，获取了训练集大小，并将该大小 (`train_data_size`) 作为参数传递给了 `RegionLitModel` 的构造函数。
6.  **`RegionLitModel` 构造函数不接受 `train_data_size`**：修改了 `aiyeast/get_model/get_model/run_region.py` 中 `RegionLitModel` 的 `__init__` 方法，使其接受 `train_data_size` 参数，并将其正确传递给其父类 `LitModel` 的构造函数。
7.  **`LitModel.configure_optimizers` 仍依赖 `self.dm`**：在 `LitModel` 的 `__init__` 中存储了传递进来的 `train_data_size`，并修改了 `configure_optimizers` 方法，使其直接使用存储的 `_train_data_size`，不再依赖于 `self.dm`。
8.  **`pickle.UnpicklingError` 兼容性问题 (回顾)**：虽然在之前的对话中解决，但也属于重要修改。我们修改了 `load_checkpoint` 函数 (`aiyeast/get_model/get_model/utils.py`) 以兼容 PyTorch 2.6 及以上版本加载旧检查点时出现的 `WeightsUnpickler error`，实现了 `weights_only=True` 和 `weights_only=False` 的回退加载逻辑。

#### 模型结构、输入输出处理与微调

当前您正在使用的模型是 `GETRegionFinetune` 类，它继承自 `BaseGETModel`。其核心结构和数据处理流程如下：

*   **基础架构**: `GETRegionFinetune` 主要包含以下几个关键组件：
    *   `RegionEmbed`: 负责将输入的区域特征（包括 motif 和 accessibility）编码成 embedding。
    *   `Encoder`: 一个 Transformer 编码器，用于处理区域 embedding，捕捉区域之间的相互作用和长距离依赖。这部分是预训练模型的核心。
    *   `head_exp`: 一个 Expression Head，负责根据 Transformer 编码器的输出预测表达水平。在 `GETRegionFinetune` 中，这是一个回归头。

*   **输入处理 (`get_input` 方法)**:
    *   `GETRegionFinetune` 的 `get_input` 方法从数据批次 (`batch`) 中获取 `region_motif` 数据作为模型的直接输入。根据代码，实验条件 (`condition`) 数据也包含在 `batch` 中，**理论上应该在模型的某个地方被利用**。然而，从 `GETRegionFinetune` 的 `forward` 方法签名 `forward(self, region_motif)` 来看，它**直接接收的输入似乎只包含 `region_motif`**。这可能意味着当前的 `GETRegionFinetune` 模型结构并未直接在 `forward` 方法中处理实验条件。如果需要将实验条件融入模型，可能需要在 `GETRegionFinetune` 类中修改 `get_input` 和 `forward` 方法，或者在 `RegionEmbed` 或 `Encoder` 之后添加额外的融合模块来结合实验条件 embedding。
    *   数据批次 (`batch`) 是由 `YeastDataModule` 中的 `YeastZarrDataset` 生成的字典，包含 `region_motif`, `condition`, `exp_label` 等键。

*   **前向传播 (`forward` 方法)**:
    *   `forward` 方法接收 `region_motif` 作为输入。
    *   `region_motif` 首先经过 `RegionEmbed` 层生成区域 embedding。
    *   生成的区域 embedding 随后输入到 `Encoder` (Transformer) 进行处理。
    *   Transformer 编码器的输出然后传递给 `head_exp` (Expression Head) 进行表达水平预测。
    *   最终输出是预测的表达水平。

*   **预训练与微调**:
    *   **预训练模型** (`checkpoint-799.pth`) 是在一个更通用的任务或更大的数据集上训练的 GET 模型。根据文件名，它可能是在 fetal 和 adult 数据上预训练的，可能涉及表达预测或其他相关的基因组任务。
    *   **微调过程**：我们加载预训练检查点，并在这个基础上应用 LoRA。这意味着预训练模型的 `RegionEmbed` 和 `Encoder` 的大部分权重是冻结的。LoRA 在 `encoder` 和 `head_exp` 层添加了可训练的低秩适配矩阵。因此，微调主要调整了 `encoder` 和 `head_exp` 中与 LoRA 相关的参数，以及少数未冻结的原始参数。
    *   **输出转换**：预训练模型的输出头（或任务）可能与我们当前的酵母表达预测任务不同。在 `GETRegionFinetune` 类中，我们使用了 `head_exp` 这个专门用于表达预测的头。当加载预训练检查点时，`load_state_dict` 函数会尝试匹配检查点中的权重到当前模型的层结构。如果预训练模型有不同的输出头，`strict=False` 参数（在配置文件中设置）允许忽略检查点中与当前模型结构不匹配的键，从而实现从预训练任务到当前微调任务的"输出转换"。Essentially, the pre-trained weights for the core `RegionEmbed` and `Encoder` are reused, and a new `head_exp` is potentially either initialized randomly (if not present in the checkpoint) or adapted from a similar head in the checkpoint (if strict=False allows it and names match partially), and then trained along with the LoRA layers for the specific yeast expression prediction task.

*   **`before_loss` 方法**: 这个方法负责将模型的原始输出 (`output`) 和数据批次 (`batch`) 中的真实标签进行匹配和格式化，以便计算损失和评估指标。在 `GETRegionFinetune` 中，它将模型的预测输出和 `batch` 中的 `exp_label` 提取出来，准备进行损失计算和指标评估。

总结来说，微调是基于预训练模型的核心特征提取能力（`RegionEmbed` 和 `Encoder`），通过 LoRA 和训练新的/ adapted 的 `head_exp` 来使其适应酵母表达预测任务。预训练和微调的输出头通常是不同的，通过 `load_state_dict` 的 `strict=False` 机制实现了权重的重用和任务的切换。