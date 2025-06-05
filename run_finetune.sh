#!/bin/bash
#SBATCH --job-name=yeast_finetune
#SBATCH --output=logs/finetune_%j.log
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# 激活环境
source activate get

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4

# 创建日志目录
mkdir -p logs

# 运行训练脚本
python train_yeast.py 