#!/bin/bash
#SBATCH --job-name=open_r1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu

set -x -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate grpo

accelerate launch --config_file recipes/accelerate_configs/ddp.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-3B-Instruct/sft/config_demo_eval.yaml