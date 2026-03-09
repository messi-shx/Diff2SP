#!/bin/bash
#SBATCH -J opt_all
#SBATCH -A liu334
#SBATCH --mem=100G
#SBATCH --partition a100-40gb
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=all
#SBATCH -N 1
#SBATCH -t 5-00:00
#SBATCH --gres=gpu:1
#SBATCH --output=gan_sample.out
#SBATCH --error=gan_sample.err

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

cd /home/sun1321/src/diff2sp_new
python main.py --mode sample