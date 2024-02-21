#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out


module load pytorch/1.13
. ./env.sh

train.py \
  --data-dir data/streams/combined \
  --output-dir results/aagcn \
  --age-file metadata/combined.csv \
  --learning-rate 0.01 \
  --batch-size 32 \
  --num-workers 16 \
  --streams j \
  --k-folds 10 \
  --epochs 20 \
  --adaptive \
  --attention
