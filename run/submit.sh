#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out


module load pytorch/1.13
. ./env.sh

train.py \
  --data-dir data/streams/combined \
  --output-dir results/jb-aagcn-coord \
  --streams j,b \
  --adaptive \
  --attention \
  --edges coord