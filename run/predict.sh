#!/bin/bash
#SBATCH --account=project_2004522
#SBATCH --ntasks=6
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out


module load pytorch/1.13
. ./env.sh

declare -a models=(
    'stgcn-physical' 'stgcn-xy' 'stgcn' 'aagcn' 'ms-aagcn' 'ms-aagcn-fts'
)

for model in "${models[@]}"; do
    echo "Running prediction for model: $model"
    srun --exclusive -N1 -n1 predict.py --model-dir results/$model --output-dir predictions &
done

wait
echo "All prediction tasks completed."