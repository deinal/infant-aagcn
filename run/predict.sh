#!/bin/bash

declare -a models=(
    'jb-aagcn-coord' 'jb-aagcn-coord-xy'
)

# Submit each job separately using sbatch
for model in "${models[@]}"; do
  sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out

. ./env.sh
module load pytorch/1.13

srun predict.py --model-dir results/$model --output-dir predictions
EOT
  sleep 1
done
