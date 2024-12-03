#!/bin/bash

declare -a commands=(
  # Kt
  "train.py --data-dir data/streams/combined --output-dir results/stgcn-kt3 --streams j --edges phys --kt 3"
  "train.py --data-dir data/streams/combined --output-dir results/stgcn-kt5 --streams j --edges phys --kt 5"
  "train.py --data-dir data/streams/combined --output-dir results/stgcn-kt7 --streams j --edges phys --kt 7"
  "train.py --data-dir data/streams/combined --output-dir results/stgcn-kt9 --streams j --edges phys --kt 9"
  "train.py --data-dir data/streams/combined --output-dir results/stgcn-kt11 --streams j --edges phys --kt 11"
  "train.py --data-dir data/streams/combined --output-dir results/stgcn-kt13 --streams j --edges phys --kt 13"
  "train.py --data-dir data/streams/combined --output-dir results/stgcn-kt15 --streams j --edges phys --kt 15"
  "train.py --data-dir data/streams/combined --output-dir results/stgcn-kt17 --streams j --edges phys --kt 17"

  # Adaptive graph and inits
  "train.py --data-dir data/streams/combined --output-dir results/agcn --streams j --adaptive --edges phys"
  "train.py --data-dir data/streams/combined --output-dir results/aagcn --streams j --adaptive --attention --edges phys"
  "train.py --data-dir data/streams/combined --output-dir results/aagcn-coord --streams j --adaptive --attention --edges coord"
  "train.py --data-dir data/streams/combined --output-dir results/aagcn-fc --streams j --adaptive --attention --edges fc"

  # Input streams
  "train.py --data-dir data/streams/combined --output-dir results/jb-aagcn-coord --streams j,b --adaptive --attention --edges coord"
  "train.py --data-dir data/streams/combined --output-dir results/jbv-aagcn-coord --streams j,b,v --adaptive --attention --edges coord"
  "train.py --data-dir data/streams/combined --output-dir results/jbva-aagcn-coord --streams j,b,v,a --adaptive --attention --edges coord"

  # Rotation / 2D
  "train.py --data-dir data/streams_raw/combined --output-dir results/jb-aagcn-coord-raw --streams j,b --adaptive --attention --edges coord"
  "train.py --data-dir data/streams_2d/combined --output-dir results/jb-aagcn-coord-xy --streams j,b --xy-data --adaptive --attention --edges coord"
  "train.py --data-dir data/streams_raw_2d/combined --output-dir results/jb-aagcn-coord-raw-xy --streams j,b --xy-data --adaptive --attention --edges coord"

  # + features
  "train.py --data-dir data/streams/combined --output-dir results/jb-aagcn-coord-fts --streams j,b --concat-features --adaptive --attention --edges coord"
)

# Submit each job separately
for cmd in "${commands[@]}"; do
  sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out

# Load the environment and modules
. ./env.sh
module load pytorch/1.13

# Execute the command
echo "Running command: $cmd"
srun $cmd
EOT
  sleep 1
done
