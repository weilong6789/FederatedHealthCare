#!/bin/bash
# One ISIC run on one GPU. Pass the isic.py arguments after the script name:
#   sbatch isic.sbatch --method pooled --seed 0
#   sbatch isic.sbatch --method fedavg --K 100 --seed 0
#SBATCH -J isic
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out

module load anaconda3
conda activate ~/scratch/envs/flamby
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python isic.py "$@"
