#!/bin/bash
# Quick check of the fresh-Adam explanation for FedAvg K = 1 on ISIC: one run,
# seed 0, with each hospital keeping its Adam state across rounds.
#   mkdir -p logs && sbatch keepopt_check.sbatch
# About an hour on an H100. Compare the last "all: bal acc" line in the log
# with fresh-Adam K = 1 (0.485) and pooled (0.748).
#SBATCH -J isic-keepopt-T1
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o logs/%x-%j.out

module load anaconda3
conda activate ~/scratch/envs/flamby
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python isic.py --method fedavg --T 1 --seed 0 --keep-opt --out results_keepopt
echo "final balanced accuracy, all test images (fresh Adam K=1: 0.485, pooled: 0.748):"
awk -F, '$1=="fedavg" && $2==1 && $3==0 && $7=="all" {b=$9} END {print b}' results_keepopt/isic.csv
