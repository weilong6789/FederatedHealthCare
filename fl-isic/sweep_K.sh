#!/bin/bash
# Submit the FedAvg local-steps sweep on ISIC: one job per K and seed.
#   bash sweep_K.sh                        # default K values, seeds 0-2, 20 epochs, into results/
#   KS="1 5" SEEDS="0" bash sweep_K.sh     # a subset
#   bash sweep_K.sh --out results_sweep    # extra arguments go to isic.py
#
# Every K divides the 20-epoch budget (5820 steps), so each run takes exactly
# the steps pooled takes: from K = 1 (5820 rounds) to K = 5820 (one round).
#   K      1     5    20   97  291  970  5820
#   rounds 5820  1164 291  60  20   6    1
# Pooled doesn't depend on K; its 20-epoch rows in results/ are the reference.
set -e
cd "$(dirname "$0")"
mkdir -p logs
KS=${KS:-"1 5 20 97 291 970 5820"}
SEEDS=${SEEDS:-"0 1 2"}
for K in $KS; do
  for s in $SEEDS; do
    sbatch -J "isic-K$K" isic.sbatch --method fedavg --K "$K" --seed "$s" "$@"
  done
done
