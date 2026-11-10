#!/bin/bash
# Submit the FedAvg local-steps sweep on ISIC: one job per T and seed.
#   bash sweep_T.sh                        # default T values, seeds 0-2, 20 epochs, into results/
#   TS="1 5" SEEDS="0" bash sweep_T.sh     # a subset
#   bash sweep_T.sh --out results_sweep    # extra arguments go to isic.py
#
# Every T divides the 20-epoch budget (5820 steps), so each run takes exactly
# the steps pooled takes: from T = 1 (5820 rounds) to T = 5820 (one round).
#   T      1     5    20   97  291  970  5820
#   rounds 5820  1164 291  60  20   6    1
# Pooled doesn't depend on T; its 20-epoch rows in results/ are the reference.
set -e
cd "$(dirname "$0")"
mkdir -p logs
TS=${TS:-"1 5 20 97 291 970 5820"}
SEEDS=${SEEDS:-"0 1 2"}
for T in $TS; do
  for s in $SEEDS; do
    sbatch -J "isic-T$T" isic.sbatch --method fedavg --T "$T" --seed "$s" "$@"
  done
done
