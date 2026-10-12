#!/usr/bin/env bash
# Runs every baseline for 5 seeds, then builds the table and plots.
set -e
SEEDS="0 1 2 3 4"
LR=0.01          # set this to the value you picked in step 3

mkdir -p results
if [ -f results/results.csv ]; then     # keep old runs, don't mix them in
  mv results/results.csv "results/old_results_$(date +%Y%m%d_%H%M%S).csv"
fi

python baselines.py --seeds $SEEDS --lr $LR --epochs 50
for s in $SEEDS; do
  flwr run . --run-config "seed=$s lr=$LR" --stream
done
python summarize.py
