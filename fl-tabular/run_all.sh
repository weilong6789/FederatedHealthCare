#!/usr/bin/env bash
# Runs every baseline for 5 seeds, then builds the tables and plots.
# Uses the settings tune.py picked (results/tuned.env) if it has been run.
set -e
SEEDS="0 1 2 3 4"
POOLED_LR=0.01 LOCAL_LR=0.01 FEDAVG_LR=0.01 FEDAVG_EPOCHS=1   # untuned defaults
if [ -f results/tuned.env ]; then
  source results/tuned.env
  echo "using tuned settings from results/tuned.env"
else
  echo "no results/tuned.env: using untuned defaults (run python tune.py first)"
fi

mkdir -p results
STAMP=$(date +%Y%m%d_%H%M%S)
for f in results predictions; do       # keep old runs, don't mix them in
  if [ -f "results/$f.csv" ]; then mv "results/$f.csv" "results/old_${f}_$STAMP.csv"; fi
done

python baselines.py --seeds $SEEDS --pooled-lr $POOLED_LR --local-lr $LOCAL_LR --epochs 50
for s in $SEEDS; do
  flwr run . --run-config "seed=$s lr=$FEDAVG_LR local-epochs=$FEDAVG_EPOCHS" --stream
done
python summarize.py
