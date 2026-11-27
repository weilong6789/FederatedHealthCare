#!/bin/bash
# Full kept-Adam rerun on ISIC in one go: FedAvg K = 1, 5, 20 x seeds 0-2 with
# --keep-opt, into results_keepopt/, then summarize.py once every run ends.
#   bash run_keepopt.sh
# 9 GPU jobs (about 10 GPU-hours, ~1.5 h wall time if GPUs are free) and one
# CPU job. In the morning: results_keepopt/summary.md and logs/isic-keepopt-*.
set -e
cd "$(dirname "$0")"
OUT=results_keepopt
if [ -e "$OUT/isic.csv" ]; then
  echo "$OUT/isic.csv already exists; move it away first so runs aren't counted twice." >&2
  exit 1
fi
mkdir -p logs "$OUT"
# pooled is the reference summarize.py compares against; it doesn't depend on K
for f in isic.csv isic_predictions.csv; do
  awk -F, 'NR==1 || $1=="pooled"' "results/$f" > "$OUT/$f"
done
ids=()
for T in ${TS:-1 5 20}; do
  for s in ${SEEDS:-0 1 2}; do
    ids+=("$(sbatch --parsable -J "isic-keepopt-T$T" isic.sbatch \
             --method fedavg --T "$T" --seed "$s" --keep-opt --out "$OUT")")
  done
done
echo "submitted: ${ids[*]}"
dep=$(IFS=:; echo "${ids[*]}")
sbatch -J isic-keepopt-summary -N1 --cpus-per-task=4 --mem=16G -t 2:00:00 \
  -o logs/%x-%j.out --dependency="afterany:$dep" \
  --wrap "module load anaconda3 && conda activate ~/scratch/envs/flamby && cd $PWD && python summarize.py $OUT"
