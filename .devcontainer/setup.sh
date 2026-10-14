#!/usr/bin/env bash
set -e
ROOT="$(pwd)"

[ -d /workspaces/FLamby ] || git clone https://github.com/owkin/FLamby.git /workspaces/FLamby
pip install -e "/workspaces/FLamby[heart]"
pip install "numpy<2"        # FLamby breaks on NumPy 2

cd /workspaces/FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset

cd "$ROOT/fl-tabular"
pip install -e .
pip install "numpy<2"        # in case the install above pulled NumPy 2 back in
flwr federation simulation-config --num-supernodes 4
chmod +x run_all.sh
