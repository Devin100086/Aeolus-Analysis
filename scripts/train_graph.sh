#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

python exp/Network_exp/Dynamic_delay_gnn.py \
  --graph-dir data/Aeolus/Flight_network/network_data_2024 \
  --start-date 2024-01-01 \
  --end-date 2024-12-30 \
  --feature-mode split \
  --batch-nodes 0 \
  --oversample \
  --epochs 50