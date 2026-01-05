#!/usr/bin/env bash
set -euo pipefail

python exp/Network_exp/train_vgae_node_embedding.py \
  --graph-dir data/Aeolus/Flight_network/network_data_2024 \
  --start-date 2024-01-01 \
  --end-date 2024-12-30 \
  --output-path processed_data/vgae/node_embedding_2024.pth \
  --epochs 50
