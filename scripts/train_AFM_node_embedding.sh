#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4

python exp/Network_exp/AFM_node_embedding.py \
  --graph-dir data/Aeolus/Flight_network/network_data_2016 \
  --start-date 2016-01-01 \
  --end-date 2016-12-30 \
  --vgae-model-path vgae/node_embedding.pth \
  --plot
