#!/usr/bin/env bash
set -euo pipefail

pids=()

run_cmd() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" "$@" &
  pids+=("$!")
}

run_cmd 0 python exp/Network_exp/Dynamic_delay_gnn.py \
  --graph-dir data/Aeolus/Flight_network/network_data_2024 \
  --start-date 2024-01-01 \
  --end-date 2024-12-30 \
  --feature-mode split \
  --epochs 50

run_cmd 1 python exp/Network_exp/Temporal_delay_gnn.py \
 --graph-dir data/Aeolus/Flight_network/network_data_2024 \
  --start-date 2024-01-01 \
  --end-date 2024-12-30 \
  --feature-mode split \
  --epochs 50

run_cmd 2 python exp/Network_exp/AFM_node_embedding.py \
  --graph-dir data/Aeolus/Flight_network/network_data_2024 \
  --start-date 2024-01-01 \
  --end-date 2024-12-30 \
  --vgae-model-path vgae/node_embedding.pth \
  --plot

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"