#!/usr/bin/env bash
set -euo pipefail

pids=()

run_cmd() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" "$@" &
  pids+=("$!")
}

run_cmd 0 python exp/Chain_exp/CNN_LSTM.py \
  --data-dir data/Aeolus/Flight_chain/chain_data_2023 \
  --year 2023 \
  --feature-columns processed_data/2023/new_feature_columns.pkl \
  --label-mode both \
  --checkpoint-path checkpoints/CNN_LSTM/cnnlstm_model_checkpoint.pth \
  --best-model-path checkpoints/CNN_LSTM/cnnlstm_best_model.pth \
  --max-epochs 50 \
  --plot

run_cmd 1 python exp/Chain_exp/GRU.py \
  --data-dir data/Aeolus/Flight_chain/chain_data_2023 \
  --year 2023 \
  --feature-columns processed_data/2023/new_feature_columns.pkl \
  --label-mode both \
  --checkpoint-path checkpoints/GRU/gru_model_checkpoint.pth \
  --best-model-path checkpoints/GRU/gru_best_model.pth \
  --max-epochs 50 \
  --plot

run_cmd 2 python exp/Chain_exp/LSTM.py \
  --data-dir data/Aeolus/Flight_chain/chain_data_2023 \
  --year 2023 \
  --feature-columns processed_data/2023/new_feature_columns.pkl \
  --label-mode both \
  --checkpoint-path checkpoints/LSTM/lstm_model_checkpoint.pth \
  --best-model-path checkpoints/LSTM/lstm_best_model.pth \
  --max-epochs 50 \
  --plot

run_cmd 3 python exp/Chain_exp/MogrifierLSTM.py \
  --data-dir data/Aeolus/Flight_chain/chain_data_2023 \
  --year 2023 \
  --feature-columns processed_data/2023/new_feature_columns.pkl \
  --label-mode both \
  --checkpoint-path checkpoints/MogrifierLSTM/mogrifierlstm_model_checkpoint.pth \
  --best-model-path checkpoints/MogrifierLSTM/mogrifierlstm_best_model.pth \
  --max-epochs 50 \
  --plot

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
