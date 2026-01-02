#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3

python exp/Chain_exp/MogrifierLSTM.py \
  --data-dir data/Aeolus/Flight_chain/chain_data_2024 \
  --year 2024 \
  --feature-columns results/new_feature_columns.pkl \
  --checkpoint-path checkpoints/MogrifierLSTM/mogrifierlstm_model_checkpoint.pth \
  --best-model-path checkpoints/MogrifierLSTM/mogrifierlstm_best_model.pth \
  --max-epochs 1 \
  --plot
