#!/usr/bin/env bash
set -euo pipefail

pids=()

run_cmd() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" "$@" &
  pids+=("$!")
}

run_cmd 0 python exp/Tab_exp/Deep_Model_Classifier.py
run_cmd 1 python exp/Tab_exp/Deep_Model_Regressor.py
run_cmd 2 python exp/Tab_exp/Deep_Model_LSS.py

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
