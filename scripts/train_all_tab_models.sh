#!/usr/bin/env bash
set -euo pipefail

INFO_YAML="processed_data/arr_delay_data_info.yaml"
MAX_EPOCHS=50

run_year() {
  local year="$1"
  local data_csv="processed_data/arr_delay_data_${year}_06_01_to_06_15.csv"

  if [[ ! -f "$data_csv" ]]; then
    echo "Skip ${year}: missing ${data_csv}"
    return
  fi

  local pids=()

  run_cmd() {
    local gpu="$1"
    shift
    CUDA_VISIBLE_DEVICES="$gpu" "$@" &
    pids+=("$!")
  }

  run_cmd 0 python exp/Tab_exp/Deep_Model_Classifier.py --data-csv "$data_csv" --info-yaml "$INFO_YAML" --max-epochs "$MAX_EPOCHS" --target "ARR_DELAY"
  # run_cmd 1 python exp/Tab_exp/Deep_Model_Classifier.py --data-csv "$data_csv" --info-yaml "$INFO_YAML" --max-epochs "$MAX_EPOCHS" --target "ARR_DELAY"
  # run_cmd 1 python exp/Tab_exp/Deep_Model_Regressor.py --data-csv "$data_csv" --info-yaml "$INFO_YAML" --max-epochs "$MAX_EPOCHS"
  run_cmd 2 python exp/Tab_exp/Deep_Model_LSS.py --data-csv "$data_csv" --info-yaml "$INFO_YAML" --max-epochs "$MAX_EPOCHS" --target "ARR_DELAY"
  # run_cmd 3 python exp/Tab_exp/Deep_Model_LSS.py --data-csv "$data_csv" --info-yaml "$INFO_YAML" --max-epochs "$MAX_EPOCHS" --target "ARR_DELAY"

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done

  return "$status"
}

status=0
for year in {2016..2024}; do
  echo "Running Tab models for ${year}..."
  if ! run_year "$year"; then
    status=1
  fi
done

exit "$status"
