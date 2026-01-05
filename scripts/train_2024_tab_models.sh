#!/usr/bin/env bash
set -euo pipefail

INFO_YAML="processed_data/arr_delay_data_info.yaml"
MAX_EPOCHS=50
data_csv="processed_data/arr_delay_data_2024_06_01_to_06_15.csv"

run_cmd() {
local gpu="$1"
shift
CUDA_VISIBLE_DEVICES="$gpu" "$@" &
pids+=("$!")
}

run_cmd 0 python exp/Tab_exp/Deep_Model_Classifier.py --data-csv "$data_csv" --info-yaml "$INFO_YAML" --max-epochs "$MAX_EPOCHS"
run_cmd 1 python exp/Tab_exp/Deep_Model_Regressor.py --data-csv "$data_csv" --info-yaml "$INFO_YAML" --max-epochs "$MAX_EPOCHS"
run_cmd 2 python exp/Tab_exp/Deep_Model_LSS.py --data-csv "$data_csv" --info-yaml "$INFO_YAML" --max-epochs "$MAX_EPOCHS"


# python exp/Tab_exp/Deep_Model_Regressor.py --data-csv "processed_data/arr_delay_data_2024_06_01_to_06_15.csv" --info-yaml "processed_data/arr_delay_data_info.yaml" --max-epochs 50 --target-scale standard