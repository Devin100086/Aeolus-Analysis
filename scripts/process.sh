#!/usr/bin/env bash
set -euo pipefail

# Process Arrival Delay Data for each year in parallel
pids=()

run_cmd() {
  "$@" &
  pids+=("$!")
}

run_cmd python util/utils/run_dataset.py pro --input processed_data/filtered_flight_data_2024.csv --output processed_data/arr_delay_data_2024.csv
run_cmd python util/utils/run_dataset.py pro --input processed_data/filtered_flight_data_2023.csv --output processed_data/arr_delay_data_2023.csv
run_cmd python util/utils/run_dataset.py pro --input processed_data/filtered_flight_data_2022.csv --output processed_data/arr_delay_data_2022.csv
run_cmd python util/utils/run_dataset.py pro --input processed_data/filtered_flight_data_2021.csv --output processed_data/arr_delay_data_2021.csv
run_cmd python util/utils/run_dataset.py pro --input processed_data/filtered_flight_data_2020.csv --output processed_data/arr_delay_data_2020.csv
run_cmd python util/utils/run_dataset.py pro --input processed_data/filtered_flight_data_2019.csv --output processed_data/arr_delay_data_2019.csv
run_cmd python util/utils/run_dataset.py pro --input processed_data/filtered_flight_data_2018.csv --output processed_data/arr_delay_data_2018.csv
run_cmd python util/utils/run_dataset.py pro --input processed_data/filtered_flight_data_2017.csv --output processed_data/arr_delay_data_2017.csv
run_cmd python util/utils/run_dataset.py pro --input processed_data/filtered_flight_data_2016.csv --output processed_data/arr_delay_data_2016.csv

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
