#!/usr/bin/env bash
set -euo pipefail

# Extract relevant data for each year in parallel
pids=()

run_cmd() {
  "$@" &
  pids+=("$!")
}

run_cmd python util/utils/run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2024.csv --output processed_data/filtered_flight_data_2024.csv --start-date 2024-01-01 --end-date 2024-12-31
run_cmd python util/utils/run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2023.csv --output processed_data/filtered_flight_data_2023.csv --start-date 2023-01-01 --end-date 2023-12-31
run_cmd python util/utils/run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2022.csv --output processed_data/filtered_flight_data_2022.csv --start-date 2022-01-01 --end-date 2022-12-31
run_cmd python util/utils/run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2021.csv --output processed_data/filtered_flight_data_2021.csv --start-date 2021-01-01 --end-date 2021-12-31
run_cmd python util/utils/run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2020.csv --output processed_data/filtered_flight_data_2020.csv --start-date 2020-01-01 --end-date 2020-12-31
run_cmd python util/utils/run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2019.csv --output processed_data/filtered_flight_data_2019.csv --start-date 2019-01-01 --end-date 2019-12-31
run_cmd python util/utils/run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2018.csv --output processed_data/filtered_flight_data_2018.csv --start-date 2018-01-01 --end-date 2018-12-31
run_cmd python util/utils/run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2017.csv --output processed_data/filtered_flight_data_2017.csv --start-date 2017-01-01 --end-date 2017-12-31
run_cmd python util/utils/run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2016.csv --output processed_data/filtered_flight_data_2016.csv --start-date 2016-01-01 --end-date 2016-12-31

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
