python run_dataset.py extract --input data/Aeolus/Flight_Tab/flight_with_weather_2024.csv --output results/filtered_flight_data.csv --start-date 2024-01-01 --end-date 2024-12-31
python run_dataset.py pro --input results/filtered_flight_data.csv --output results/arr_delay_data.csv
python run_dataset.py tab --input-dir data/Aeolus/Flight_Tab --years 2020 2021
python run_dataset.py chain --input-dir data/Aeolus/Flight_Tab --years 2020
python run_dataset.py network --input-dir data/Aeolus/Flight_Tab --years 2020
