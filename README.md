# Aeolus-Analysis

An experimental project for flight delay analysis and prediction. It builds three modeling pipelines from flight and weather data: tabular data, flight-chain sequences, and flight-network graphs.

## Overview
- Data extraction and feature processing (by year or date range)
- Tabular models: AutoInt/FTTransformer/MLP/SAINT (mambular)
- Flight-chain models: LSTM/GRU/CNN-LSTM/MogrifierLSTM
- Flight-network models: Dynamic/Temporal GNN, VGAE + AFM embedding

## Repository Layout
- `data/Aeolus/Flight_Tab`: raw annual CSVs; tabular outputs go to `Tab/`
- `data/Aeolus/Flight_chain`: flight-chain datasets (`.pt`)
- `data/Aeolus/Flight_network`: daily flight graphs (`.dgl`)
- `Datasets`: dataset processing scripts (tab/chain/network)
- `exp`: training scripts (`Tab_exp`/`Chain_exp`/`Network_exp`)
- `util/utils`: dataset entrypoint and feature-column generation
- `scripts`: one-click processing and training
- `processed_data`: filtered/feature data and YAML descriptions
- `checkpoints`, `model_checkpoints`, `lightning_logs`: training outputs

## Requirements
- Python + PyTorch
- pandas / numpy / scikit-learn / PyYAML / tqdm
- DGL (graph models)
- mambular (tabular models)

## Data Preparation
1. Place raw data at:
   `data/Aeolus/Flight_Tab/flight_with_weather_YYYY.csv`
2. One-click processing (extract + featurize + generate tab/chain/network data):
```bash
bash scripts/run_dataset.sh
```
3. Run individually (adjust year/date as needed):
```bash
python util/utils/run_dataset.py extract \
  --input data/Aeolus/Flight_Tab/flight_with_weather_2024.csv \
  --output processed_data/filtered_flight_data_2024.csv \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

python util/utils/run_dataset.py pro \
  --input processed_data/filtered_flight_data_2024.csv \
  --output processed_data/arr_delay_data_2024.csv

python util/utils/run_dataset.py tab --input-dir data/Aeolus/Flight_Tab --years 2024
python util/utils/run_dataset.py chain --input-dir data/Aeolus/Flight_Tab --years 2024
python util/utils/run_dataset.py network --input-dir data/Aeolus/Flight_Tab --years 2024
```
4. Field metadata and summary:
   `processed_data/arr_delay_data_info.yaml`

### Flight-Chain Feature Columns
Chain models require `new_feature_columns.pkl`. Generate it with:
```bash
python util/utils/generate_new_feature_columns.py \
  --data-dir data/Aeolus/Flight_chain/chain_data_2023 \
  --year 2023 \
  --output processed_data/2023/new_feature_columns.pkl
```

## Training Examples
### Tabular Models
```bash
bash scripts/train_2024_tab_models.sh
```
Or run directly:
```bash
python exp/Tab_exp/Deep_Model_Classifier.py \
  --data-csv processed_data/arr_delay_data_2024_06_01_to_06_15.csv \
  --info-yaml processed_data/arr_delay_data_info.yaml \
  --max-epochs 50 \
  --target ARR_DELAY
```

### Flight-Chain Models
```bash
bash scripts/train_LSTM.sh
```
Or run directly:
```bash
python exp/Chain_exp/LSTM.py \
  --data-dir data/Aeolus/Flight_chain/chain_data_2024 \
  --year 2024 \
  --feature-columns processed_data/new_feature_columns.pkl
```

### Flight-Network Models
```bash
bash scripts/train_graph.sh
bash scripts/train_temporal_graph.sh
```
VGAE + AFM:
```bash
bash scripts/train_vgae_node_embedding.sh
bash scripts/train_AFM_node_embedding.sh
```

## Results and Logs
- Model checkpoints: `checkpoints/`, `model_checkpoints/`
- Processed data/features: `processed_data/`
- Training logs: `lightning_logs/` (if used)

## Notes
- Chain dataset filenames must match scripts: `train_flight_chain_YYYY.pt`/`val_flight_chain_YYYY.pt`/`test_flight_chain_YYYY.pt`.
- Most scripts default to GPU usage and fixed year/date ranges; adjust via script or CLI args.
- `scripts/train_2024_tab_models.sh`/`scripts/train_all_tab_models.sh` expect `processed_data/arr_delay_data_YYYY_06_01_to_06_15.csv`; run `extract` with that date window first.
