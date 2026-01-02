import inspect
import os
import re
import argparse
import numpy as np
import pandas as pd
import yaml
from mambular.models import AutoIntRegressor, FTTransformerRegressor, MLPRegressor, TangosRegressor, \
    TabulaRNNRegressor, SAINTRegressor, ResNetRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

parser = argparse.ArgumentParser(description="Tabular regression experiment")
parser.add_argument(
    "--max-epochs",
    type=int,
    default=1,
    help="Training epochs per trial (default: 1)",
)
args = parser.parse_args()

file_name = '/data0/ps/wucunqi/homework/Aeolus-Analysis/data/Aeolus/Flight_Tab/Tab/Flight_tab_2021.csv'
df = pd.read_csv(file_name)
year_match = re.search(r"(20\d{2})", file_name)
year_tag = year_match.group(1) if year_match else "all"

with open('/data0/ps/wucunqi/homework/Aeolus-Analysis/Datasets/arr_delay_data_info.yaml', 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

continuous_columns = [col for col in continuous_columns if col not in ['FLIGHTS', 'DEP_DELAY']]
categorical_columns = [col for col in categorical_columns if col != 'FL_YEAR' and col != 'FL_MONTH']

# Drop columns missing entirely to avoid downstream preprocessors failing.
all_candidate_cols = [col for col in categorical_columns + continuous_columns if col in df.columns]
all_nan_cols = [col for col in all_candidate_cols if df[col].isna().all()]
if all_nan_cols:
    df = df.drop(columns=all_nan_cols)
    categorical_columns = [col for col in categorical_columns if col not in all_nan_cols]
    continuous_columns = [col for col in continuous_columns if col not in all_nan_cols]

categorical_columns = [col for col in categorical_columns if col in df.columns]
continuous_columns = [col for col in continuous_columns if col in df.columns]

df = df[df['DEP_DELAY'].notna()].copy()
df = df[df['FL_DAY'].notna()].copy()
df.loc[:, categorical_columns] = df[categorical_columns].fillna("UNK")
df.loc[:, continuous_columns] = df[continuous_columns].fillna(df[continuous_columns].median(numeric_only=True))

df_train = df[df['FL_DAY'] <= 9].copy()
df_valid = df[(df['FL_DAY'] > 9) & (df['FL_DAY'] <= 12)].copy()
df_test = df[df['FL_DAY'] > 12].copy()

if categorical_columns:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_train.loc[:, categorical_columns] = encoder.fit_transform(df_train[categorical_columns])
    df_valid.loc[:, categorical_columns] = encoder.transform(df_valid[categorical_columns])
    df_test.loc[:, categorical_columns] = encoder.transform(df_test[categorical_columns])

    for df_split in (df_train, df_valid, df_test):
        # Shift ids to keep categories non-negative (0 reserved for unseen values).
        df_split.loc[:, categorical_columns] = df_split[categorical_columns].astype("int64") + 1

if continuous_columns:
    scaler = StandardScaler()
    df_train.loc[:, continuous_columns] = scaler.fit_transform(df_train[continuous_columns])
    df_valid.loc[:, continuous_columns] = scaler.transform(df_valid[continuous_columns])
    df_test.loc[:, continuous_columns] = scaler.transform(df_test[continuous_columns])

X_train = df_train[categorical_columns + continuous_columns]
X_valid = df_valid[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train['DEP_DELAY']
y_valid = df_valid['DEP_DELAY']
y_test = df_test['DEP_DELAY']


models = {
    "AutoInt": AutoIntRegressor(d_model=64, n_layers=8),
    "FTTransformer": FTTransformerRegressor(d_model=64, n_layers=8),
    "MLP": MLPRegressor(d_model=64),
    "Tangos": TangosRegressor(d_model=64),
    'TabulaRNN': TabulaRNNRegressor(d_model=64),
    'SAINT': SAINTRegressor(d_model=64),
    'ResNet': ResNetRegressor(),
}

param_dist = {
    'd_model': [64, 128, 256],
    'n_layers': [2, 6, 10],
    'lr': [1e-5, 1e-4, 1e-3]
}

results_df = pd.DataFrame(columns=['Model', 'MSE', 'MAE'])

def filter_param_dist(model, full_param_dist):
    try:
        model_params = set(model.get_params().keys())
    except Exception:
        try:
            sig = inspect.signature(model.__class__.__init__)
            model_params = set(sig.parameters.keys())
            model_params.discard("self")
        except Exception:
            return {}
    return {k: v for k, v in full_param_dist.items() if k in model_params}


n_iter = 5
fit_params = {
    "max_epochs": args.max_epochs,
    "rebuild": True,
    "X_val": X_valid,
    "y_val": y_valid,
    "patience": 5,
    "devices": 1,
    "accelerator": "auto",
}

for model_name, base_model in models.items():
    print(f"Searching params for {model_name}...")
    checkpoint_dir = os.path.join("checkpoints", "Tab_exp", "Regressor", model_name, year_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    fit_params["checkpoint_path"] = checkpoint_dir
    model_param_dist = filter_param_dist(base_model, param_dist)
    if model_param_dist:
        param_iter = ParameterSampler(model_param_dist, n_iter=n_iter, random_state=42)
    else:
        param_iter = [dict()]

    best_model = None
    best_params = None
    best_val_mse = np.inf

    for params in param_iter:
        model = clone(base_model)
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train, **fit_params)
        val_pred = np.ravel(model.predict(X_valid))
        val_mse = mean_squared_error(y_valid, val_pred)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = params
            best_model = model

    print("Best Parameters:", best_params)
    print("Best Val MSE:", best_val_mse)

    y_pred = np.ravel(best_model.predict(X_test))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"{model_name} - MSE: {mse}, MAE: {mae}")

    result = pd.DataFrame({'Model': [model_name], 'MSE': [mse], 'MAE': [mae]})
    results_df = pd.concat([results_df, result], ignore_index=True)

results_df.to_csv('Regressor_results_DEP.csv', index=False)

print('end')
