import inspect
import os
import re
import argparse
import numpy as np
import pandas as pd
import yaml
from mambular.models import AutoIntLSS, FTTransformerLSS, MLPLSS, TangosLSS, TabulaRNNLSS, SAINTLSS, ResNetLSS
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

parser = argparse.ArgumentParser(description="Tabular LSS experiment")
parser.add_argument(
    "--data-csv",
    default="results/arr_delay_data.csv",
    help="Path to input CSV (default: results/arr_delay_data.csv)",
)
parser.add_argument(
    "--info-yaml",
    default="results/arr_delay_data_info.yaml",
    help="Path to feature info YAML (default: results/arr_delay_data_info.yaml)",
)
parser.add_argument(
    "--max-epochs",
    type=int,
    default=1,
    help="Training epochs per trial (default: 1)",
)
args = parser.parse_args()

file_name = args.data_csv
df = pd.read_csv(file_name)
df = df.dropna()

year_match = re.search(r"(20\d{2})", file_name)
year_tag = year_match.group(1) if year_match else "all"

with open(args.info_yaml, 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

continuous_columns = [col for col in continuous_columns if col not in ['FLIGHTS', 'DEP_DELAY']]
categorical_columns = [col for col in categorical_columns if col != 'FL_YEAR' and col != 'FL_MONTH']

df_train = df[df['FL_DAY'] <= 9].copy()
df_valid = df[(df['FL_DAY'] > 9) & (df['FL_DAY'] <= 12)].copy()
df_test = df[df['FL_DAY'] > 12].copy()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_train.loc[:, categorical_columns] = encoder.fit_transform(df_train[categorical_columns])
df_valid.loc[:, categorical_columns] = encoder.transform(df_valid[categorical_columns])
df_test.loc[:, categorical_columns] = encoder.transform(df_test[categorical_columns])

for df_split in (df_train, df_valid, df_test):
    # Shift ids to keep categories non-negative (0 reserved for unseen values).
    df_split.loc[:, categorical_columns] = df_split[categorical_columns].astype("int64") + 1

feature_scaler = StandardScaler()
df_train.loc[:, continuous_columns] = feature_scaler.fit_transform(df_train[continuous_columns])
df_valid.loc[:, continuous_columns] = feature_scaler.transform(df_valid[continuous_columns])
df_test.loc[:, continuous_columns] = feature_scaler.transform(df_test[continuous_columns])

target_scaler = StandardScaler()
df_train.loc[:, ['DEP_DELAY']] = target_scaler.fit_transform(df_train[['DEP_DELAY']])
df_valid.loc[:, ['DEP_DELAY']] = target_scaler.transform(df_valid[['DEP_DELAY']])
df_test.loc[:, ['DEP_DELAY']] = target_scaler.transform(df_test[['DEP_DELAY']])

X_train = df_train[categorical_columns + continuous_columns]
X_valid = df_valid[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train['DEP_DELAY']
y_valid = df_valid['DEP_DELAY']
y_test = df_test['DEP_DELAY']


models = {
    "AutoInt": AutoIntLSS(d_model=64, n_layers=8),
    "FTTransformer": FTTransformerLSS(d_model=64, n_layers=8),
    "MLP": MLPLSS(d_model=64),
    "Tangos": TangosLSS(d_model=64),
    'TabulaRNN': TabulaRNNLSS(d_model=64),
    'SAINT': SAINTLSS(d_model=64),
    'ResNet': ResNetLSS(),
}

param_dist = {
    'd_model': [64, 128, 256],
    'n_layers': [2, 6, 10],
    'lr': [1e-5, 1e-4, 1e-3]
}

results_df = pd.DataFrame(columns=['Model', 'NLL', 'CRPS'])

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
family = "normal"
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
    checkpoint_dir = os.path.join("checkpoints", "Tab_exp", "LSS", model_name, year_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    fit_params["checkpoint_path"] = checkpoint_dir
    model_param_dist = filter_param_dist(base_model, param_dist)
    if model_param_dist:
        param_iter = ParameterSampler(model_param_dist, n_iter=n_iter, random_state=42)
    else:
        param_iter = [dict()]

    best_model = None
    best_params = None
    best_val_nll = np.inf

    for params in param_iter:
        model = clone(base_model)
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train, family=family, **fit_params)
        val_pred = model.predict(X_valid)
        val_pred_mean = val_pred[:, 0]
        val_pred_std = np.maximum(val_pred[:, 1], 1e-6)
        val_delta = y_valid - val_pred_mean
        val_sigma_sq = val_pred_std ** 2
        val_nll = 0.5 * np.mean(val_delta ** 2 / val_sigma_sq + np.log(val_sigma_sq) + np.log(2 * np.pi))
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_params = params
            best_model = model

    print("Best Parameters:", best_params)
    print("Best Val NLL:", best_val_nll)

    y_pred = best_model.predict(X_test)
    y_pred_mean = y_pred[:, 0]
    y_pred_std = np.maximum(y_pred[:, 1], 1e-6)

    delta = y_test - y_pred_mean
    sigma_sq = y_pred_std ** 2
    nll = 0.5 * np.mean(delta ** 2 / sigma_sq + np.log(sigma_sq) + np.log(2 * np.pi))
    print(f"NLL (calculated): {nll}")

    from scipy.special import erf
    z = delta / y_pred_std
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
    crps_values = y_pred_std * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    crps = crps_values.mean()
    print(f"CRPS: {crps}")

    print(f"{model_name} - NLL: {nll}, CRPS: {crps}")

    result = pd.DataFrame({'Model': [model_name], 'NLL': [nll], 'CRPS': [crps]})
    results_df = pd.concat([results_df, result], ignore_index=True)

results_df.to_csv('LSS_results_DEP.csv', index=False)

print('end')
