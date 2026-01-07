import inspect
import os
import re
import argparse
import numpy as np
import pandas as pd
import yaml
from mambular.models import AutoIntClassifier, FTTransformerClassifier, MLPClassifier, TangosClassifier, \
    TabulaRNNClassifier, SAINTClassifier, ResNetClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

parser = argparse.ArgumentParser(description="Tabular classification experiment")
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
parser.add_argument(
    "--target",
    choices=["ARR_DELAY", "DEP_DELAY"],
    default="ARR_DELAY",
    help="Target column to classify (default: ARR_DELAY)",
)
parser.add_argument(
    "--delay-threshold",
    type=float,
    default=15.0,
    help="Delay threshold in minutes for positive class (default: 15.0)",
)
parser.add_argument(
    "--delay-label",
    choices=["abs", "positive"],
    default=None,
    help="Label rule: abs uses |delay| > threshold, positive uses delay > threshold "
         "(default: abs for ARR_DELAY, positive for DEP_DELAY)",
)
parser.add_argument(
    "--results-csv",
    default=None,
    help="Output CSV path for metrics (default: Classifier_results.csv for ARR_DELAY, "
         "Classifier_results_{TARGET}.csv otherwise)",
)
args = parser.parse_args()

file_name = args.data_csv
df = pd.read_csv(file_name)
df = df.dropna()

year_match = re.search(r"(20\d{2})", file_name)
year_tag = year_match.group(1) if year_match else "all"

target_col = args.target
if target_col not in df.columns:
    raise ValueError(f"Missing target column: {target_col}")

delay_label = args.delay_label
if delay_label is None:
    delay_label = "abs" if target_col == "ARR_DELAY" else "positive"

if delay_label == "abs":
    df[target_col] = (df[target_col].abs() > args.delay_threshold).astype(int)
else:
    df[target_col] = (df[target_col] > args.delay_threshold).astype(int)

df_train = df[df['FL_DAY'] <= 9].copy()
df_valid = df[(df['FL_DAY'] > 9) & (df['FL_DAY'] <= 12)].copy()
df_test = df[df['FL_DAY'] > 12].copy()

with open(args.info_yaml, 'r') as yaml_file:
    data_info = yaml.load(yaml_file, Loader=yaml.FullLoader)

categorical_columns = data_info['columns_info']['Categorical Features']
continuous_columns = data_info['columns_info']['Continuous Features']

continuous_columns = [col for col in continuous_columns if col != 'FLIGHTS']
categorical_columns = [col for col in categorical_columns if col != 'FL_YEAR' and col != 'FL_MONTH']
continuous_columns = [col for col in continuous_columns if col not in ['ARR_DELAY', 'DEP_DELAY']]
categorical_columns = [col for col in categorical_columns if col not in ['ARR_DELAY', 'DEP_DELAY']]

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_train.loc[:, categorical_columns] = encoder.fit_transform(df_train[categorical_columns])
df_valid.loc[:, categorical_columns] = encoder.transform(df_valid[categorical_columns])
df_test.loc[:, categorical_columns] = encoder.transform(df_test[categorical_columns])

for df_split in (df_train, df_valid, df_test):
    # Shift ids to keep categories non-negative (0 reserved for unseen values).
    df_split.loc[:, categorical_columns] = df_split[categorical_columns].astype("int64") + 1

scaler = StandardScaler()
df_train.loc[:, continuous_columns] = scaler.fit_transform(df_train[continuous_columns])
df_valid.loc[:, continuous_columns] = scaler.transform(df_valid[continuous_columns])
df_test.loc[:, continuous_columns] = scaler.transform(df_test[continuous_columns])

X_train = df_train[categorical_columns + continuous_columns]
X_valid = df_valid[categorical_columns + continuous_columns]
X_test = df_test[categorical_columns + continuous_columns]

y_train = df_train[target_col]
y_valid = df_valid[target_col]
y_test = df_test[target_col]


models = {
    'MLP': MLPClassifier(d_model=128),
    'AutoInt': AutoIntClassifier(d_model=128, n_layers=4),
    'ResNet': ResNetClassifier(),
    'FTTransformer': FTTransformerClassifier(d_model=128, n_layers=4),
    'Tangos': TangosClassifier(d_model=128),
    'TabulaRNN': TabulaRNNClassifier(d_model=128),
    'SAINT': SAINTClassifier(d_model=128),
}

results_df = pd.DataFrame(columns=['Model', 'AUC', 'ACC'])

param_dist = {
    'd_model': [64, 128, 256],
    'n_layers': [2, 6, 10],
    'lr': [1e-5, 1e-4, 1e-3]
}

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
    checkpoint_dir = os.path.join("checkpoints", "Tab_exp", "Classifier", target_col, model_name, year_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    fit_params["checkpoint_path"] = checkpoint_dir
    model_param_dist = filter_param_dist(base_model, param_dist)
    if model_param_dist:
        param_iter = ParameterSampler(model_param_dist, n_iter=n_iter, random_state=42)
    else:
        param_iter = [dict()]

    best_model = None
    best_params = None
    best_val_acc = -np.inf

    for params in param_iter:
        model = clone(base_model)
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train, **fit_params)
        val_pred = model.predict(X_valid)
        val_acc = accuracy_score(y_valid, val_pred)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            best_model = model

    print("Best Parameters:", best_params)
    print("Best Val ACC:", best_val_acc)

    y_pred = best_model.predict(X_test)
    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(X_test)
        if proba.ndim == 1 or proba.shape[1] == 1:
            y_score = np.ravel(proba)
        else:
            y_score = proba[:, 1]
        auc = roc_auc_score(y_test, y_score)
    elif hasattr(best_model, "decision_function"):
        y_score = best_model.decision_function(X_test)
        auc = roc_auc_score(y_test, y_score)
    else:
        auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} - AUC: {auc}, ACC: {acc}")

    result = pd.DataFrame({'Model': [model_name], 'AUC': [auc], 'ACC': [acc]})
    results_df = pd.concat([results_df, result], ignore_index=True)

results_path = args.results_csv
if results_path is None:
    if target_col == "ARR_DELAY":
        results_path = "Classifier_results.csv"
    else:
        results_path = f"Classifier_results_{target_col}.csv"
results_df.to_csv(results_path, index=False)
print(f"Saved results to {results_path}")

print('end')
