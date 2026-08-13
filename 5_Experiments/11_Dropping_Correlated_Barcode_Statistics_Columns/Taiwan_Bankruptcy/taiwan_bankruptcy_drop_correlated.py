# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 2026

Dataset: Taiwanese Bankruptcy Prediction

@author: lEO
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils import (
    eda,
    drop_correlated_features,
    visualize_correlation_drop_maps,
    train_multiple_dataset_tda_drop_correlated,
    store_results,
    store_data_as_csv_or_json,
    rename_barcode_statistics_columns,
)

warnings.filterwarnings("ignore")

# =============================================================================
# GET DATASET
# =============================================================================
TDA_DIR = ROOT / "1_Data" / "TDA_Datasets" / "Taiwan_Bankruptcy" / "3_PH_Default_Parameters"
if not (TDA_DIR / "data_L10.csv").exists() or not (TDA_DIR / "data_L20.csv").exists():
    raise SystemExit(
        f"Missing Experiment 3 matrices under {TDA_DIR}. "
        "Run 5_Experiments/3_PH_Default_Parameters/Taiwan_Bankruptcy/ first."
    )
data_L10 = pd.read_csv(TDA_DIR / "data_L10.csv")
data_L20 = pd.read_csv(TDA_DIR / "data_L20.csv")

data_L10 = rename_barcode_statistics_columns(data_L10)
data_L20 = rename_barcode_statistics_columns(data_L20)

# =============================================================================
# SHUFFLE, SPLIT DATASET AND SCALE
# =============================================================================
data_L10 = data_L10.sample(frac=1, random_state=42).reset_index(drop=True)
data_L20 = data_L20.sample(frac=1, random_state=42).reset_index(drop=True)

X_data_L10 = data_L10.drop(columns="label")
X_data_L20 = data_L20.drop(columns="label")
y_data_L10 = data_L10["label"]
y_data_L20 = data_L20["label"]

X_train_data_L10, X_test_data_L10, y_train_data_L10, y_test_data_L10 = train_test_split(
    X_data_L10, y_data_L10, test_size=0.2, random_state=42, stratify=y_data_L10
)
X_train_data_L20, X_test_data_L20, y_train_data_L20, y_test_data_L20 = train_test_split(
    X_data_L20, y_data_L20, test_size=0.2, random_state=42, stratify=y_data_L20
)

scaler_data_L10 = MinMaxScaler()
scaler_data_L20 = MinMaxScaler()

X_train_data_L10 = pd.DataFrame(
    scaler_data_L10.fit_transform(X_train_data_L10), columns=scaler_data_L10.feature_names_in_
)
X_train_data_L20 = pd.DataFrame(
    scaler_data_L20.fit_transform(X_train_data_L20), columns=scaler_data_L20.feature_names_in_
)
X_test_data_L10 = pd.DataFrame(
    scaler_data_L10.transform(X_test_data_L10), columns=scaler_data_L10.feature_names_in_
)
X_test_data_L20 = pd.DataFrame(
    scaler_data_L20.transform(X_test_data_L20), columns=scaler_data_L20.feature_names_in_
)

# =============================================================================
# EDA
# =============================================================================
eda_data_L10 = eda(data_L10)
eda_data_L20 = eda(data_L20)
eda_X_train_L10 = eda(X_train_data_L10)
eda_X_train_L20 = eda(X_train_data_L20)
eda_X_test_L10 = eda(X_test_data_L10)
eda_X_test_L20 = eda(X_test_data_L20)

# =============================================================================
# DROP CORRELATED VARIABLES
# =============================================================================
data_L10_var, dropped_data_L10 = drop_correlated_features(
    X_train_data_L10,
    drop_columns=[0, 4, 8],
    threshold=0.80,
    feature_label=True,
    target=y_train_data_L10,
    strategy="high_variance",
)
data_L10_target, data_L10_dropped = drop_correlated_features(
    X_train_data_L10,
    drop_columns=[0, 4, 8],
    threshold=0.80,
    feature_label=True,
    strategy="target_corr",
    target=y_train_data_L10,
)
data_L20_var, dropped_data_L20 = drop_correlated_features(
    X_train_data_L20,
    drop_columns=[0, 4, 8],
    threshold=0.80,
    feature_label=True,
    target=y_train_data_L20,
    strategy="high_variance",
)
data_L20_target, data_L20_dropped = drop_correlated_features(
    X_train_data_L20,
    drop_columns=[0, 4, 8],
    threshold=0.80,
    feature_label=True,
    strategy="target_corr",
    target=y_train_data_L20,
)

data_L10_var_columns = data_L10_var.drop("label", axis=1).columns
data_L10_target_columns = data_L10_target.drop("label", axis=1).columns
data_L20_var_columns = data_L20_var.drop("label", axis=1).columns
data_L20_target_columns = data_L20_target.drop("label", axis=1).columns

data_L10_var_test = X_test_data_L10[data_L10_var_columns]
data_L10_target_test = X_test_data_L10[data_L10_target_columns]
data_L20_var_test = X_test_data_L20[data_L20_var_columns]
data_L20_target_test = X_test_data_L20[data_L20_target_columns]

# =============================================================================
# STORE DATAFRAMES RESULTS
# =============================================================================
save_path_var = str(ROOT / "1_Data/TDA_Datasets/Taiwan_Bankruptcy/11_Dropping_Correlated_Barcode_Statistics_Columns/Using_High_Variance_For_Correlation")
save_path_target = str(ROOT / "1_Data/TDA_Datasets/Taiwan_Bankruptcy/11_Dropping_Correlated_Barcode_Statistics_Columns/Using_Target_Variable_For_Correlation")
save_path_results = str(ROOT / "6_Results/11_Dropping_Correlated_Barcode_Statistics_Columns/Taiwan_Bankruptcy")

store_data_as_csv_or_json(
    path=save_path_var,
    csv=True,
    save_as=["data_L10_var", "data_L20_var"],
    data_object=[data_L10_var, data_L20_var],
)
store_data_as_csv_or_json(
    path=save_path_target,
    csv=True,
    save_as=["data_L10_target", "data_L20_target"],
    data_object=[data_L10_target, data_L20_target],
)
store_data_as_csv_or_json(
    path=save_path_results,
    csv=False,
    save_as=[
        "data_L10_var_drop",
        "data_L10_target_drop",
        "data_L20_var_drop",
        "data_L20_target_drop",
    ],
    data_object=[dropped_data_L10, data_L10_dropped, dropped_data_L20, data_L20_dropped],
)

# =============================================================================
# VISUALIZE DROPPED CORRELATED COLUMNS
# =============================================================================
visualize_correlation_drop_maps(
    drop_maps=[dropped_data_L10, dropped_data_L20, data_L10_dropped, data_L20_dropped],
    corr_matrices=[
        X_train_data_L10.corr().abs(),
        X_train_data_L20.corr().abs(),
        X_train_data_L10.corr().abs(),
        X_train_data_L20.corr().abs(),
    ],
    dataset_labels=[
        "Data L10 (High Variance)",
        "Data L20 (High Variance)",
        "Data L10 (Target Corr)",
        "Data L20 (Target Corr)",
    ],
    save_path=f"{save_path_results}/correlation_graphs",
    save_png=True,
    save_html=True,
    use_pyvis=True,
)

# =============================================================================
# TRAIN MACHINE LEARNING MODEL WITH DEFAULT PARAMETERS
# =============================================================================
data_to_use = {
    "data_L10": {
        "data": data_L10_target,
        "X_test": data_L10_target_test,
        "y_test": y_test_data_L10,
    },
    "data_L20": {
        "data": data_L20_target,
        "X_test": data_L20_target_test,
        "y_test": y_test_data_L20,
    },
}

model_results = train_multiple_dataset_tda_drop_correlated(
    data_objects=data_to_use,
    test_size=0.2,
    random_state=42,
    xgb={"eval_metric": "logloss"},
)

print(model_results)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
store_results(path=save_path_results, save_name="model_results", result_object=model_results)
