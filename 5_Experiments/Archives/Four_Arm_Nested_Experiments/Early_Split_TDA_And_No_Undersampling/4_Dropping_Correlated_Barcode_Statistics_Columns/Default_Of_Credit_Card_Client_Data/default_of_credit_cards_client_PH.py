# -*- coding: utf-8 -*-
"""
Early Split TDA And No Undersampling / 4_Dropping_Correlated_Barcode_Statistics_Columns
Dataset: Default of Credit Card Client

This experiment does not run Ripser. It drops correlated barcode-statistic columns (threshold 0.80), then trains.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings

import pandas as pd

# This file lives four folders below the repository root (where utils.py is).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import (
    drop_correlated_features,
    rename_barcode_statistics_columns,
    store_data_as_csv_or_json,
    store_results,
    train_multiple_dataset_tda_drop_correlated,
    win_long_path,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

PROTOCOL_BUCKET = "Early_Split_TDA_And_No_Undersampling"
EXPERIMENT = "4_Dropping_Correlated_Barcode_Statistics_Columns"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Default_Of_Credit_Card_Client_Data"
CORR_THRESHOLD = 0.80

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
save_path = str(win_long_path(os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)))
target_dir = str(win_long_path(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "Using_Target_Variable_For_Correlation")))

data_objects = {}
dropped_payload = []
dropped_names = []

# =============================================================================
# Drop correlated columns - L5 (fit on TRAIN rows only)
# =============================================================================
train_L5 = rename_barcode_statistics_columns(pd.read_csv(os.path.join(src_dir, "train", "data_L5.csv")))
test_L5 = rename_barcode_statistics_columns(pd.read_csv(os.path.join(src_dir, "test", "data_L5.csv")))
X_train_L5 = train_L5.drop(columns=["label"])
y_train_L5 = train_L5["label"]
X_test_L5 = test_L5.drop(columns=["label"])
y_test_L5 = test_L5["label"]
kept_target_L5, dropped_target_L5 = drop_correlated_features(
    X_train_L5,
    threshold=CORR_THRESHOLD,
    feature_label=True,
    strategy="target_corr",
    target=y_train_L5,
)
keep_cols_L5 = [c for c in kept_target_L5.columns if c != "label"]
data_objects["data_L5"] = {
    "data": kept_target_L5,
    "X_test": X_test_L5[keep_cols_L5],
    "y_test": y_test_L5,
}
dropped_payload.append(dropped_target_L5)
dropped_names.append("data_L5_target_drop")
os.makedirs(target_dir, exist_ok=True)
kept_target_L5.to_csv(os.path.join(target_dir, "data_L5_target.csv"), index=False)
print("Kept", len(keep_cols_L5), "columns from data_L5")

# =============================================================================
# Drop correlated columns - L15 (fit on TRAIN rows only)
# =============================================================================
train_L15 = rename_barcode_statistics_columns(pd.read_csv(os.path.join(src_dir, "train", "data_L15.csv")))
test_L15 = rename_barcode_statistics_columns(pd.read_csv(os.path.join(src_dir, "test", "data_L15.csv")))
X_train_L15 = train_L15.drop(columns=["label"])
y_train_L15 = train_L15["label"]
X_test_L15 = test_L15.drop(columns=["label"])
y_test_L15 = test_L15["label"]
kept_target_L15, dropped_target_L15 = drop_correlated_features(
    X_train_L15,
    threshold=CORR_THRESHOLD,
    feature_label=True,
    strategy="target_corr",
    target=y_train_L15,
)
keep_cols_L15 = [c for c in kept_target_L15.columns if c != "label"]
data_objects["data_L15"] = {
    "data": kept_target_L15,
    "X_test": X_test_L15[keep_cols_L15],
    "y_test": y_test_L15,
}
dropped_payload.append(dropped_target_L15)
dropped_names.append("data_L15_target_drop")
os.makedirs(target_dir, exist_ok=True)
kept_target_L15.to_csv(os.path.join(target_dir, "data_L15_target.csv"), index=False)
print("Kept", len(keep_cols_L15), "columns from data_L15")

store_data_as_csv_or_json(path=save_path, csv=False, save_as=dropped_names, data_object=dropped_payload)

# =============================================================================
# Train models
# =============================================================================
model_results = train_multiple_dataset_tda_drop_correlated(
    data_objects=data_objects,
    test_size=0.2,
    random_state=42,
    xgb={"eval_metric": "logloss"},
)
print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
