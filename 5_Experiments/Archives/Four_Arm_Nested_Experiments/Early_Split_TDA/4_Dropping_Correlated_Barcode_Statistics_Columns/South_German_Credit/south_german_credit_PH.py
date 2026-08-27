# -*- coding: utf-8 -*-
"""
Early Split TDA / 4_Dropping_Correlated_Barcode_Statistics_Columns
Dataset: South German Credit

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

PROTOCOL_BUCKET = "Early_Split_TDA"
EXPERIMENT = "4_Dropping_Correlated_Barcode_Statistics_Columns"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "South_German_Credit"
CORR_THRESHOLD = 0.80

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
save_path = str(win_long_path(os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)))
target_dir = str(win_long_path(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "Using_Target_Variable_For_Correlation")))

data_objects = {}
dropped_payload = []
dropped_names = []

# =============================================================================
# Drop correlated columns - L10 (fit on TRAIN rows only)
# =============================================================================
train_L10 = rename_barcode_statistics_columns(pd.read_csv(os.path.join(src_dir, "train", "data_L10.csv")))
test_L10 = rename_barcode_statistics_columns(pd.read_csv(os.path.join(src_dir, "test", "data_L10.csv")))
X_train_L10 = train_L10.drop(columns=["label"])
y_train_L10 = train_L10["label"]
X_test_L10 = test_L10.drop(columns=["label"])
y_test_L10 = test_L10["label"]
kept_target_L10, dropped_target_L10 = drop_correlated_features(
    X_train_L10,
    threshold=CORR_THRESHOLD,
    feature_label=True,
    strategy="target_corr",
    target=y_train_L10,
)
keep_cols_L10 = [c for c in kept_target_L10.columns if c != "label"]
data_objects["data_L10"] = {
    "data": kept_target_L10,
    "X_test": X_test_L10[keep_cols_L10],
    "y_test": y_test_L10,
}
dropped_payload.append(dropped_target_L10)
dropped_names.append("data_L10_target_drop")
os.makedirs(target_dir, exist_ok=True)
kept_target_L10.to_csv(os.path.join(target_dir, "data_L10_target.csv"), index=False)
print("Kept", len(keep_cols_L10), "columns from data_L10")

# =============================================================================
# Drop correlated columns - L20 (fit on TRAIN rows only)
# =============================================================================
train_L20 = rename_barcode_statistics_columns(pd.read_csv(os.path.join(src_dir, "train", "data_L20.csv")))
test_L20 = rename_barcode_statistics_columns(pd.read_csv(os.path.join(src_dir, "test", "data_L20.csv")))
X_train_L20 = train_L20.drop(columns=["label"])
y_train_L20 = train_L20["label"]
X_test_L20 = test_L20.drop(columns=["label"])
y_test_L20 = test_L20["label"]
kept_target_L20, dropped_target_L20 = drop_correlated_features(
    X_train_L20,
    threshold=CORR_THRESHOLD,
    feature_label=True,
    strategy="target_corr",
    target=y_train_L20,
)
keep_cols_L20 = [c for c in kept_target_L20.columns if c != "label"]
data_objects["data_L20"] = {
    "data": kept_target_L20,
    "X_test": X_test_L20[keep_cols_L20],
    "y_test": y_test_L20,
}
dropped_payload.append(dropped_target_L20)
dropped_names.append("data_L20_target_drop")
os.makedirs(target_dir, exist_ok=True)
kept_target_L20.to_csv(os.path.join(target_dir, "data_L20_target.csv"), index=False)
print("Kept", len(keep_cols_L20), "columns from data_L20")

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
