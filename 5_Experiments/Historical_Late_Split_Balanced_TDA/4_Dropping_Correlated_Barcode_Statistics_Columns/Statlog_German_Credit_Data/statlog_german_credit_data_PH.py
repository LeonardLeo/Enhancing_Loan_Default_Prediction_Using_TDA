# -*- coding: utf-8 -*-
"""
Historical Late Split, Balanced TDA / 4_Dropping_Correlated_Barcode_Statistics_Columns
Dataset: Statlog German Credit

This experiment does not run Ripser. It drops correlated barcode-statistic columns (threshold 0.80), then trains.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

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

PROTOCOL_BUCKET = "Historical_Late_Split_Balanced_TDA"
EXPERIMENT = "4_Dropping_Correlated_Barcode_Statistics_Columns"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Statlog_German_Credit_Data"
CORR_THRESHOLD = 0.80

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
save_path = str(win_long_path(os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)))
target_dir = str(win_long_path(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "Using_Target_Variable_For_Correlation")))

data_objects = {}
dropped_payload = []
dropped_names = []

# =============================================================================
# Drop correlated columns - L30 (fit on the 80% train split of this table)
# =============================================================================
src_L30 = os.path.join(src_dir, "data_L30.csv")
table_L30 = rename_barcode_statistics_columns(pd.read_csv(src_L30))
X_L30 = table_L30.drop(columns=["label"])
y_L30 = table_L30["label"]
X_train_L30, X_test_L30, y_train_L30, y_test_L30 = train_test_split(
    X_L30, y_L30, test_size=0.2, random_state=42, stratify=y_L30
)
kept_target_L30, dropped_target_L30 = drop_correlated_features(
    X_train_L30,
    threshold=CORR_THRESHOLD,
    feature_label=True,
    strategy="target_corr",
    target=y_train_L30,
)
keep_cols_L30 = [c for c in kept_target_L30.columns if c != "label"]
data_objects["data_L30"] = {
    "data": kept_target_L30,
    "X_test": X_test_L30[keep_cols_L30],
    "y_test": y_test_L30,
}
dropped_payload.append(dropped_target_L30)
dropped_names.append("data_L30_target_drop")
os.makedirs(target_dir, exist_ok=True)
kept_target_L30.to_csv(os.path.join(target_dir, "data_L30_target.csv"), index=False)
print("Kept", len(keep_cols_L30), "columns from data_L30")

# =============================================================================
# Drop correlated columns - L60 (fit on the 80% train split of this table)
# =============================================================================
src_L60 = os.path.join(src_dir, "data_L60.csv")
table_L60 = rename_barcode_statistics_columns(pd.read_csv(src_L60))
X_L60 = table_L60.drop(columns=["label"])
y_L60 = table_L60["label"]
X_train_L60, X_test_L60, y_train_L60, y_test_L60 = train_test_split(
    X_L60, y_L60, test_size=0.2, random_state=42, stratify=y_L60
)
kept_target_L60, dropped_target_L60 = drop_correlated_features(
    X_train_L60,
    threshold=CORR_THRESHOLD,
    feature_label=True,
    strategy="target_corr",
    target=y_train_L60,
)
keep_cols_L60 = [c for c in kept_target_L60.columns if c != "label"]
data_objects["data_L60"] = {
    "data": kept_target_L60,
    "X_test": X_test_L60[keep_cols_L60],
    "y_test": y_test_L60,
}
dropped_payload.append(dropped_target_L60)
dropped_names.append("data_L60_target_drop")
os.makedirs(target_dir, exist_ok=True)
kept_target_L60.to_csv(os.path.join(target_dir, "data_L60_target.csv"), index=False)
print("Kept", len(keep_cols_L60), "columns from data_L60")

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
