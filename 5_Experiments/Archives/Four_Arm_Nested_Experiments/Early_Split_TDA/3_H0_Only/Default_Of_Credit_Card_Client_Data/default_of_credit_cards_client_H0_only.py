# -*- coding: utf-8 -*-
"""
Early Split TDA / 3_H0_Only
Dataset: Default of Credit Card Client

This experiment does not run Ripser. It keeps only H0 (dimension-0) barcode columns, then trains.
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
    store_results,
    train_multiple_dataset_tda_presplit,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

PROTOCOL_BUCKET = "Early_Split_TDA"
EXPERIMENT = "3_H0_Only"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Default_Of_Credit_Card_Client_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)

# =============================================================================
# Get Data - L5 TRAIN and TEST (keep only H0 / dimension-0 columns)
# =============================================================================
src_train_L5 = os.path.join(src_dir, "train", "data_L5.csv")
src_test_L5 = os.path.join(src_dir, "test", "data_L5.csv")
dest_train_L5 = os.path.join(dest_dir, "train", "data_L5.csv")
dest_test_L5 = os.path.join(dest_dir, "test", "data_L5.csv")
os.makedirs(os.path.dirname(dest_train_L5), exist_ok=True)
os.makedirs(os.path.dirname(dest_test_L5), exist_ok=True)

data_train_L5 = pd.read_csv(src_train_L5)
h0_train_L5 = [c for c in data_train_L5.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_train_L5 = data_train_L5[h0_train_L5]
data_train_L5.to_csv(dest_train_L5, index=False)

data_test_L5 = pd.read_csv(src_test_L5)
h0_test_L5 = [c for c in data_test_L5.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_test_L5 = data_test_L5[h0_test_L5]
data_test_L5.to_csv(dest_test_L5, index=False)
print("L5 train H0 columns:", list(data_train_L5.columns))

# =============================================================================
# Get Data - L15 TRAIN and TEST (keep only H0 / dimension-0 columns)
# =============================================================================
src_train_L15 = os.path.join(src_dir, "train", "data_L15.csv")
src_test_L15 = os.path.join(src_dir, "test", "data_L15.csv")
dest_train_L15 = os.path.join(dest_dir, "train", "data_L15.csv")
dest_test_L15 = os.path.join(dest_dir, "test", "data_L15.csv")
os.makedirs(os.path.dirname(dest_train_L15), exist_ok=True)
os.makedirs(os.path.dirname(dest_test_L15), exist_ok=True)

data_train_L15 = pd.read_csv(src_train_L15)
h0_train_L15 = [c for c in data_train_L15.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_train_L15 = data_train_L15[h0_train_L15]
data_train_L15.to_csv(dest_train_L15, index=False)

data_test_L15 = pd.read_csv(src_test_L15)
h0_test_L15 = [c for c in data_test_L15.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_test_L15 = data_test_L15[h0_test_L15]
data_test_L15.to_csv(dest_test_L15, index=False)
print("L15 train H0 columns:", list(data_train_L15.columns))

# =============================================================================
# Train models on the H0 tables
# =============================================================================
pairs = {
    "data_L5": {
        "train": os.path.join(dest_dir, "train", "data_L5.csv"),
        "test": os.path.join(dest_dir, "test", "data_L5.csv"),
    },
    "data_L15": {
        "train": os.path.join(dest_dir, "train", "data_L15.csv"),
        "test": os.path.join(dest_dir, "test", "data_L15.csv"),
    },
}
model_results = train_multiple_dataset_tda_presplit(
    train_test_pairs=pairs,
    y_col_name="label",
    random_state=42,
    xgb={"eval_metric": "logloss"},
)

print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
