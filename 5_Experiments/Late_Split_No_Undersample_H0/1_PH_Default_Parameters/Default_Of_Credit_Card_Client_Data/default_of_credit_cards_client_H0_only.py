# -*- coding: utf-8 -*-
"""
Late split, no undersample, using just H0 / 1_PH_Default_Parameters
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
    barcode_source_bucket,
    store_results,
    train_multiple_dataset_tda,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

PROTOCOL_BUCKET = "Late_Split_No_Undersample_H0"
SOURCE_BUCKET = barcode_source_bucket(PROTOCOL_BUCKET)
EXPERIMENT = "1_PH_Default_Parameters"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Default_Of_Credit_Card_Client_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", SOURCE_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)

# =============================================================================
# Get Data - L5  (keep only H0 / dimension-0 columns)
# =============================================================================
src_L5 = os.path.join(src_dir, "data_L5.csv")
dest_L5 = os.path.join(dest_dir, "data_L5.csv")
data_L5 = pd.read_csv(src_L5)
h0_columns_L5 = [c for c in data_L5.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L5 = data_L5[h0_columns_L5]
os.makedirs(dest_dir, exist_ok=True)
data_L5.to_csv(dest_L5, index=False)
print("L5 H0 columns:", list(data_L5.columns))

# =============================================================================
# Get Data - L15  (keep only H0 / dimension-0 columns)
# =============================================================================
src_L15 = os.path.join(src_dir, "data_L15.csv")
dest_L15 = os.path.join(dest_dir, "data_L15.csv")
data_L15 = pd.read_csv(src_L15)
h0_columns_L15 = [c for c in data_L15.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L15 = data_L15[h0_columns_L15]
os.makedirs(dest_dir, exist_ok=True)
data_L15.to_csv(dest_L15, index=False)
print("L15 H0 columns:", list(data_L15.columns))

# =============================================================================
# Train models on the H0 tables
# =============================================================================
paths = [os.path.join(dest_dir, "data_L5.csv"), os.path.join(dest_dir, "data_L15.csv")]
model_results = train_multiple_dataset_tda(
    path_datasets=paths,
    y_col_name="label",
    test_size=0.2,
    random_state=42,
    xgb={"eval_metric": "logloss"},
)

print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
