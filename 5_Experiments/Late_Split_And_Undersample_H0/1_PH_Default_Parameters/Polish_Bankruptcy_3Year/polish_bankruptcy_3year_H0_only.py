# -*- coding: utf-8 -*-
"""
Late split and undersample (the original historical run), using just H0 / 1_PH_Default_Parameters
Dataset: Polish Companies Bankruptcy (3 year)

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

PROTOCOL_BUCKET = "Late_Split_And_Undersample_H0"
SOURCE_BUCKET = barcode_source_bucket(PROTOCOL_BUCKET)
EXPERIMENT = "1_PH_Default_Parameters"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Polish_Bankruptcy_3Year"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", SOURCE_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)

# =============================================================================
# Get Data - L10  (keep only H0 / dimension-0 columns)
# =============================================================================
src_L10 = os.path.join(src_dir, "data_L10.csv")
dest_L10 = os.path.join(dest_dir, "data_L10.csv")
data_L10 = pd.read_csv(src_L10)
h0_columns_L10 = [c for c in data_L10.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L10 = data_L10[h0_columns_L10]
os.makedirs(dest_dir, exist_ok=True)
data_L10.to_csv(dest_L10, index=False)
print("L10 H0 columns:", list(data_L10.columns))

# =============================================================================
# Get Data - L20  (keep only H0 / dimension-0 columns)
# =============================================================================
src_L20 = os.path.join(src_dir, "data_L20.csv")
dest_L20 = os.path.join(dest_dir, "data_L20.csv")
data_L20 = pd.read_csv(src_L20)
h0_columns_L20 = [c for c in data_L20.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L20 = data_L20[h0_columns_L20]
os.makedirs(dest_dir, exist_ok=True)
data_L20.to_csv(dest_L20, index=False)
print("L20 H0 columns:", list(data_L20.columns))

# =============================================================================
# Train models on the H0 tables
# =============================================================================
paths = [os.path.join(dest_dir, "data_L10.csv"), os.path.join(dest_dir, "data_L20.csv")]
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
