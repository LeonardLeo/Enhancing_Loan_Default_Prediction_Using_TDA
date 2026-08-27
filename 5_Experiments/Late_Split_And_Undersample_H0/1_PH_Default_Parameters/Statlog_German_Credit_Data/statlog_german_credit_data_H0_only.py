# -*- coding: utf-8 -*-
"""
Late split and undersample (the original historical run), using just H0 / 1_PH_Default_Parameters
Dataset: Statlog German Credit

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
FOLDER = "Statlog_German_Credit_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", SOURCE_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)

# =============================================================================
# Get Data - L30  (keep only H0 / dimension-0 columns)
# =============================================================================
src_L30 = os.path.join(src_dir, "data_L30.csv")
dest_L30 = os.path.join(dest_dir, "data_L30.csv")
data_L30 = pd.read_csv(src_L30)
h0_columns_L30 = [c for c in data_L30.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L30 = data_L30[h0_columns_L30]
os.makedirs(dest_dir, exist_ok=True)
data_L30.to_csv(dest_L30, index=False)
print("L30 H0 columns:", list(data_L30.columns))

# =============================================================================
# Get Data - L60  (keep only H0 / dimension-0 columns)
# =============================================================================
src_L60 = os.path.join(src_dir, "data_L60.csv")
dest_L60 = os.path.join(dest_dir, "data_L60.csv")
data_L60 = pd.read_csv(src_L60)
h0_columns_L60 = [c for c in data_L60.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L60 = data_L60[h0_columns_L60]
os.makedirs(dest_dir, exist_ok=True)
data_L60.to_csv(dest_L60, index=False)
print("L60 H0 columns:", list(data_L60.columns))

# =============================================================================
# Train models on the H0 tables
# =============================================================================
paths = [os.path.join(dest_dir, "data_L30.csv"), os.path.join(dest_dir, "data_L60.csv")]
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
