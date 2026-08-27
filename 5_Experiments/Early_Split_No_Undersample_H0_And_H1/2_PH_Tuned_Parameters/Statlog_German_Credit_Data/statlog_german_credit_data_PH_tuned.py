# -*- coding: utf-8 -*-
"""
Early split, no undersample, using both H0 and H1 / 2_PH_Tuned_Parameters
Dataset: Statlog German Credit

This experiment does not run Ripser. It reloads Experiment 1 barcode tables and retunes the five classifiers.
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
    DEFAULT_TDA_TUNED_MODEL_CONFIGS,
    store_results,
    train_models_on_multiple_presplit_datasets,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

PROTOCOL_BUCKET = "Early_Split_No_Undersample_H0_And_H1"
EXPERIMENT = "2_PH_Tuned_Parameters"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Statlog_German_Credit_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)

# =============================================================================
# Train models (GridSearchCV, scoring = F1)
# =============================================================================
pairs = {
    "data_L30": {
        "train": os.path.join(src_dir, "train", "data_L30.csv"),
        "test": os.path.join(src_dir, "test", "data_L30.csv"),
    },
    "data_L60": {
        "train": os.path.join(src_dir, "train", "data_L60.csv"),
        "test": os.path.join(src_dir, "test", "data_L60.csv"),
    },
}
print("Table L30 train:", pairs["data_L30"]["train"])
print("Table L30 test :", pairs["data_L30"]["test"])
print("Table L60 train:", pairs["data_L60"]["train"])
print("Table L60 test :", pairs["data_L60"]["test"])
model_results = train_models_on_multiple_presplit_datasets(
    train_test_pairs=pairs,
    model_configs=DEFAULT_TDA_TUNED_MODEL_CONFIGS,
    target_column="label",
    scoring_metric="f1",
    random_state=42,
)

print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
