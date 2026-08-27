# -*- coding: utf-8 -*-
"""
Late split, no undersample, using both H0 and H1 / 2_PH_Tuned_Parameters
Dataset: Default of Credit Card Client

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
    train_models_on_multiple_datasets,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

PROTOCOL_BUCKET = "Late_Split_No_Undersample_H0_And_H1"
EXPERIMENT = "2_PH_Tuned_Parameters"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Default_Of_Credit_Card_Client_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)

# =============================================================================
# Load Experiment 1 barcode tables
# =============================================================================
print("Table L5:", os.path.join(src_dir, "data_L5.csv"))
print("Table L15:", os.path.join(src_dir, "data_L15.csv"))

# =============================================================================
# Train models (GridSearchCV, scoring = F1)
# =============================================================================
paths = [
    os.path.join(src_dir, "data_L5.csv"),
    os.path.join(src_dir, "data_L15.csv"),
]
model_results = train_models_on_multiple_datasets(
    data_paths=paths,
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
