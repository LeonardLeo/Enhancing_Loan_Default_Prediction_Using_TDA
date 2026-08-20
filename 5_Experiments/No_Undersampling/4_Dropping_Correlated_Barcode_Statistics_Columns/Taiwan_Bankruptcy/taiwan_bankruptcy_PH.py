# -*- coding: utf-8 -*-
"""
No_Undersampling / 4_Dropping_Correlated_Barcode_Statistics_Columns
Dataset: Taiwanese Bankruptcy Prediction

This file is the method document for this dataset. Heavy Ripser / IO helpers
live in utils.py; the pipeline itself is written here in order.

Protocol
--------
- Split timing : late
- Undersample  : False
- PCA rank     : 10  (historical Exp 3 rank for this table)
- Snapshot size percents : [10.0, 20.0]
- Number of snapshots    : 500
This experiment does not run Ripser. It loads Experiment 1 barcode tables,
drops correlated barcode-statistic columns (threshold 0.80), then trains.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (
    drop_correlated_features,
    rename_barcode_statistics_columns,
    store_data_as_csv_or_json,
    store_results,
    tda_artefact_dir,
    tda_results_dir,
    train_multiple_dataset_tda_drop_correlated,
    _percent_token,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'taiwan_bankruptcy'
PROTOCOL_BUCKET = 'No_Undersampling'
EXPERIMENT = "4_Dropping_Correlated_Barcode_Statistics_Columns"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = 'Taiwan_Bankruptcy'

SPLIT_TIMING = 'late'
UNDERSAMPLE = False
LANDMARK_PERCENTAGES = [10.0, 20.0]
CORR_THRESHOLD = 0.80
RANDOM_STATE = 42
TEST_SIZE = 0.2

save_path = str(tda_results_dir(PROTOCOL_BUCKET, EXPERIMENT, FOLDER))
var_dir = tda_artefact_dir(
    "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "Using_High_Variance_For_Correlation"
)
target_dir = tda_artefact_dir(
    "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "Using_Target_Variable_For_Correlation"
)
var_dir.mkdir(parents=True, exist_ok=True)
target_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load barcode tables, 80/20 split, drop correlated columns on train
# =============================================================================
data_objects = {}
dropped_payload = []
dropped_names = []
for pct in LANDMARK_PERCENTAGES:
    token = _percent_token(pct)
    src = tda_artefact_dir(
        "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER, f"data_L{token}.csv"
    )
    df = rename_barcode_statistics_columns(pd.read_csv(src)).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    kept_var, dropped_var = drop_correlated_features(
        X_train,
        threshold=CORR_THRESHOLD,
        feature_label=True,
        strategy="high_variance",
        target=y_train,
    )
    kept_target, dropped_target = drop_correlated_features(
        X_train,
        threshold=CORR_THRESHOLD,
        feature_label=True,
        strategy="target_corr",
        target=y_train,
    )
    keep_cols = [c for c in kept_target.columns if c != "label"]
    name = f"data_L{token}"
    data_objects[name] = {
        "data": kept_target,
        "X_test": X_test[keep_cols],
        "y_test": y_test,
    }
    kept_var.to_csv(var_dir / f"{name}_var.csv", index=False)
    kept_target.to_csv(target_dir / f"{name}_target.csv", index=False)
    dropped_payload.extend([dropped_var, dropped_target])
    dropped_names.extend([f"{name}_var_drop", f"{name}_target_drop"])
    print(f"Kept {len(keep_cols)} columns from {name}")

store_data_as_csv_or_json(
    path=save_path, csv=False, save_as=dropped_names, data_object=dropped_payload
)

# =============================================================================
# Train models
# =============================================================================
model_results = train_multiple_dataset_tda_drop_correlated(
    data_objects=data_objects,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    xgb={"eval_metric": "logloss"},
)
print(model_results)

# =============================================================================
# Store results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
