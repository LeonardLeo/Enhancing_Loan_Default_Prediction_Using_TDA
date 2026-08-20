# -*- coding: utf-8 -*-
"""
Early_Split_TDA / 5_Linear_Regression_For_Prediction
Dataset: Taiwanese Bankruptcy Prediction

This file is the method document for this dataset. Heavy Ripser / IO helpers
live in utils.py; the pipeline itself is written here in order.

Protocol
--------
- Split timing : early
- Undersample  : True
- PCA rank     : 10  (historical Exp 3 rank for this table)
- Snapshot size percents : [10.0, 20.0]
- Number of snapshots    : 500
This experiment does not run Ripser. It loads Experiment 1 barcode tables,
keeps the H0 slice, and fits linear regression as a classifier (threshold 0.5).
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (
    _percent_token,
    _write_h0_slice,
    store_results,
    tda_artefact_dir,
    tda_results_dir,
    train_multiple_dataset_tda_linear_regression,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'taiwan_bankruptcy'
PROTOCOL_BUCKET = 'Early_Split_TDA'
EXPERIMENT = "5_Linear_Regression_For_Prediction"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = 'Taiwan_Bankruptcy'

SPLIT_TIMING = 'early'
UNDERSAMPLE = True
LANDMARK_PERCENTAGES = [10.0, 20.0]
RANDOM_STATE = 42
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.5

# =============================================================================
# Load H0 barcode tables and fit linear regression on the customer split
# =============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

model_results = {}
dest_root = tda_artefact_dir("TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
for pct in LANDMARK_PERCENTAGES:
    token = _percent_token(pct)
    train_src = tda_artefact_dir(
        "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER, "train", f"data_L{token}.csv"
    )
    test_src = tda_artefact_dir(
        "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER, "test", f"data_L{token}.csv"
    )
    dest_train = dest_root / "train" / f"data_L{token}.csv"
    dest_test = dest_root / "test" / f"data_L{token}.csv"
    _write_h0_slice(train_src, dest_train)
    _write_h0_slice(test_src, dest_test)
    train_df = pd.read_csv(dest_train)
    test_df = pd.read_csv(dest_test)
    feature_cols = [c for c in train_df.columns if c != "label"]
    model = LinearRegression()
    model.fit(train_df[feature_cols], train_df["label"])
    scores = model.predict(test_df[feature_cols])
    y_pred = (scores >= 0.5).astype(int)
    y_test = test_df["label"].astype(int)
    model_results[f"data_L{token}"] = {
        "linear_regression": {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
    }
    print(f"Linear regression (presplit) data_L{token}")

print(model_results)

# =============================================================================
# Store results
# =============================================================================
save_path = tda_results_dir(PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
store_results(path=str(save_path), save_name="model_results", result_object=model_results)
