# -*- coding: utf-8 -*-
"""
Early Split TDA / 5_Linear_Regression_For_Prediction
Dataset: PKDD'99 Czech Financial

This experiment does not run Ripser. It keeps H0 columns and fits linear regression as a classifier (threshold 0.5).
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# This file lives four folders below the repository root (where utils.py is).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import (
    store_results,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

PROTOCOL_BUCKET = "Early_Split_TDA"
EXPERIMENT = "5_Linear_Regression_For_Prediction"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "PKDD_Czech_Financial"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
model_results = {}

# =============================================================================
# Get Data - L10 (already split by customer in Experiment 1)
# =============================================================================
src_train_L10 = os.path.join(src_dir, "train", "data_L10.csv")
src_test_L10 = os.path.join(src_dir, "test", "data_L10.csv")
dest_train_L10 = os.path.join(dest_dir, "train", "data_L10.csv")
dest_test_L10 = os.path.join(dest_dir, "test", "data_L10.csv")
os.makedirs(os.path.dirname(dest_train_L10), exist_ok=True)
os.makedirs(os.path.dirname(dest_test_L10), exist_ok=True)

train_L10 = pd.read_csv(src_train_L10)
test_L10 = pd.read_csv(src_test_L10)
h0_L10 = [c for c in train_L10.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
train_L10 = train_L10[h0_L10]
test_L10 = test_L10[h0_L10]
train_L10.to_csv(dest_train_L10, index=False)
test_L10.to_csv(dest_test_L10, index=False)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L10
# =============================================================================
feature_cols_L10 = [c for c in train_L10.columns if c != "label"]
model_L10 = LinearRegression()
model_L10.fit(train_L10[feature_cols_L10], train_L10["label"])
scores_L10 = model_L10.predict(test_L10[feature_cols_L10])
y_pred_L10 = (scores_L10 >= 0.5).astype(int)
y_true_L10 = test_L10["label"].astype(int)
model_results[f"data_L10.csv"] = {
    "linear_regression": {
        "model": model_L10,
        "accuracy": accuracy_score(y_true_L10, y_pred_L10),
        "precision": precision_score(y_true_L10, y_pred_L10, zero_division=0),
        "recall": recall_score(y_true_L10, y_pred_L10, zero_division=0),
        "f1_score": f1_score(y_true_L10, y_pred_L10, zero_division=0),
        "classification_report": classification_report(y_true_L10, y_pred_L10, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_L10, y_pred_L10),
    }
}
print("Linear regression (presplit) data_L10")

# =============================================================================
# Get Data - L20 (already split by customer in Experiment 1)
# =============================================================================
src_train_L20 = os.path.join(src_dir, "train", "data_L20.csv")
src_test_L20 = os.path.join(src_dir, "test", "data_L20.csv")
dest_train_L20 = os.path.join(dest_dir, "train", "data_L20.csv")
dest_test_L20 = os.path.join(dest_dir, "test", "data_L20.csv")
os.makedirs(os.path.dirname(dest_train_L20), exist_ok=True)
os.makedirs(os.path.dirname(dest_test_L20), exist_ok=True)

train_L20 = pd.read_csv(src_train_L20)
test_L20 = pd.read_csv(src_test_L20)
h0_L20 = [c for c in train_L20.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
train_L20 = train_L20[h0_L20]
test_L20 = test_L20[h0_L20]
train_L20.to_csv(dest_train_L20, index=False)
test_L20.to_csv(dest_test_L20, index=False)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L20
# =============================================================================
feature_cols_L20 = [c for c in train_L20.columns if c != "label"]
model_L20 = LinearRegression()
model_L20.fit(train_L20[feature_cols_L20], train_L20["label"])
scores_L20 = model_L20.predict(test_L20[feature_cols_L20])
y_pred_L20 = (scores_L20 >= 0.5).astype(int)
y_true_L20 = test_L20["label"].astype(int)
model_results[f"data_L20.csv"] = {
    "linear_regression": {
        "model": model_L20,
        "accuracy": accuracy_score(y_true_L20, y_pred_L20),
        "precision": precision_score(y_true_L20, y_pred_L20, zero_division=0),
        "recall": recall_score(y_true_L20, y_pred_L20, zero_division=0),
        "f1_score": f1_score(y_true_L20, y_pred_L20, zero_division=0),
        "classification_report": classification_report(y_true_L20, y_pred_L20, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_L20, y_pred_L20),
    }
}
print("Linear regression (presplit) data_L20")

print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
