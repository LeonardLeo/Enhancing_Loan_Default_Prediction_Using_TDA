# -*- coding: utf-8 -*-
"""
Early Split TDA / 5_Linear_Regression_For_Prediction
Dataset: Default of Credit Card Client

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
FOLDER = "Default_Of_Credit_Card_Client_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
model_results = {}

# =============================================================================
# Get Data - L5 (already split by customer in Experiment 1)
# =============================================================================
src_train_L5 = os.path.join(src_dir, "train", "data_L5.csv")
src_test_L5 = os.path.join(src_dir, "test", "data_L5.csv")
dest_train_L5 = os.path.join(dest_dir, "train", "data_L5.csv")
dest_test_L5 = os.path.join(dest_dir, "test", "data_L5.csv")
os.makedirs(os.path.dirname(dest_train_L5), exist_ok=True)
os.makedirs(os.path.dirname(dest_test_L5), exist_ok=True)

train_L5 = pd.read_csv(src_train_L5)
test_L5 = pd.read_csv(src_test_L5)
h0_L5 = [c for c in train_L5.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
train_L5 = train_L5[h0_L5]
test_L5 = test_L5[h0_L5]
train_L5.to_csv(dest_train_L5, index=False)
test_L5.to_csv(dest_test_L5, index=False)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L5
# =============================================================================
feature_cols_L5 = [c for c in train_L5.columns if c != "label"]
model_L5 = LinearRegression()
model_L5.fit(train_L5[feature_cols_L5], train_L5["label"])
scores_L5 = model_L5.predict(test_L5[feature_cols_L5])
y_pred_L5 = (scores_L5 >= 0.5).astype(int)
y_true_L5 = test_L5["label"].astype(int)
model_results[f"data_L5"] = {
    "linear_regression": {
        "model": model_L5,
        "accuracy": accuracy_score(y_true_L5, y_pred_L5),
        "precision": precision_score(y_true_L5, y_pred_L5, zero_division=0),
        "recall": recall_score(y_true_L5, y_pred_L5, zero_division=0),
        "f1_score": f1_score(y_true_L5, y_pred_L5, zero_division=0),
        "classification_report": classification_report(y_true_L5, y_pred_L5, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_L5, y_pred_L5),
    }
}
print("Linear regression (presplit) data_L5")

# =============================================================================
# Get Data - L15 (already split by customer in Experiment 1)
# =============================================================================
src_train_L15 = os.path.join(src_dir, "train", "data_L15.csv")
src_test_L15 = os.path.join(src_dir, "test", "data_L15.csv")
dest_train_L15 = os.path.join(dest_dir, "train", "data_L15.csv")
dest_test_L15 = os.path.join(dest_dir, "test", "data_L15.csv")
os.makedirs(os.path.dirname(dest_train_L15), exist_ok=True)
os.makedirs(os.path.dirname(dest_test_L15), exist_ok=True)

train_L15 = pd.read_csv(src_train_L15)
test_L15 = pd.read_csv(src_test_L15)
h0_L15 = [c for c in train_L15.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
train_L15 = train_L15[h0_L15]
test_L15 = test_L15[h0_L15]
train_L15.to_csv(dest_train_L15, index=False)
test_L15.to_csv(dest_test_L15, index=False)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L15
# =============================================================================
feature_cols_L15 = [c for c in train_L15.columns if c != "label"]
model_L15 = LinearRegression()
model_L15.fit(train_L15[feature_cols_L15], train_L15["label"])
scores_L15 = model_L15.predict(test_L15[feature_cols_L15])
y_pred_L15 = (scores_L15 >= 0.5).astype(int)
y_true_L15 = test_L15["label"].astype(int)
model_results[f"data_L15"] = {
    "linear_regression": {
        "model": model_L15,
        "accuracy": accuracy_score(y_true_L15, y_pred_L15),
        "precision": precision_score(y_true_L15, y_pred_L15, zero_division=0),
        "recall": recall_score(y_true_L15, y_pred_L15, zero_division=0),
        "f1_score": f1_score(y_true_L15, y_pred_L15, zero_division=0),
        "classification_report": classification_report(y_true_L15, y_pred_L15, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_L15, y_pred_L15),
    }
}
print("Linear regression (presplit) data_L15")

print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
