# -*- coding: utf-8 -*-
"""
Early Split TDA And No Undersampling / 5_Linear_Regression_For_Prediction
Dataset: Statlog German Credit

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

PROTOCOL_BUCKET = "Early_Split_TDA_And_No_Undersampling"
EXPERIMENT = "5_Linear_Regression_For_Prediction"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Statlog_German_Credit_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
model_results = {}

# =============================================================================
# Get Data - L30 (already split by customer in Experiment 1)
# =============================================================================
src_train_L30 = os.path.join(src_dir, "train", "data_L30.csv")
src_test_L30 = os.path.join(src_dir, "test", "data_L30.csv")
dest_train_L30 = os.path.join(dest_dir, "train", "data_L30.csv")
dest_test_L30 = os.path.join(dest_dir, "test", "data_L30.csv")
os.makedirs(os.path.dirname(dest_train_L30), exist_ok=True)
os.makedirs(os.path.dirname(dest_test_L30), exist_ok=True)

train_L30 = pd.read_csv(src_train_L30)
test_L30 = pd.read_csv(src_test_L30)
h0_L30 = [c for c in train_L30.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
train_L30 = train_L30[h0_L30]
test_L30 = test_L30[h0_L30]
train_L30.to_csv(dest_train_L30, index=False)
test_L30.to_csv(dest_test_L30, index=False)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L30
# =============================================================================
feature_cols_L30 = [c for c in train_L30.columns if c != "label"]
model_L30 = LinearRegression()
model_L30.fit(train_L30[feature_cols_L30], train_L30["label"])
scores_L30 = model_L30.predict(test_L30[feature_cols_L30])
y_pred_L30 = (scores_L30 >= 0.5).astype(int)
y_true_L30 = test_L30["label"].astype(int)
model_results[f"data_L30"] = {
    "linear_regression": {
        "model": model_L30,
        "accuracy": accuracy_score(y_true_L30, y_pred_L30),
        "precision": precision_score(y_true_L30, y_pred_L30, zero_division=0),
        "recall": recall_score(y_true_L30, y_pred_L30, zero_division=0),
        "f1_score": f1_score(y_true_L30, y_pred_L30, zero_division=0),
        "classification_report": classification_report(y_true_L30, y_pred_L30, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_L30, y_pred_L30),
    }
}
print("Linear regression (presplit) data_L30")

# =============================================================================
# Get Data - L60 (already split by customer in Experiment 1)
# =============================================================================
src_train_L60 = os.path.join(src_dir, "train", "data_L60.csv")
src_test_L60 = os.path.join(src_dir, "test", "data_L60.csv")
dest_train_L60 = os.path.join(dest_dir, "train", "data_L60.csv")
dest_test_L60 = os.path.join(dest_dir, "test", "data_L60.csv")
os.makedirs(os.path.dirname(dest_train_L60), exist_ok=True)
os.makedirs(os.path.dirname(dest_test_L60), exist_ok=True)

train_L60 = pd.read_csv(src_train_L60)
test_L60 = pd.read_csv(src_test_L60)
h0_L60 = [c for c in train_L60.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
train_L60 = train_L60[h0_L60]
test_L60 = test_L60[h0_L60]
train_L60.to_csv(dest_train_L60, index=False)
test_L60.to_csv(dest_test_L60, index=False)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L60
# =============================================================================
feature_cols_L60 = [c for c in train_L60.columns if c != "label"]
model_L60 = LinearRegression()
model_L60.fit(train_L60[feature_cols_L60], train_L60["label"])
scores_L60 = model_L60.predict(test_L60[feature_cols_L60])
y_pred_L60 = (scores_L60 >= 0.5).astype(int)
y_true_L60 = test_L60["label"].astype(int)
model_results[f"data_L60"] = {
    "linear_regression": {
        "model": model_L60,
        "accuracy": accuracy_score(y_true_L60, y_pred_L60),
        "precision": precision_score(y_true_L60, y_pred_L60, zero_division=0),
        "recall": recall_score(y_true_L60, y_pred_L60, zero_division=0),
        "f1_score": f1_score(y_true_L60, y_pred_L60, zero_division=0),
        "classification_report": classification_report(y_true_L60, y_pred_L60, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_L60, y_pred_L60),
    }
}
print("Linear regression (presplit) data_L60")

print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
