# -*- coding: utf-8 -*-
"""
Historical Late Split, Balanced TDA / 5_Linear_Regression_For_Prediction
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
from sklearn.model_selection import train_test_split

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

PROTOCOL_BUCKET = "Historical_Late_Split_Balanced_TDA"
EXPERIMENT = "5_Linear_Regression_For_Prediction"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Statlog_German_Credit_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
model_results = {}

# =============================================================================
# Get Data - L30
# =============================================================================
src_L30 = os.path.join(src_dir, "data_L30.csv")
dest_L30 = os.path.join(dest_dir, "data_L30.csv")
os.makedirs(dest_dir, exist_ok=True)
data_L30 = pd.read_csv(src_L30)
h0_columns_L30 = [c for c in data_L30.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L30 = data_L30[h0_columns_L30]
data_L30.to_csv(dest_L30, index=False)

features_L30 = data_L30.drop(columns=["label"])
label_L30 = data_L30["label"]

# =============================================================================
# Train / test split on barcode rows - L30
# =============================================================================
X_train_L30, X_test_L30, y_train_L30, y_test_L30 = train_test_split(
    features_L30, label_L30, test_size=0.2, random_state=42, stratify=label_L30
)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L30
# =============================================================================
model_L30 = LinearRegression()
model_L30.fit(X_train_L30, y_train_L30)
scores_L30 = model_L30.predict(X_test_L30)
y_pred_L30 = (scores_L30 >= 0.5).astype(int)
y_true_L30 = y_test_L30.astype(int)
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
print("Linear regression data_L30")

# =============================================================================
# Get Data - L60
# =============================================================================
src_L60 = os.path.join(src_dir, "data_L60.csv")
dest_L60 = os.path.join(dest_dir, "data_L60.csv")
os.makedirs(dest_dir, exist_ok=True)
data_L60 = pd.read_csv(src_L60)
h0_columns_L60 = [c for c in data_L60.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L60 = data_L60[h0_columns_L60]
data_L60.to_csv(dest_L60, index=False)

features_L60 = data_L60.drop(columns=["label"])
label_L60 = data_L60["label"]

# =============================================================================
# Train / test split on barcode rows - L60
# =============================================================================
X_train_L60, X_test_L60, y_train_L60, y_test_L60 = train_test_split(
    features_L60, label_L60, test_size=0.2, random_state=42, stratify=label_L60
)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L60
# =============================================================================
model_L60 = LinearRegression()
model_L60.fit(X_train_L60, y_train_L60)
scores_L60 = model_L60.predict(X_test_L60)
y_pred_L60 = (scores_L60 >= 0.5).astype(int)
y_true_L60 = y_test_L60.astype(int)
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
print("Linear regression data_L60")

print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
