# -*- coding: utf-8 -*-
"""
No Undersampling / 5_Linear_Regression_For_Prediction
Dataset: South German Credit

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

PROTOCOL_BUCKET = "No_Undersampling"
EXPERIMENT = "5_Linear_Regression_For_Prediction"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "South_German_Credit"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
model_results = {}

# =============================================================================
# Get Data - L10
# =============================================================================
src_L10 = os.path.join(src_dir, "data_L10.csv")
dest_L10 = os.path.join(dest_dir, "data_L10.csv")
os.makedirs(dest_dir, exist_ok=True)
data_L10 = pd.read_csv(src_L10)
h0_columns_L10 = [c for c in data_L10.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L10 = data_L10[h0_columns_L10]
data_L10.to_csv(dest_L10, index=False)

features_L10 = data_L10.drop(columns=["label"])
label_L10 = data_L10["label"]

# =============================================================================
# Train / test split on barcode rows - L10
# =============================================================================
X_train_L10, X_test_L10, y_train_L10, y_test_L10 = train_test_split(
    features_L10, label_L10, test_size=0.2, random_state=42, stratify=label_L10
)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L10
# =============================================================================
model_L10 = LinearRegression()
model_L10.fit(X_train_L10, y_train_L10)
scores_L10 = model_L10.predict(X_test_L10)
y_pred_L10 = (scores_L10 >= 0.5).astype(int)
y_true_L10 = y_test_L10.astype(int)
model_results[f"data_L10"] = {
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
print("Linear regression data_L10")

# =============================================================================
# Get Data - L20
# =============================================================================
src_L20 = os.path.join(src_dir, "data_L20.csv")
dest_L20 = os.path.join(dest_dir, "data_L20.csv")
os.makedirs(dest_dir, exist_ok=True)
data_L20 = pd.read_csv(src_L20)
h0_columns_L20 = [c for c in data_L20.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L20 = data_L20[h0_columns_L20]
data_L20.to_csv(dest_L20, index=False)

features_L20 = data_L20.drop(columns=["label"])
label_L20 = data_L20["label"]

# =============================================================================
# Train / test split on barcode rows - L20
# =============================================================================
X_train_L20, X_test_L20, y_train_L20, y_test_L20 = train_test_split(
    features_L20, label_L20, test_size=0.2, random_state=42, stratify=label_L20
)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L20
# =============================================================================
model_L20 = LinearRegression()
model_L20.fit(X_train_L20, y_train_L20)
scores_L20 = model_L20.predict(X_test_L20)
y_pred_L20 = (scores_L20 >= 0.5).astype(int)
y_true_L20 = y_test_L20.astype(int)
model_results[f"data_L20"] = {
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
print("Linear regression data_L20")

print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
