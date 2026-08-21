# -*- coding: utf-8 -*-
"""
Historical Late Split, Balanced TDA / 5_Linear_Regression_For_Prediction
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
FOLDER = "Default_Of_Credit_Card_Client_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
dest_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
model_results = {}

# =============================================================================
# Get Data - L5
# =============================================================================
src_L5 = os.path.join(src_dir, "data_L5.csv")
dest_L5 = os.path.join(dest_dir, "data_L5.csv")
os.makedirs(dest_dir, exist_ok=True)
data_L5 = pd.read_csv(src_L5)
h0_columns_L5 = [c for c in data_L5.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L5 = data_L5[h0_columns_L5]
data_L5.to_csv(dest_L5, index=False)

features_L5 = data_L5.drop(columns=["label"])
label_L5 = data_L5["label"]

# =============================================================================
# Train / test split on barcode rows - L5
# =============================================================================
X_train_L5, X_test_L5, y_train_L5, y_test_L5 = train_test_split(
    features_L5, label_L5, test_size=0.2, random_state=42, stratify=label_L5
)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L5
# =============================================================================
model_L5 = LinearRegression()
model_L5.fit(X_train_L5, y_train_L5)
scores_L5 = model_L5.predict(X_test_L5)
y_pred_L5 = (scores_L5 >= 0.5).astype(int)
y_true_L5 = y_test_L5.astype(int)
model_results[f"data_L5.csv"] = {
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
print("Linear regression data_L5")

# =============================================================================
# Get Data - L15
# =============================================================================
src_L15 = os.path.join(src_dir, "data_L15.csv")
dest_L15 = os.path.join(dest_dir, "data_L15.csv")
os.makedirs(dest_dir, exist_ok=True)
data_L15 = pd.read_csv(src_L15)
h0_columns_L15 = [c for c in data_L15.columns if c == "label" or str(c).endswith("_0") or "(Dim 0)" in str(c)]
data_L15 = data_L15[h0_columns_L15]
data_L15.to_csv(dest_L15, index=False)

features_L15 = data_L15.drop(columns=["label"])
label_L15 = data_L15["label"]

# =============================================================================
# Train / test split on barcode rows - L15
# =============================================================================
X_train_L15, X_test_L15, y_train_L15, y_test_L15 = train_test_split(
    features_L15, label_L15, test_size=0.2, random_state=42, stratify=label_L15
)

# =============================================================================
# Linear Regression as a classifier (threshold 0.5) - L15
# =============================================================================
model_L15 = LinearRegression()
model_L15.fit(X_train_L15, y_train_L15)
scores_L15 = model_L15.predict(X_test_L15)
y_pred_L15 = (scores_L15 >= 0.5).astype(int)
y_true_L15 = y_test_L15.astype(int)
model_results[f"data_L15.csv"] = {
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
print("Linear regression data_L15")

print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
