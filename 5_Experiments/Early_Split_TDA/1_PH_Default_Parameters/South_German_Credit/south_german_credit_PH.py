# -*- coding: utf-8 -*-
"""
Early Split TDA / 1_PH_Default_Parameters
Dataset: South German Credit

Early split: customers are split before scaling and PCA. Train and test snapshots never share people.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# This file lives four folders below the repository root (where utils.py is).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import (
    data_preprocessing_pipeline,
    generate_landmark_sets,
    compute_barcodes_from_multiple_landmarks,
    build_final_barcode_statistics_data,
    train_multiple_dataset_tda_presplit,
    store_results,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Settings for this dataset
# =============================================================================
DATASET_KEY = "south_german_credit"
PROTOCOL_BUCKET = "Early_Split_TDA"
EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "South_German_Credit"
LANDMARK_PERCENTAGES = [10.0, 20.0]
N_FILES = 500
HOMOLOGY_DIM = 2
SKIP_EXISTING = True

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_csv(os.path.join(REPO_ROOT, "1_Data", "Processed_Datasets", "South_German_Credit", "processed_data.csv"))
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
data = data_preprocessing_pipeline(data, log_col=["hoehe", "laufzeit"])
X = data.drop(columns=["target"])
y = data["target"].astype(int)
X = X.select_dtypes(include=[np.number]).copy()

# =============================================================================
# Customer split FIRST (80% train / 20% test, stratified on the label)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# Scale + PCA fitted on TRAIN only, then applied to test
# =============================================================================
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

pca = PCA(n_components=10, random_state=42)
X_train_pca = pd.DataFrame(
    pca.fit_transform(X_train_scaled),
    columns=[f"PCA_{num}" for num in range(1, 10 + 1)],
    index=X_train.index,
)
X_test_pca = pd.DataFrame(
    pca.transform(X_test_scaled),
    columns=[f"PCA_{num}" for num in range(1, 10 + 1)],
    index=X_test.index,
)
print(f"Variance retained with train-fit PCA: {pca.explained_variance_ratio_.sum():.2%}")

# =============================================================================
# Balance classes INSIDE train, then INSIDE test (never mix the two pools)
# =============================================================================
train_frame = X_train_pca.copy()
train_frame["Class"] = y_train.values
default_train = train_frame[train_frame["Class"] == 1].reset_index(drop=True)
non_default_train = train_frame[train_frame["Class"] == 0].reset_index(drop=True)
n_train = min(len(default_train), len(non_default_train))
default_train = default_train.sample(n=n_train, random_state=42).reset_index(drop=True)
non_default_train = non_default_train.sample(n=n_train, random_state=42).reset_index(drop=True)

test_frame = X_test_pca.copy()
test_frame["Class"] = y_test.values
default_test = test_frame[test_frame["Class"] == 1].reset_index(drop=True)
non_default_test = test_frame[test_frame["Class"] == 0].reset_index(drop=True)
n_test = min(len(default_test), len(non_default_test))
default_test = default_test.sample(n=n_test, random_state=42).reset_index(drop=True)
non_default_test = non_default_test.sample(n=n_test, random_state=42).reset_index(drop=True)

print("TRAIN default:", len(default_train), " TRAIN non-default:", len(non_default_train))
print("TEST  default:", len(default_test), " TEST  non-default:", len(non_default_test))
train_pools = {
    "default": default_train.drop(columns=["Class"]),
    "non-default": non_default_train.drop(columns=["Class"]),
}
test_pools = {
    "default": default_test.drop(columns=["Class"]),
    "non-default": non_default_test.drop(columns=["Class"]),
}

# =============================================================================
# Paths for TRAIN landmarks/barcodes and TEST landmarks/barcodes
# =============================================================================
train_landmark_dir = os.path.join(REPO_ROOT, "1_Data", "Landmark_Sets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "train")
train_barcode_dir = os.path.join(REPO_ROOT, "1_Data", "Barcode_Statistics", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "train")
train_tda_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "train")
test_landmark_dir = os.path.join(REPO_ROOT, "1_Data", "Landmark_Sets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "test")
test_barcode_dir = os.path.join(REPO_ROOT, "1_Data", "Barcode_Statistics", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "test")
test_tda_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER, "test")
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)

train_table_L10 = os.path.join(train_tda_dir, "data_L10.csv")
train_table_L20 = os.path.join(train_tda_dir, "data_L20.csv")
test_table_L10 = os.path.join(test_tda_dir, "data_L10.csv")
test_table_L20 = os.path.join(test_tda_dir, "data_L20.csv")

already_built = (
    SKIP_EXISTING
    and os.path.exists(train_table_L10)
    and os.path.exists(train_table_L20)
    and os.path.exists(test_table_L10)
    and os.path.exists(test_table_L20)
)

# =============================================================================
# Landmark selection + barcodes for TRAIN customers
# =============================================================================
if not already_built:
    print("----- TRAIN snapshots -----")
    generate_landmark_sets(
        class_label_and_data=train_pools,
        landmark_percentages=LANDMARK_PERCENTAGES,
        dataset_to_use=DATASET_KEY,
        experiment_name=EXPERIMENT,
        add_optional_path="train",
        n_files_per_percentage=N_FILES,
        protocol_bucket=PROTOCOL_BUCKET,
    )
    compute_barcodes_from_multiple_landmarks(
        landmark_percentages=LANDMARK_PERCENTAGES,
        landmark_dir=train_landmark_dir,
        barcode_output_dir=train_barcode_dir,
        dim=HOMOLOGY_DIM,
        label={1: "default", 0: "non-default"},
    )
    build_final_barcode_statistics_data(
        landmark_percentages=LANDMARK_PERCENTAGES,
        barcode_dir=train_barcode_dir,
        output_dir=train_tda_dir,
        label={1: "default", 0: "non-default"},
    )

    # =============================================================================
    # Landmark selection + barcodes for TEST customers
    # =============================================================================
    print("----- TEST snapshots -----")
    generate_landmark_sets(
        class_label_and_data=test_pools,
        landmark_percentages=LANDMARK_PERCENTAGES,
        dataset_to_use=DATASET_KEY,
        experiment_name=EXPERIMENT,
        add_optional_path="test",
        n_files_per_percentage=N_FILES,
        protocol_bucket=PROTOCOL_BUCKET,
    )
    compute_barcodes_from_multiple_landmarks(
        landmark_percentages=LANDMARK_PERCENTAGES,
        landmark_dir=test_landmark_dir,
        barcode_output_dir=test_barcode_dir,
        dim=HOMOLOGY_DIM,
        label={1: "default", 0: "non-default"},
    )
    build_final_barcode_statistics_data(
        landmark_percentages=LANDMARK_PERCENTAGES,
        barcode_dir=test_barcode_dir,
        output_dir=test_tda_dir,
        label={1: "default", 0: "non-default"},
    )
else:
    print("Barcode tables already exist; skipping Ripser.")

# =============================================================================
# Train on train barcodes; score on test barcodes (no extra 80/20 mix)
# =============================================================================
pairs = {
    "data_L10": {"train": train_table_L10, "test": test_table_L10},
    "data_L20": {"train": train_table_L20, "test": test_table_L20},
}

model_results = train_multiple_dataset_tda_presplit(
    train_test_pairs=pairs,
    y_col_name="label",
    random_state=42,
    xgb={"eval_metric": "logloss"},
)
print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
