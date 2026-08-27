# -*- coding: utf-8 -*-
"""
Late split, no undersample, using both H0 and H1 / 1_PH_Default_Parameters
Dataset: Taiwanese Bankruptcy Prediction

Step by step: load table, scale, PCA, split classes, make snapshots, Ripser, train, store.
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
from sklearn.preprocessing import MinMaxScaler

# This file lives four folders below the repository root (where utils.py is).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import (
    data_preprocessing_pipeline,
    generate_landmark_sets,
    compute_barcodes_from_multiple_landmarks,
    build_final_barcode_statistics_data,
    train_multiple_dataset_tda,
    store_results,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Settings for this dataset
# =============================================================================
DATASET_KEY = "taiwan_bankruptcy"
PROTOCOL_BUCKET = "Late_Split_No_Undersample_H0_And_H1"
EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Taiwan_Bankruptcy"
LANDMARK_PERCENTAGES = [10.0, 20.0]
N_FILES = 500
HOMOLOGY_DIM = 2
SKIP_EXISTING = True

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_csv(os.path.join(REPO_ROOT, "1_Data", "Processed_Datasets", "Taiwan_Bankruptcy", "processed_data.csv"))
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
data = data_preprocessing_pipeline(data)
X = data.drop(columns=["target"])
y = data["target"].astype(int)
X = X.select_dtypes(include=[np.number]).copy()

# =============================================================================
# Scale features (MinMax so every column sits in [0, 1])
# =============================================================================
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# =============================================================================
# PCA on the full table (late-split geometry)
# =============================================================================
pca = PCA(n_components=10, random_state=42)
X_reduced = pd.DataFrame(
    pca.fit_transform(X_normalized),
    columns=[f"PCA_{num}" for num in range(1, 10 + 1)],
    index=X.index,
)
print(f"Variance retained with PCA components: {pca.explained_variance_ratio_.sum():.2%}")

# =============================================================================
# Keep both classes at their original sizes (no undersampling)
# =============================================================================
reduced_data = X_reduced.copy()
reduced_data["Class"] = y.values
default_data = reduced_data[reduced_data["Class"] == 1].reset_index(drop=True)
non_default_data = reduced_data[reduced_data["Class"] == 0].reset_index(drop=True)

print("default rows:", len(default_data), " non-default rows:", len(non_default_data))
pools = {
    "default": default_data.drop(columns=["Class"]),
    "non-default": non_default_data.drop(columns=["Class"]),
}

# =============================================================================
# Paths for landmarks, barcodes, and the final TDA tables
# =============================================================================
landmark_dir = os.path.join(REPO_ROOT, "1_Data", "Landmark_Sets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
barcode_dir = os.path.join(REPO_ROOT, "1_Data", "Barcode_Statistics", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
tda_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)

already_built = SKIP_EXISTING and all(
    os.path.exists(os.path.join(tda_dir, f"data_L{int(p) if float(p).is_integer() else p}.csv"))
    for p in LANDMARK_PERCENTAGES
)

# =============================================================================
# Landmark selection
# =============================================================================
# Points per snapshot = floor(class count x snapshot size percent / 100)
if not already_built:
    for name, frame in pools.items():
        for pct in LANDMARK_PERCENTAGES:
            points_per_snapshot = max(2, int(len(frame) * pct / 100.0))
            print(f"  {name}  percent={pct:g}  n={len(frame)}  points_per_snapshot={points_per_snapshot}  n_snapshots={N_FILES}")
    generate_landmark_sets(
        class_label_and_data=pools,
        landmark_percentages=LANDMARK_PERCENTAGES,
        dataset_to_use=DATASET_KEY,
        experiment_name=EXPERIMENT,
        n_files_per_percentage=N_FILES,
        protocol_bucket=PROTOCOL_BUCKET,
    )

    # =============================================================================
    # Compute barcode statistics (Ripser on each snapshot)
    # =============================================================================
    compute_barcodes_from_multiple_landmarks(
        landmark_percentages=LANDMARK_PERCENTAGES,
        landmark_dir=landmark_dir,
        barcode_output_dir=barcode_dir,
        dim=HOMOLOGY_DIM,
        label={1: "default", 0: "non-default"},
    )

    # =============================================================================
    # Merge barcode statistics into one table per snapshot-size percent
    # =============================================================================
    build_final_barcode_statistics_data(
        landmark_percentages=LANDMARK_PERCENTAGES,
        barcode_dir=barcode_dir,
        output_dir=tda_dir,
        label={1: "default", 0: "non-default"},
    )
else:
    print("Barcode tables already exist; skipping Ripser.")

# =============================================================================
# Train machine learning models with default parameters
# =============================================================================
paths = [
    os.path.join(tda_dir, "data_L10.csv"),
    os.path.join(tda_dir, "data_L20.csv"),
]
model_results = train_multiple_dataset_tda(
    path_datasets=paths,
    y_col_name="label",
    test_size=0.2,
    random_state=42,
    xgb={"eval_metric": "logloss"},
)
print(model_results)

# =============================================================================
# Store model results
# =============================================================================
store_results(path=save_path, save_name="model_results", result_object=model_results)
