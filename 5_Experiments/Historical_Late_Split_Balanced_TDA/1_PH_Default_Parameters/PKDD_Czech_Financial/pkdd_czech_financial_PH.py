# -*- coding: utf-8 -*-
"""
Historical_Late_Split_Balanced_TDA / 1_PH_Default_Parameters
Dataset: PKDD'99 Czech Financial

This file is the method document for this dataset. Heavy Ripser / IO helpers
live in utils.py; the pipeline itself is written here in order.

Protocol
--------
- Split timing : late
- Undersample  : True
- PCA rank     : 10  (historical Exp 3 rank for this table)
- Snapshot size percents : [10.0, 20.0]
- Number of snapshots    : 500
This experiment BUILDS landmarks and Ripser barcodes, then trains five
classifiers with default hyperparameters. Downstream experiments 2–5 and
7–8 reuse the barcode tables written here.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (
    load_processed_features,
    class_pools_from_features,
    generate_landmark_sets,
    compute_barcodes_from_multiple_landmarks,
    build_final_barcode_statistics_data,
    store_results,
    store_data_as_csv_or_json,
    tda_artefact_dir,
    tda_results_dir,
    protocol_tda_matrices_exist,
    train_multiple_dataset_tda,
    late_split_barcode_paths,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'pkdd_czech'
PROTOCOL_BUCKET = 'Historical_Late_Split_Balanced_TDA'
EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = 'PKDD_Czech_Financial'

SPLIT_TIMING = 'late'
UNDERSAMPLE = True          # yes
PCA_N_COMPONENTS = 10
LANDMARK_PERCENTAGES = [10.0, 20.0]
N_FILES = 500                    # snapshots per class per percent
HOMOLOGY_DIM = 2
RANDOM_STATE = 42
TEST_SIZE = 0.2
SKIP_EXISTING = True              # do not rebuild barcodes that already exist
LABEL_MAP = {1: "default", 0: "non-default"}

# Processed table (shared across arms):
# ../../../../1_Data/Processed_Datasets/PKDD_Czech_Financial/processed_data.csv

# =============================================================================
# Load data
# =============================================================================
X, y, cfg = load_processed_features(DATASET_KEY)

# =============================================================================
# Scale and PCA (full table)
# =============================================================================
# MinMaxScaler + PCA fitted on the full table (historical geometry).
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
pca_cols = [f"PCA_{i}" for i in range(1, PCA_N_COMPONENTS + 1)]
X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=pca_cols, index=X.index)
variance_retained = float(pca.explained_variance_ratio_.sum())
print(f"Variance retained (full-table PCA): {variance_retained:.2%}")

# =============================================================================
# Balance classes
# =============================================================================
# Undersample the majority class to the minority class count before snapshots.
pools = class_pools_from_features(
    X_pca, y, undersample=UNDERSAMPLE, random_state=RANDOM_STATE
)
print("Class pools:", {k: len(v) for k, v in pools.items()})

already_built = SKIP_EXISTING and protocol_tda_matrices_exist(
    PROTOCOL_BUCKET, FOLDER, LANDMARK_PERCENTAGES, SPLIT_TIMING, EXPERIMENT
)

# =============================================================================
# Generate landmark sets, compute barcodes, merge barcode statistics
# =============================================================================
# Points per snapshot = floor(class count x snapshot size percent / 100)
# on the pool after optional undersample.
if not already_built:
    for name, frame in pools.items():
        for pct in LANDMARK_PERCENTAGES:
            points_per_snapshot = max(2, int(len(frame) * pct / 100.0))
            print(
                f"  {name}  percent={pct:g}  n={len(frame)}  "
                f"points_per_snapshot={points_per_snapshot}  n_snapshots={N_FILES}"
            )
    generate_landmark_sets(
        class_label_and_data=pools,
        landmark_percentages=LANDMARK_PERCENTAGES,
        dataset_to_use=DATASET_KEY,
        experiment_name=EXPERIMENT,
        n_files_per_percentage=N_FILES,
        protocol_bucket=PROTOCOL_BUCKET,
    )
    landmark_dir = tda_artefact_dir("Landmark_Sets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
    barcode_dir = tda_artefact_dir("Barcode_Statistics", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
    tda_dir = tda_artefact_dir("TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
    compute_barcodes_from_multiple_landmarks(
        landmark_percentages=LANDMARK_PERCENTAGES,
        landmark_dir=str(landmark_dir),
        barcode_output_dir=str(barcode_dir),
        dim=HOMOLOGY_DIM,
        label=LABEL_MAP,
    )
    build_final_barcode_statistics_data(
        landmark_percentages=LANDMARK_PERCENTAGES,
        barcode_dir=str(barcode_dir),
        output_dir=str(tda_dir),
        label=LABEL_MAP,
    )
else:
    print("Barcode tables already exist; skipping Ripser.")

# =============================================================================
# Store protocol metadata
# =============================================================================
save_path = tda_results_dir(PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
store_data_as_csv_or_json(
    path=str(save_path),
    csv=False,
    save_as=["protocol_metadata"],
    data_object=[{
        "dataset": FOLDER,
        "protocol_bucket": PROTOCOL_BUCKET,
        "split_timing": SPLIT_TIMING,
        "undersample": UNDERSAMPLE,
        "pca_n_components": PCA_N_COMPONENTS,
        "landmark_percentages": LANDMARK_PERCENTAGES,
        "n_files_per_percentage": N_FILES,
        "homology_dim": HOMOLOGY_DIM,
        "skipped_existing": already_built,
    }],
)

# =============================================================================
# Train models (default hyperparameters)
# =============================================================================
# 80/20 on barcode rows (late / snapshot-level hold-out).
paths = late_split_barcode_paths(PROTOCOL_BUCKET, FOLDER, LANDMARK_PERCENTAGES, EXPERIMENT)
model_results = train_multiple_dataset_tda(
    path_datasets=paths,
    y_col_name="label",
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    xgb={"eval_metric": "logloss"},
)

print(model_results)

# =============================================================================
# Store results
# =============================================================================
store_results(path=str(save_path), save_name="model_results", result_object=model_results)
