# -*- coding: utf-8 -*-
"""
Experiment 3 — Persistent homology with default classifiers
Dataset: PKDD'99 Czech Financial

Stages
------
1. Load processed loans; fill holes; dummy-encode categories.
2. MinMax-scale and PCA(10) on the full table (historical protocol).
3. Undersample non-defaults to the default count.
4. Draw 500 landmark files per class at 10% and 20%.
5. Ripser → barcode statistics → data_L10.csv / data_L20.csv.
6. Train five default-parameter classifiers on those matrices.

Downstream experiments (7, 8, 10, 11, 15, 17, 19, 21, 25, 27) read the
CSV artefacts this script writes.  For a leak-free split see Experiment 23.

Results: 6_Results/3_PH_Default_Parameters/PKDD_Czech_Financial/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from utils import (
    data_preprocessing_pipeline,
    generate_landmark_sets,
    compute_barcodes_from_multiple_landmarks,
    build_final_barcode_statistics_data,
    store_results,
    train_multiple_dataset_tda,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_csv(
    os.path.abspath("../../../1_Data/Processed_Datasets/PKDD_Czech_Financial/processed_data.csv")
)

# =============================================================================
# Preprocessing data
# =============================================================================
# Missing values (numeric → median; categorical → explicit missing token)
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().any():
        data[col] = data[col].fillna(data[col].median())
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].fillna("missing").astype(str)

dummy_col = [
    "frequency",
    "type",
    "sex",
    "A2",
    "A3",
    "A12",
    "A15",
    "preloan_card_type",
]
data = data_preprocessing_pipeline(
    data,
    log_col=["amount", "payments", "tx_amount_sum", "tx_amount_mean"],
    dummy_col=dummy_col,
)

# =============================================================================
# Select dependent and independent variables
# =============================================================================
X = data.drop(columns=["target"])
y = data["target"]

# =============================================================================
# Normalize the features
# =============================================================================
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# =============================================================================
# APPLYING PCA
# =============================================================================
# Target is ~90% variance (DatasetConfig.pca_variance). DCCCD used 7 comps
# (~94%) and Statlog used 15 (~89%) as paper choices. The four new tables
# share 10 components so Ripser spaces stay comparable; 10 is the rank that
# put Taiwan nearest 90%. On this dummy-expanded PKDD table 10 PCs keep far
# less — that miss is documented, not silently re-ranked (re-ranking would
# invalidate downstream data_L*.csv). Experiment 13 matches variance instead
# of component count. Full rationale: docs/Design_Decisions.md
pca = PCA(n_components=10)
X_reduced = pd.DataFrame(
    pca.fit_transform(X_normalized),
    columns=[f"PCA_{num}" for num in range(1, 11)],
)
variance_ratio = pca.explained_variance_ratio_.sum()
print(f"Variance retained with PCA components: {variance_ratio:.2%}")

# =============================================================================
# Balance classes (undersample majority to minority count)
# =============================================================================
reduced_data = X_reduced.copy()
reduced_data["target"] = y

default_data = reduced_data[reduced_data["target"] == 1].reset_index(drop=True)
non_default_data = reduced_data[reduced_data["target"] == 0].reset_index(drop=True)

n_samples = len(default_data)
balanced_non_default = non_default_data.sample(n=n_samples, random_state=42)

# =============================================================================
# SET SAMPLING PERCENTAGE
# =============================================================================
# L10/L20 is not a copy of DCCCD L5/L15 or Statlog L30/L60.
# DCCCD can use 5% because n1=6630 still gives t=331. Statlog must use 30/60
# because n1=300. Here n1=76: 5% would give t=3 (PH dies) and Statlog's 30%
# would over-reuse a tiny class. L10 is the smallest shared percent that
# still gives t=7; L20 is the 2x companion (same doubling as Statlog 30->60)
# so PKDD/Polish/Taiwan/South German stay comparable. See docs/Design_Decisions.md
percentages = [10, 20]

# =============================================================================
# LANDMARK SELECTION
# =============================================================================
generate_landmark_sets(
    class_label_and_data={
        "default": default_data.copy().drop("target", axis=1),
        "non-default": balanced_non_default.copy().drop("target", axis=1),
    },
    landmark_percentages=percentages,
    dataset_to_use="pkdd_czech",
    n_files_per_percentage=500,
    experiment_name="3_PH_Default_Parameters",
)

# =============================================================================
# COMPUTE BARCODE STATISTICS
# =============================================================================
compute_barcodes_from_multiple_landmarks(
    landmark_percentages=percentages,
    landmark_dir="../../../1_Data/Landmark_Sets/PKDD_Czech_Financial/3_PH_Default_Parameters",
    barcode_output_dir="../../../1_Data/Barcode_Statistics/PKDD_Czech_Financial/3_PH_Default_Parameters",
    dim=2,
    label={1: "default", 0: "non-default"},
)

# =============================================================================
# MERGE BARCODE STATISTICS - Create TDA Dataset for Model Building
# =============================================================================
build_final_barcode_statistics_data(
    landmark_percentages=percentages,
    barcode_dir="../../../1_Data/Barcode_Statistics/PKDD_Czech_Financial/3_PH_Default_Parameters",
    output_dir="../../../1_Data/TDA_Datasets/PKDD_Czech_Financial/3_PH_Default_Parameters",
    label={1: "default", 0: "non-default"},
)

# =============================================================================
# TRAIN MACHINE LEARNING MODEL
# =============================================================================
paths = [
    "../../../1_Data/TDA_Datasets/PKDD_Czech_Financial/3_PH_Default_Parameters/data_L10.csv",
    "../../../1_Data/TDA_Datasets/PKDD_Czech_Financial/3_PH_Default_Parameters/data_L20.csv",
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
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../6_Results/3_PH_Default_Parameters/PKDD_Czech_Financial"
store_results(path=save_path, save_name="model_results", result_object=model_results)
