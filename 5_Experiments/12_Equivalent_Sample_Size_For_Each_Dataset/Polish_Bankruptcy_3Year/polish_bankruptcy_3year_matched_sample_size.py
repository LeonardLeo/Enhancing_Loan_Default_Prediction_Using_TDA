# -*- coding: utf-8 -*-
"""
Experiment 12 — Equivalent Sample Size For Each Dataset
Dataset: Polish Companies Bankruptcy (3 year)

Results: 6_Results/12_Equivalent_Sample_Size_For_Each_Dataset/Polish_Bankruptcy_3Year/
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(Path(__file__).resolve().parent)

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils import data_preprocessing_pipeline

warnings.filterwarnings("ignore")

TARGET = "target"
FOLDER = "Polish_Bankruptcy_3Year"
DATA_PATH = "../../../1_Data/Processed_Datasets/Polish_Bankruptcy_3Year/processed_data.csv"
PCA_COMPONENTS = 10
PERCENTAGES = [10, 20]
RANDOM_STATE = 42
DATASET_KEY = "polish_bankruptcy"
DATASET_TO_USE = "polish_bankruptcy"

from utils import (
    train_multiple_dataset_tda,
    generate_landmark_sets,
    compute_barcodes_from_multiple_landmarks,
    build_final_barcode_statistics_data,
    store_results,
)

EXPERIMENT_NAME = "12_Equivalent_Sample_Size_For_Each_Dataset"

# =============================================================================
# Load and preprocess data
# =============================================================================
dataset = pd.read_csv(os.path.abspath(DATA_PATH))
for col in dataset.columns:
    if col != "target" and dataset[col].isnull().any():
        dataset[col] = dataset[col].fillna(dataset[col].median())
dataset = data_preprocessing_pipeline(dataset)
X = dataset.drop(columns=[TARGET])
y = dataset[TARGET]

# =============================================================================
# Normalize features
# =============================================================================
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# =============================================================================
# Apply PCA
# =============================================================================
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_reduced = pd.DataFrame(
    pca.fit_transform(X_normalized),
    columns=[f"PCA_{num}" for num in range(1, PCA_COMPONENTS + 1)],
)
variance_ratio = pca.explained_variance_ratio_.sum()
print(f"Variance retained with PCA components: {variance_ratio:.2%}")

reduced_data = X_reduced.copy()
reduced_data["Class"] = y

default_data = reduced_data[reduced_data["Class"] == 1].reset_index(drop=True)
non_default_data = reduced_data[reduced_data["Class"] == 0].reset_index(drop=True)
n_samples = len(default_data)
balanced_non_default = non_default_data.sample(n=n_samples, random_state=RANDOM_STATE)


# =============================================================================
# Matched landmark percentages (Statlog L30/L60 reference landmark counts)
# =============================================================================
n_majority = int(max((y == 0).sum(), (y == 1).sum()))
percentages = [
    round(90 / n_majority * 100, 2),
    round(180 / n_majority * 100, 2),
]
print(f"Matched landmark percentages: {percentages}")

# =============================================================================
# Landmark selection
# =============================================================================
generate_landmark_sets(
    class_label_and_data={
        "default": default_data.copy().drop("Class", axis=1),
        "non-default": balanced_non_default.copy().drop("Class", axis=1),
    },
    landmark_percentages=percentages,
    dataset_to_use=DATASET_TO_USE,
    n_files_per_percentage=500,
    experiment_name=EXPERIMENT_NAME,
)

# =============================================================================
# Compute barcode statistics
# =============================================================================
compute_barcodes_from_multiple_landmarks(
    landmark_percentages=percentages,
    landmark_dir=f"../../../1_Data/Landmark_Sets/{FOLDER}/{EXPERIMENT_NAME}",
    barcode_output_dir=f"../../../1_Data/Barcode_Statistics/{FOLDER}/{EXPERIMENT_NAME}",
    dim=2,
    label={1: "default", 0: "non-default"},
)

# =============================================================================
# Merge barcode statistics — build TDA datasets
# =============================================================================
build_final_barcode_statistics_data(
    landmark_percentages=percentages,
    barcode_dir=f"../../../1_Data/Barcode_Statistics/{FOLDER}/{EXPERIMENT_NAME}",
    output_dir=f"../../../1_Data/TDA_Datasets/{FOLDER}/{EXPERIMENT_NAME}",
    label={1: "default", 0: "non-default"},
)

# =============================================================================
# Train models
# =============================================================================
paths = [
    f"../../../1_Data/TDA_Datasets/{FOLDER}/{EXPERIMENT_NAME}/data_L{p}.csv"
    for p in percentages
]
model_results = train_multiple_dataset_tda(
    path_datasets=paths,
    y_col_name="label",
    test_size=0.2,
    random_state=RANDOM_STATE,
    xgb={"eval_metric": "logloss"},
)
print(model_results)

# =============================================================================
# Store results
# =============================================================================
store_results(
    path=f"../../../6_Results/{EXPERIMENT_NAME}/{FOLDER}",
    save_name="model_results",
    result_object=model_results,
)
