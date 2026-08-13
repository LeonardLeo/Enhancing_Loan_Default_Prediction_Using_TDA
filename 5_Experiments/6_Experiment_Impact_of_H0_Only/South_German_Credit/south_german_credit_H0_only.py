# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 2026

Dataset: South German Credit

@author: lEO
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
    os.path.abspath("../../../1_Data/Processed_Datasets/South_German_Credit/processed_data.csv")
)

# =============================================================================
# Preprocessing data
# =============================================================================
data = data_preprocessing_pipeline(
    data,
    log_col=["hoehe", "laufzeit"],
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
    dataset_to_use="south_german_credit",
    n_files_per_percentage=500,
    experiment_name="6_Experiment_Impact_of_H0_Only",
)

# =============================================================================
# COMPUTE BARCODE STATISTICS
# =============================================================================
compute_barcodes_from_multiple_landmarks(
    landmark_percentages=percentages,
    landmark_dir="../../../1_Data/Landmark_Sets/South_German_Credit/6_Experiment_Impact_of_H0_Only",
    barcode_output_dir="../../../1_Data/Barcode_Statistics/South_German_Credit/6_Experiment_Impact_of_H0_Only",
    dim=1,
    label={1: "default", 0: "non-default"},
)

# =============================================================================
# MERGE BARCODE STATISTICS - Create TDA Dataset for Model Building
# =============================================================================
build_final_barcode_statistics_data(
    landmark_percentages=percentages,
    barcode_dir="../../../1_Data/Barcode_Statistics/South_German_Credit/6_Experiment_Impact_of_H0_Only",
    output_dir="../../../1_Data/TDA_Datasets/South_German_Credit/6_Experiment_Impact_of_H0_Only",
    label={1: "default", 0: "non-default"},
)

# =============================================================================
# TRAIN MACHINE LEARNING MODEL
# =============================================================================
paths = [
    "../../../1_Data/TDA_Datasets/South_German_Credit/6_Experiment_Impact_of_H0_Only/data_L10.csv",
    "../../../1_Data/TDA_Datasets/South_German_Credit/6_Experiment_Impact_of_H0_Only/data_L20.csv",
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
save_path = "../../../6_Results/6_Experiment_Impact_of_H0_Only/South_German_Credit"
store_results(path=save_path, save_name="model_results", result_object=model_results)
