# -*- coding: utf-8 -*-
"""
Experiment 13 — Similar Variance Retained After PCA
Dataset: Taiwanese Bankruptcy Prediction

Results: 6_Results/Archives/13_Similar_Variance_Retained_After_PCA/Taiwan_Bankruptcy/
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
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
FOLDER = "Taiwan_Bankruptcy"
DATA_PATH = "../../../../1_Data/Processed_Datasets/Taiwan_Bankruptcy/processed_data.csv"
PCA_COMPONENTS = 10
PERCENTAGES = [10, 20]
RANDOM_STATE = 42
DATASET_KEY = "taiwan_bankruptcy"
DATASET_TO_USE = "taiwan_bankruptcy"

from utils import (
    train_multiple_dataset_tda,
    generate_landmark_sets,
    compute_barcodes_from_multiple_landmarks,
    build_final_barcode_statistics_data,
    store_results,
)

EXPERIMENT_NAME = "13_Similar_Variance_Retained_After_PCA"
VARIANCE_TARGET = 0.90

# =============================================================================
# Load and preprocess data
# =============================================================================
dataset = pd.read_csv(os.path.abspath(DATA_PATH))
dataset = data_preprocessing_pipeline(dataset)
X = dataset.drop(columns=[TARGET])
y = dataset[TARGET]

scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# =============================================================================
# PCA components matched to reference variance (~90%)
# =============================================================================
pca_probe = PCA(random_state=RANDOM_STATE)
pca_probe.fit(X_normalized)
cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
PCA_COMPONENTS = int(np.searchsorted(cumvar, VARIANCE_TARGET) + 1)
PCA_COMPONENTS = min(PCA_COMPONENTS, X_normalized.shape[1])
X_reduced = pd.DataFrame(
    PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE).fit_transform(X_normalized),
    columns=[f"PCA_{num}" for num in range(1, PCA_COMPONENTS + 1)],
)
variance_ratio = float(cumvar[PCA_COMPONENTS - 1])
print(f"Matched-variance PCA components={PCA_COMPONENTS}, variance={variance_ratio:.2%}")

reduced_data = X_reduced.copy()
reduced_data["Class"] = y
default_data = reduced_data[reduced_data["Class"] == 1].reset_index(drop=True)
non_default_data = reduced_data[reduced_data["Class"] == 0].reset_index(drop=True)
n_samples = len(default_data)
balanced_non_default = non_default_data.sample(n=n_samples, random_state=RANDOM_STATE)

percentages = PERCENTAGES

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

compute_barcodes_from_multiple_landmarks(
    landmark_percentages=percentages,
    landmark_dir=f"../../../../1_Data/Landmark_Sets/{FOLDER}/{EXPERIMENT_NAME}",
    barcode_output_dir=f"../../../../1_Data/Barcode_Statistics/{FOLDER}/{EXPERIMENT_NAME}",
    dim=2,
    label={1: "default", 0: "non-default"},
)

build_final_barcode_statistics_data(
    landmark_percentages=percentages,
    barcode_dir=f"../../../../1_Data/Barcode_Statistics/{FOLDER}/{EXPERIMENT_NAME}",
    output_dir=f"../../../../1_Data/TDA_Datasets/{FOLDER}/{EXPERIMENT_NAME}",
    label={1: "default", 0: "non-default"},
)

paths = [
    f"../../../../1_Data/TDA_Datasets/{FOLDER}/{EXPERIMENT_NAME}/data_L{p}.csv"
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

store_results(
    path=f"../../../../6_Results/{EXPERIMENT_NAME}/{FOLDER}",
    save_name="model_results",
    result_object=model_results,
)
