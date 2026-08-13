# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 2026

Dataset: Taiwanese Bankruptcy Prediction

@author: lEO
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import os
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import (
    data_preprocessing_pipeline,
    train_models_on_multiple_datasets,
    generate_landmark_sets,
    compute_barcodes_from_multiple_landmarks,
    build_final_barcode_statistics_data,
    store_results,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_csv(
    os.path.abspath("../../../1_Data/Processed_Datasets/Taiwan_Bankruptcy/processed_data.csv")
)

# =============================================================================
# Preprocessing data
# =============================================================================
data = data_preprocessing_pipeline(data)

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
    dataset_to_use="taiwan_bankruptcy",
    n_files_per_percentage=500,
    experiment_name="4_PH_Tuned_Parameters",
)

# =============================================================================
# COMPUTE BARCODE STATISTICS
# =============================================================================
compute_barcodes_from_multiple_landmarks(
    landmark_percentages=percentages,
    landmark_dir="../../../1_Data/Landmark_Sets/Taiwan_Bankruptcy/4_PH_Tuned_Parameters",
    barcode_output_dir="../../../1_Data/Barcode_Statistics/Taiwan_Bankruptcy/4_PH_Tuned_Parameters",
    dim=2,
    label={1: "default", 0: "non-default"},
)

# =============================================================================
# MERGE BARCODE STATISTICS - Create TDA Dataset for Model Building
# =============================================================================
build_final_barcode_statistics_data(
    landmark_percentages=percentages,
    barcode_dir="../../../1_Data/Barcode_Statistics/Taiwan_Bankruptcy/4_PH_Tuned_Parameters",
    output_dir="../../../1_Data/TDA_Datasets/Taiwan_Bankruptcy/4_PH_Tuned_Parameters",
    label={1: "default", 0: "non-default"},
)

# =============================================================================
# TRAIN MACHINE LEARNING MODEL WITH TUNED PARAMETERS
# =============================================================================
paths = [
    "../../../1_Data/TDA_Datasets/Taiwan_Bankruptcy/4_PH_Tuned_Parameters/data_L10.csv",
    "../../../1_Data/TDA_Datasets/Taiwan_Bankruptcy/4_PH_Tuned_Parameters/data_L20.csv",
]

model_configs = {
    "svm": {
        "model": SVC(),
        "params": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "degree": [2, 3, 4],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        },
    },
    "knn": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 10],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "leaf_size": [10, 20, 30, 50],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        },
    },
    "xgb": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    },
    "logistic": {
        "model": LogisticRegression(),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear", "lbfgs", "sag", "saga", "newton-cg"],
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "max_iter": [100, 200, 500],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
        },
    },
}

model_results = train_models_on_multiple_datasets(
    data_paths=paths,
    model_configs=model_configs,
    target_column="label",
    test_size=0.2,
    scoring_metric="f1",
    scale_features=True,
    random_state=42,
    n_splits_kfold=5,
)

print(model_results)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../6_Results/4_PH_Tuned_Parameters/Taiwan_Bankruptcy"
store_results(path=save_path, save_name="model_results", result_object=model_results)
