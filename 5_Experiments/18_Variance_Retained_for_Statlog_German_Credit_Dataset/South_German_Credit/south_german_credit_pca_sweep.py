# -*- coding: utf-8 -*-
"""
Experiment 18 — PCA variance sweep
Dataset: South German Credit

Results: 6_Results/18_Variance_Retained_for_Statlog_German_Credit_Dataset/South_German_Credit/
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
FOLDER = "South_German_Credit"
DATA_PATH = "../../../1_Data/Processed_Datasets/South_German_Credit/processed_data.csv"
PCA_COMPONENTS = 10
PERCENTAGES = [10, 20]
RANDOM_STATE = 42
DATASET_KEY = "south_german_credit"
DATASET_TO_USE = "south_german_credit"

from utils import (
    generate_landmark_sets,
    compute_barcodes_from_multiple_landmarks,
    build_final_barcode_statistics_data,
    train_multiple_dataset_tda,
    plot_all_metrics_vs_pca_components,
    store_results,
)

EXPERIMENT_NAME = "18_Variance_Retained_for_Statlog_German_Credit_Dataset"
COMPONENTS_LIST = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# =============================================================================
# Load and preprocess data
# =============================================================================
dataset = pd.read_csv(os.path.abspath(DATA_PATH))
dataset = data_preprocessing_pipeline(
    dataset,
    log_col=["hoehe", "laufzeit"],
)
X = dataset.drop(columns=[TARGET])
y = dataset[TARGET]
all_results = {}

for n_components in COMPONENTS_LIST:
    if n_components > X.shape[1]:
        continue
    add_optional_path = f"Using_{n_components}_Components"
    landmark_dir = f"../../../1_Data/Landmark_Sets/{FOLDER}/{EXPERIMENT_NAME}/{add_optional_path}"
    barcode_dir = f"../../../1_Data/Barcode_Statistics/{FOLDER}/{EXPERIMENT_NAME}/{add_optional_path}"
    tda_dir = f"../../../1_Data/TDA_Datasets/{FOLDER}/{EXPERIMENT_NAME}/{add_optional_path}"
    print(f"\nRunning PCA sweep with {n_components} components")

    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_reduced = pd.DataFrame(
        pca.fit_transform(X_normalized),
        columns=[f"PCA_{i + 1}" for i in range(n_components)],
    )
    variance_ratio = pca.explained_variance_ratio_.sum()
    print(f"Variance retained: {variance_ratio:.2%}")

    reduced_data = X_reduced.copy()
    reduced_data["Class"] = y
    default_data = reduced_data[reduced_data["Class"] == 1].reset_index(drop=True)
    non_default_data = reduced_data[reduced_data["Class"] == 0].reset_index(drop=True)
    n_samples = len(default_data)
    balanced_non_default = non_default_data.sample(n=n_samples, random_state=RANDOM_STATE)

    generate_landmark_sets(
        class_label_and_data={
            "default": default_data.drop("Class", axis=1),
            "non-default": balanced_non_default.drop("Class", axis=1),
        },
        landmark_percentages=PERCENTAGES,
        dataset_to_use=DATASET_TO_USE,
        experiment_name=EXPERIMENT_NAME,
        add_optional_path=add_optional_path,
    )
    compute_barcodes_from_multiple_landmarks(
        landmark_percentages=PERCENTAGES,
        landmark_dir=landmark_dir,
        barcode_output_dir=barcode_dir,
        dim=2,
        label={1: "default", 0: "non-default"},
    )
    build_final_barcode_statistics_data(
        landmark_percentages=PERCENTAGES,
        barcode_dir=barcode_dir,
        output_dir=tda_dir,
        label={1: "default", 0: "non-default"},
    )
    dataset_paths = [f"{tda_dir}/data_L{p}.csv" for p in PERCENTAGES]
    model_results = train_multiple_dataset_tda(
        path_datasets=dataset_paths,
        y_col_name="label",
        test_size=0.2,
        random_state=RANDOM_STATE,
        xgb={"eval_metric": "logloss"},
    )
    store_results(
        path=f"../../../6_Results/{EXPERIMENT_NAME}/{FOLDER}",
        save_name=f"model_results_using_{n_components}_components",
        result_object=model_results,
    )
    all_results[f"model_results_using_{n_components}_components"] = {
        "variance_retained": variance_ratio,
        "results": model_results,
    }

store_results(
    path=f"../../../6_Results/{EXPERIMENT_NAME}/{FOLDER}",
    save_name="model_results",
    result_object=all_results,
)

viz_base = f"../../../6_Results/{EXPERIMENT_NAME}/{FOLDER}/viz"
for model_key in ("knn", "svm", "xgb", "logistic", "random_forest"):
    plot_all_metrics_vs_pca_components(all_results=all_results, model_key=model_key, save_path=viz_base, separate_plots=True)
    plot_all_metrics_vs_pca_components(all_results=all_results, model_key=model_key, save_path=viz_base, separate_plots=False)
