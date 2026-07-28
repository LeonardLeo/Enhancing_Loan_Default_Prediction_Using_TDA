# -*- coding: utf-8 -*-
"""
Experiment 23 — Early 80/20 split BEFORE PCA / landmark snapshots (Protocol B).
Statlog German Credit Dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
REPO_ROOT = Path(__file__).resolve().parents[3]

import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from utils import (
    stratified_early_split,
    fit_scaler_pca_on_train,
    balance_binary_by_undersampling,
    generate_landmark_sets,
    compute_barcodes_from_multiple_landmarks,
    build_final_barcode_statistics_data,
    train_multiple_dataset_tda_presplit,
    train_models_on_multiple_presplit_datasets,
    store_results,
    store_data_as_csv_or_json,
)

warnings.filterwarnings("ignore")

import os
os.chdir(Path(__file__).resolve().parent)

EXPERIMENT_NAME = "23_Early_Train_Test_Split"
DATASET_KEY = "Statlog"
PCA_COMPONENTS = 15
PERCENTAGES = [30, 60]
N_FILES = 500
TARGET = "Class"
RANDOM_STATE = 42

data = pd.read_excel(
    REPO_ROOT
    / "1_Data/Processed_Datasets/Statlog_German_Credit_Data/processed_data.xlsx"
)
X = data.drop(columns=[TARGET])
y = data[TARGET]

X_train, X_test, y_train, y_test = stratified_early_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

split_meta = {
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().items()},
    "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
    "pca_components": PCA_COMPONENTS,
    "landmark_percentages": PERCENTAGES,
    "n_files_per_percentage": N_FILES,
    "protocol": "B_independent_train_test_landmarks",
}
store_data_as_csv_or_json(
    path="../../../6_Results/23_Early_Train_Test_Split/Statlog_German_Credit_Data",
    csv=False,
    save_as=["split_metadata"],
    data_object=[split_meta],
)

X_train_pca, X_test_pca, scaler, pca, var_ratio = fit_scaler_pca_on_train(
    X_train, X_test, n_components=PCA_COMPONENTS, random_state=RANDOM_STATE
)
print(f"Variance retained (train-fit PCA): {var_ratio:.2%}")

train_balanced = balance_binary_by_undersampling(
    X_train_pca, y_train, positive_label=1, random_state=RANDOM_STATE
)
test_balanced = balance_binary_by_undersampling(
    X_test_pca, y_test, positive_label=1, random_state=RANDOM_STATE
)

for split_name, balanced in [("train", train_balanced), ("test", test_balanced)]:
    default_df = balanced[balanced["Class"] == 1].drop(columns=["Class"])
    non_default_df = balanced[balanced["Class"] == 0].drop(columns=["Class"])
    generate_landmark_sets(
        class_label_and_data={
            "default": default_df,
            "non-default": non_default_df,
        },
        landmark_percentages=PERCENTAGES,
        dataset_to_use=DATASET_KEY,
        n_files_per_percentage=N_FILES,
        experiment_name=EXPERIMENT_NAME,
        add_optional_path=split_name,
    )

for split_name in ["train", "test"]:
    landmark_dir = (
        f"../../../1_Data/Landmark_Sets/Statlog_German_Credit_Data/"
        f"{EXPERIMENT_NAME}/{split_name}"
    )
    barcode_dir = (
        f"../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/"
        f"{EXPERIMENT_NAME}/{split_name}"
    )
    tda_dir = (
        f"../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/"
        f"{EXPERIMENT_NAME}/{split_name}"
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

base = "../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/23_Early_Train_Test_Split"
train_test_pairs = {
    f"data_L{p}": {
        "train": f"{base}/train/data_L{p}.csv",
        "test": f"{base}/test/data_L{p}.csv",
    }
    for p in PERCENTAGES
}

default_results = train_multiple_dataset_tda_presplit(
    train_test_pairs=train_test_pairs,
    y_col_name="label",
    random_state=RANDOM_STATE,
    xgb={"eval_metric": "logloss"},
)
store_results(
    path="../../../6_Results/23_Early_Train_Test_Split/Statlog_German_Credit_Data",
    save_name="model_results_default",
    result_object=default_results,
)

model_configs = {
    "svm": {
        "model": SVC(),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    },
    "knn": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "p": [1, 2]},
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
        "model": LogisticRegression(max_iter=1000),
        "params": {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
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

tuned_results = train_models_on_multiple_presplit_datasets(
    train_test_pairs=train_test_pairs,
    model_configs=model_configs,
    target_column="label",
    scoring_metric="f1",
    scale_features=True,
    random_state=RANDOM_STATE,
    n_splits_kfold=5,
)
store_results(
    path="../../../6_Results/23_Early_Train_Test_Split/Statlog_German_Credit_Data",
    save_name="model_results_tuned",
    result_object=tuned_results,
)

print("Experiment 23 (SGCD) complete.")
