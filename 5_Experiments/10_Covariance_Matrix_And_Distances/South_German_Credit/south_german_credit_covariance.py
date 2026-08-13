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

import pandas as pd
import warnings
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)
from utils import rename_barcode_statistics_columns, get_distance_view, store_results

warnings.filterwarnings("ignore")

# =============================================================================
# GET DATASET
# =============================================================================
TDA_DIR = ROOT / "1_Data" / "TDA_Datasets" / "South_German_Credit" / "3_PH_Default_Parameters"
data_L10 = pd.read_csv(TDA_DIR / "data_L10.csv")
data_L20 = pd.read_csv(TDA_DIR / "data_L20.csv")

data_L10 = rename_barcode_statistics_columns(data_L10)
data_L20 = rename_barcode_statistics_columns(data_L20)

# =============================================================================
# COMPUTE DISTANCES - L10
# =============================================================================
distance_matrix_L10 = get_distance_view(
    data_L10, target_col="label", return_class_view=False, random_state=42
)
distance_mean_centriod_L10 = get_distance_view(
    data_L10, target_col="label", return_class_view=True, centroid_method="mean", random_state=42
)
distance_farthest_centriod_L10 = get_distance_view(
    data_L10, target_col="label", return_class_view=True, centroid_method="farthest", random_state=42
)
distance_random_centriod_L10 = get_distance_view(
    data_L10, target_col="label", return_class_view=True, centroid_method="random", random_state=42
)

# =============================================================================
# COMPUTE DISTANCES - L20
# =============================================================================
distance_matrix_L20 = get_distance_view(
    data_L20, target_col="label", return_class_view=False, random_state=42
)
distance_mean_centriod_L20 = get_distance_view(
    data_L20, target_col="label", return_class_view=True, centroid_method="mean", random_state=42
)
distance_farthest_centriod_L20 = get_distance_view(
    data_L20, target_col="label", return_class_view=True, centroid_method="farthest", random_state=42
)
distance_random_centriod_L20 = get_distance_view(
    data_L20, target_col="label", return_class_view=True, centroid_method="random", random_state=42
)

# =============================================================================
# METRICS - L10
# =============================================================================
y_pred_mean_L10 = distance_mean_centriod_L10["closest_class"]
y_true_mean_L10 = distance_mean_centriod_L10["actual_class"]
y_pred_farthest_L10 = distance_farthest_centriod_L10["closest_class"]
y_true_farthest_L10 = distance_farthest_centriod_L10["actual_class"]
y_pred_random_L10 = distance_random_centriod_L10["closest_class"]
y_true_random_L10 = distance_random_centriod_L10["actual_class"]

L10_results = {
    "confusion_matrix_mean_L10": confusion_matrix(y_true_mean_L10, y_pred_mean_L10),
    "classification_report_mean_L10": classification_report(y_true_mean_L10, y_pred_mean_L10),
    "accuracy_mean_L10": accuracy_score(y_true_mean_L10, y_pred_mean_L10),
    "precision_mean_L10": precision_score(y_true_mean_L10, y_pred_mean_L10),
    "recall_mean_L10": recall_score(y_true_mean_L10, y_pred_mean_L10),
    "f1_mean_L10": f1_score(y_true_mean_L10, y_pred_mean_L10),
    "confusion_matrix_farthest_L10": confusion_matrix(y_true_farthest_L10, y_pred_farthest_L10),
    "classification_report_farthest_L10": classification_report(y_true_farthest_L10, y_pred_farthest_L10),
    "accuracy_farthest_L10": accuracy_score(y_true_farthest_L10, y_pred_farthest_L10),
    "precision_farthest_L10": precision_score(y_true_farthest_L10, y_pred_farthest_L10),
    "recall_farthest_L10": recall_score(y_true_farthest_L10, y_pred_farthest_L10),
    "f1_farthest_L10": f1_score(y_true_farthest_L10, y_pred_farthest_L10),
    "confusion_matrix_random_L10": confusion_matrix(y_true_random_L10, y_pred_random_L10),
    "classification_report_random_L10": classification_report(y_true_random_L10, y_pred_random_L10),
    "accuracy_random_L10": accuracy_score(y_true_random_L10, y_pred_random_L10),
    "precision_random_L10": precision_score(y_true_random_L10, y_pred_random_L10),
    "recall_random_L10": recall_score(y_true_random_L10, y_pred_random_L10),
    "f1_random_L10": f1_score(y_true_random_L10, y_pred_random_L10),
}

# =============================================================================
# METRICS - L20
# =============================================================================
y_pred_mean_L20 = distance_mean_centriod_L20["closest_class"]
y_true_mean_L20 = distance_mean_centriod_L20["actual_class"]
y_pred_farthest_L20 = distance_farthest_centriod_L20["closest_class"]
y_true_farthest_L20 = distance_farthest_centriod_L20["actual_class"]
y_pred_random_L20 = distance_random_centriod_L20["closest_class"]
y_true_random_L20 = distance_random_centriod_L20["actual_class"]

L20_results = {
    "confusion_matrix_mean_L20": confusion_matrix(y_true_mean_L20, y_pred_mean_L20),
    "classification_report_mean_L20": classification_report(y_true_mean_L20, y_pred_mean_L20),
    "accuracy_mean_L20": accuracy_score(y_true_mean_L20, y_pred_mean_L20),
    "precision_mean_L20": precision_score(y_true_mean_L20, y_pred_mean_L20),
    "recall_mean_L20": recall_score(y_true_mean_L20, y_pred_mean_L20),
    "f1_mean_L20": f1_score(y_true_mean_L20, y_pred_mean_L20),
    "confusion_matrix_farthest_L20": confusion_matrix(y_true_farthest_L20, y_pred_farthest_L20),
    "classification_report_farthest_L20": classification_report(y_true_farthest_L20, y_pred_farthest_L20),
    "accuracy_farthest_L20": accuracy_score(y_true_farthest_L20, y_pred_farthest_L20),
    "precision_farthest_L20": precision_score(y_true_farthest_L20, y_pred_farthest_L20),
    "recall_farthest_L20": recall_score(y_true_farthest_L20, y_pred_farthest_L20),
    "f1_farthest_L20": f1_score(y_true_farthest_L20, y_pred_farthest_L20),
    "confusion_matrix_random_L20": confusion_matrix(y_true_random_L20, y_pred_random_L20),
    "classification_report_random_L20": classification_report(y_true_random_L20, y_pred_random_L20),
    "accuracy_random_L20": accuracy_score(y_true_random_L20, y_pred_random_L20),
    "precision_random_L20": precision_score(y_true_random_L20, y_pred_random_L20),
    "recall_random_L20": recall_score(y_true_random_L20, y_pred_random_L20),
    "f1_random_L20": f1_score(y_true_random_L20, y_pred_random_L20),
}

# =============================================================================
# STORE RESULTS - L10
# =============================================================================
save_path = str(ROOT / "6_Results/10_Covariance_Matrix_And_Distances/South_German_Credit/L10")
store_results(path=save_path, save_name="L10_results", result_object=L10_results)
store_results(path=save_path, save_name="distance_matrix_L10", result_object=distance_matrix_L10)
store_results(path=save_path, save_name="distance_mean_centriod_L10", result_object=distance_mean_centriod_L10)
store_results(path=save_path, save_name="distance_farthest_centriod_L10", result_object=distance_farthest_centriod_L10)
store_results(path=save_path, save_name="distance_random_centriod_L10", result_object=distance_random_centriod_L10)

# =============================================================================
# STORE RESULTS - L20
# =============================================================================
save_path = str(ROOT / "6_Results/10_Covariance_Matrix_And_Distances/South_German_Credit/L20")
store_results(path=save_path, save_name="L20_results", result_object=L20_results)
store_results(path=save_path, save_name="distance_matrix_L20", result_object=distance_matrix_L20)
store_results(path=save_path, save_name="distance_mean_centriod_L20", result_object=distance_mean_centriod_L20)
store_results(path=save_path, save_name="distance_farthest_centriod_L20", result_object=distance_farthest_centriod_L20)
store_results(path=save_path, save_name="distance_random_centriod_L20", result_object=distance_random_centriod_L20)
