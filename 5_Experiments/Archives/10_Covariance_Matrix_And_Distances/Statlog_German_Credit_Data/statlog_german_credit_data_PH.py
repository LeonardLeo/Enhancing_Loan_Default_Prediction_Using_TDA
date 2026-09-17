# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:59:58 2024

@author: lEO
"""

import pandas as pd
import warnings
from sklearn.metrics import (confusion_matrix, 
                             classification_report, 
                             accuracy_score,
                             precision_score,
                             f1_score,
                             recall_score)
from utils import (rename_barcode_statistics_columns,
                   get_distance_view,
                   store_results)

# Handle warnings
warnings.filterwarnings("ignore")

# =============================================================================
# GET DATASET
# =============================================================================
data_L30 = pd.read_csv("../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/data_L30.csv")
data_L60 = pd.read_csv("../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/data_L60.csv")

data_L30 = rename_barcode_statistics_columns(data_L30)
data_L60 = rename_barcode_statistics_columns(data_L60)

# =============================================================================
# COMPUTE DISTANCES - L30
# =============================================================================
# Distance matrix
distance_matrix_L30 = get_distance_view(data_L30, 
                                    target_col='label', 
                                    return_class_view=False,  
                                    random_state=42)

# Traditional class means (supervised)
distance_mean_centriod_L30 = get_distance_view(data_L30, 
                                    target_col='label', 
                                    return_class_view=True, 
                                    centroid_method='mean', 
                                    random_state=42)

# Using the two farthest points as pseudo-centroids (unsupervised)
distance_farthest_centriod_L30 = get_distance_view(data_L30, 
                                    target_col='label', 
                                    return_class_view=True, 
                                    centroid_method='farthest', 
                                    random_state=42)

# Random selection of centroids
distance_random_centriod_L30 = get_distance_view(data_L30, 
                                    target_col='label', 
                                    return_class_view=True, 
                                    centroid_method='random', 
                                    random_state=42)

# =============================================================================
# COMPUTE DISTANCES - L60
# =============================================================================
# Distance matrix
distance_matrix_L60 = get_distance_view(data_L60, 
                                        target_col='label', 
                                        return_class_view=False,  
                                        random_state=42)

# Traditional class means (supervised)
distance_mean_centriod_L60 = get_distance_view(data_L60, 
                                        target_col='label', 
                                        return_class_view=True, 
                                        centroid_method='mean', 
                                        random_state=42)

# Using the two farthest points as pseudo-centroids (unsupervised)
distance_farthest_centriod_L60 = get_distance_view(data_L60, 
                                        target_col='label', 
                                        return_class_view=True, 
                                        centroid_method='farthest', 
                                        random_state=42)

# Random selection of centroids
distance_random_centriod_L60 = get_distance_view(data_L60, 
                                        target_col='label', 
                                        return_class_view=True, 
                                        centroid_method='random', 
                                        random_state=42)

# =============================================================================
# METRICS - L30
# =============================================================================
y_pred_mean_L30 = distance_mean_centriod_L30["closest_class"]
y_true_mean_L30 = distance_mean_centriod_L30["actual_class"]

y_pred_farthest_L30 = distance_farthest_centriod_L30["closest_class"]
y_true_farthest_L30 = distance_farthest_centriod_L30["actual_class"]

y_pred_random_L30 = distance_random_centriod_L30["closest_class"]
y_true_random_L30 = distance_random_centriod_L30["actual_class"]

# -----------------------------------------------------------------------------
L30_results = {
        "confusion_matrix_mean_L30": confusion_matrix(y_true_mean_L30, y_pred_mean_L30),
        "classification_report_mean_L30": classification_report(y_true_mean_L30, y_pred_mean_L30),
        "accuracy_mean_L30": accuracy_score(y_true_mean_L30, y_pred_mean_L30),
        "precision_mean_L30": precision_score(y_true_mean_L30, y_pred_mean_L30),
        "recall_mean_L30": recall_score(y_true_mean_L30, y_pred_mean_L30),
        "f1_mean_L30": f1_score(y_true_mean_L30, y_pred_mean_L30),
        
        "confusion_matrix_farthest_L30": confusion_matrix(y_true_farthest_L30, y_pred_farthest_L30),
        "classification_report_farthest_L30": classification_report(y_true_farthest_L30, y_pred_farthest_L30),
        "accuracy_farthest_L30": accuracy_score(y_true_farthest_L30, y_pred_farthest_L30),
        "precision_farthest_L30": precision_score(y_true_farthest_L30, y_pred_farthest_L30),
        "recall_farthest_L30": recall_score(y_true_farthest_L30, y_pred_farthest_L30),
        "f1_farthest_L30": f1_score(y_true_farthest_L30, y_pred_farthest_L30),
        
        "confusion_matrix_random_L30": confusion_matrix(y_true_random_L30, y_pred_random_L30),
        "classification_report_random_L30": classification_report(y_true_random_L30, y_pred_random_L30),
        "accuracy_random_L30": accuracy_score(y_true_random_L30, y_pred_random_L30),
        "precision_random_L30": precision_score(y_true_random_L30, y_pred_random_L30),
        "recall_random_L30": recall_score(y_true_random_L30, y_pred_random_L30),
        "f1_random_L30": f1_score(y_true_random_L30, y_pred_random_L30),
}

# =============================================================================
# METRICS - L60
# =============================================================================
y_pred_mean_L60 = distance_mean_centriod_L60["closest_class"]
y_true_mean_L60 = distance_mean_centriod_L60["actual_class"]

y_pred_farthest_L60 = distance_farthest_centriod_L60["closest_class"]
y_true_farthest_L60 = distance_farthest_centriod_L60["actual_class"]

y_pred_random_L60 = distance_random_centriod_L60["closest_class"]
y_true_random_L60 = distance_random_centriod_L60["actual_class"]

# -----------------------------------------------------------------------------
L60_results = {
        "confusion_matrix_mean_L60": confusion_matrix(y_true_mean_L60, y_pred_mean_L60),
        "classification_report_mean_L60": classification_report(y_true_mean_L60, y_pred_mean_L60),
        "accuracy_mean_L60": accuracy_score(y_true_mean_L60, y_pred_mean_L60),
        "precision_mean_L60": precision_score(y_true_mean_L60, y_pred_mean_L60),
        "recall_mean_L60": recall_score(y_true_mean_L60, y_pred_mean_L60),
        "f1_mean_L60": f1_score(y_true_mean_L60, y_pred_mean_L60),
        
        "confusion_matrix_farthest_L60": confusion_matrix(y_true_farthest_L60, y_pred_farthest_L60),
        "classification_report_farthest_L60": classification_report(y_true_farthest_L60, y_pred_farthest_L60),
        "accuracy_farthest_L60": accuracy_score(y_true_farthest_L60, y_pred_farthest_L60),
        "precision_farthest_L60": precision_score(y_true_farthest_L60, y_pred_farthest_L60),
        "recall_farthest_L60": recall_score(y_true_farthest_L60, y_pred_farthest_L60),
        "f1_farthest_L60": f1_score(y_true_farthest_L60, y_pred_farthest_L60),
        
        "confusion_matrix_random_L60": confusion_matrix(y_true_random_L60, y_pred_random_L60),
        "classification_report_random_L60": classification_report(y_true_random_L60, y_pred_random_L60),
        "accuracy_random_L60": accuracy_score(y_true_random_L60, y_pred_random_L60),
        "precision_random_L60": precision_score(y_true_random_L60, y_pred_random_L60),
        "recall_random_L60": recall_score(y_true_random_L60, y_pred_random_L60),
        "f1_random_L60": f1_score(y_true_random_L60, y_pred_random_L60),
}

# =============================================================================
# STORE RESULTS - L30
# =============================================================================
save_path = "../../../../6_Results/Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L30"

store_results(path = save_path, 
              save_name = "L30_results", 
              result_object = L30_results)

store_results(path = save_path, 
              save_name = "distance_matrix_L30", 
              result_object = distance_matrix_L30)

store_results(path = save_path, 
              save_name = "distance_mean_centriod_L30", 
              result_object = distance_mean_centriod_L30)

store_results(path = save_path, 
              save_name = "distance_farthest_centriod_L30", 
              result_object = distance_farthest_centriod_L30)

store_results(path = save_path, 
              save_name = "distance_random_centriod_L30", 
              result_object = distance_random_centriod_L30)

# =============================================================================
# STORE RESULTS - L60
# =============================================================================
save_path = "../../../../6_Results/Archives/10_Covariance_Matrix_And_Distances/Statlog_German_Credit_Data/L60"

store_results(path = save_path, 
              save_name = "L60_results", 
              result_object = L60_results)

store_results(path = save_path, 
              save_name = "distance_matrix_L60", 
              result_object = distance_matrix_L60)

store_results(path = save_path, 
              save_name = "distance_mean_centriod_L60", 
              result_object = distance_mean_centriod_L60)

store_results(path = save_path, 
              save_name = "distance_farthest_centriod_L60", 
              result_object = distance_farthest_centriod_L60)

store_results(path = save_path, 
              save_name = "distance_random_centriod_L60", 
              result_object = distance_random_centriod_L60)