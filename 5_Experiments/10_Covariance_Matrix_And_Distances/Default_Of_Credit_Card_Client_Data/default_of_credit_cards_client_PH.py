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
data_L5 = pd.read_csv("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L5.csv")
data_L15 = pd.read_csv("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L15.csv")

data_L5 = rename_barcode_statistics_columns(data_L5)
data_L15 = rename_barcode_statistics_columns(data_L15)

# =============================================================================
# COMPUTE DISTANCES - L5
# =============================================================================
# Distance matrix
distance_matrix_L5 = get_distance_view(data_L5, 
                                    target_col='label', 
                                    return_class_view=False,  
                                    random_state=42)

# Traditional class means (supervised)
distance_mean_centriod_L5 = get_distance_view(data_L5, 
                                    target_col='label', 
                                    return_class_view=True, 
                                    centroid_method='mean', 
                                    random_state=42)

# Using the two farthest points as pseudo-centroids (unsupervised)
distance_farthest_centriod_L5 = get_distance_view(data_L5, 
                                    target_col='label', 
                                    return_class_view=True, 
                                    centroid_method='farthest', 
                                    random_state=42)

# Random selection of centroids
distance_random_centriod_L5 = get_distance_view(data_L5, 
                                    target_col='label', 
                                    return_class_view=True, 
                                    centroid_method='random', 
                                    random_state=42)

# =============================================================================
# COMPUTE DISTANCES - L15
# =============================================================================
# Distance matrix
distance_matrix_L15 = get_distance_view(data_L15, 
                                        target_col='label', 
                                        return_class_view=False,  
                                        random_state=42)

# Traditional class means (supervised)
distance_mean_centriod_L15 = get_distance_view(data_L15, 
                                        target_col='label', 
                                        return_class_view=True, 
                                        centroid_method='mean', 
                                        random_state=42)

# Using the two farthest points as pseudo-centroids (unsupervised)
distance_farthest_centriod_L15 = get_distance_view(data_L15, 
                                        target_col='label', 
                                        return_class_view=True, 
                                        centroid_method='farthest', 
                                        random_state=42)

# Random selection of centroids
distance_random_centriod_L15 = get_distance_view(data_L15, 
                                        target_col='label', 
                                        return_class_view=True, 
                                        centroid_method='random', 
                                        random_state=42)

# =============================================================================
# METRICS - L5
# =============================================================================
y_pred_mean_L5 = distance_mean_centriod_L5["closest_class"]
y_true_mean_L5 = distance_mean_centriod_L5["actual_class"]

y_pred_farthest_L5 = distance_farthest_centriod_L5["closest_class"]
y_true_farthest_L5 = distance_farthest_centriod_L5["actual_class"]

y_pred_random_L5 = distance_random_centriod_L5["closest_class"]
y_true_random_L5 = distance_random_centriod_L5["actual_class"]

# -----------------------------------------------------------------------------
L5_results = {
        "confusion_matrix_mean_L5": confusion_matrix(y_true_mean_L5, y_pred_mean_L5),
        "classification_report_mean_L5": classification_report(y_true_mean_L5, y_pred_mean_L5),
        "accuracy_mean_L5": accuracy_score(y_true_mean_L5, y_pred_mean_L5),
        "precision_mean_L5": precision_score(y_true_mean_L5, y_pred_mean_L5),
        "recall_mean_L5": recall_score(y_true_mean_L5, y_pred_mean_L5),
        "f1_mean_L5": f1_score(y_true_mean_L5, y_pred_mean_L5),
        
        "confusion_matrix_farthest_L5": confusion_matrix(y_true_farthest_L5, y_pred_farthest_L5),
        "classification_report_farthest_L5": classification_report(y_true_farthest_L5, y_pred_farthest_L5),
        "accuracy_farthest_L5": accuracy_score(y_true_farthest_L5, y_pred_farthest_L5),
        "precision_farthest_L5": precision_score(y_true_farthest_L5, y_pred_farthest_L5),
        "recall_farthest_L5": recall_score(y_true_farthest_L5, y_pred_farthest_L5),
        "f1_farthest_L5": f1_score(y_true_farthest_L5, y_pred_farthest_L5),
        
        "confusion_matrix_random_L5": confusion_matrix(y_true_random_L5, y_pred_random_L5),
        "classification_report_random_L5": classification_report(y_true_random_L5, y_pred_random_L5),
        "accuracy_random_L5": accuracy_score(y_true_random_L5, y_pred_random_L5),
        "precision_random_L5": precision_score(y_true_random_L5, y_pred_random_L5),
        "recall_random_L5": recall_score(y_true_random_L5, y_pred_random_L5),
        "f1_random_L5": f1_score(y_true_random_L5, y_pred_random_L5),
}

# =============================================================================
# METRICS - L15
# =============================================================================
y_pred_mean_L15 = distance_mean_centriod_L15["closest_class"]
y_true_mean_L15 = distance_mean_centriod_L15["actual_class"]

y_pred_farthest_L15 = distance_farthest_centriod_L15["closest_class"]
y_true_farthest_L15 = distance_farthest_centriod_L15["actual_class"]

y_pred_random_L15 = distance_random_centriod_L15["closest_class"]
y_true_random_L15 = distance_random_centriod_L15["actual_class"]

# -----------------------------------------------------------------------------
L15_results = {
        "confusion_matrix_mean_L15": confusion_matrix(y_true_mean_L15, y_pred_mean_L15),
        "classification_report_mean_L15": classification_report(y_true_mean_L15, y_pred_mean_L15),
        "accuracy_mean_L15": accuracy_score(y_true_mean_L15, y_pred_mean_L15),
        "precision_mean_L15": precision_score(y_true_mean_L15, y_pred_mean_L15),
        "recall_mean_L15": recall_score(y_true_mean_L15, y_pred_mean_L15),
        "f1_mean_L15": f1_score(y_true_mean_L15, y_pred_mean_L15),
        
        "confusion_matrix_farthest_L15": confusion_matrix(y_true_farthest_L15, y_pred_farthest_L15),
        "classification_report_farthest_L15": classification_report(y_true_farthest_L15, y_pred_farthest_L15),
        "accuracy_farthest_L15": accuracy_score(y_true_farthest_L15, y_pred_farthest_L15),
        "precision_farthest_L15": precision_score(y_true_farthest_L15, y_pred_farthest_L15),
        "recall_farthest_L15": recall_score(y_true_farthest_L15, y_pred_farthest_L15),
        "f1_farthest_L15": f1_score(y_true_farthest_L15, y_pred_farthest_L15),
        
        "confusion_matrix_random_L15": confusion_matrix(y_true_random_L15, y_pred_random_L15),
        "classification_report_random_L15": classification_report(y_true_random_L15, y_pred_random_L15),
        "accuracy_random_L15": accuracy_score(y_true_random_L15, y_pred_random_L15),
        "precision_random_L15": precision_score(y_true_random_L15, y_pred_random_L15),
        "recall_random_L15": recall_score(y_true_random_L15, y_pred_random_L15),
        "f1_random_L15": f1_score(y_true_random_L15, y_pred_random_L15),
}

# =============================================================================
# STORE RESULTS - L5
# =============================================================================
save_path = "../../../6_Results/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L5"

store_results(path = save_path, 
              save_name = "L5_results", 
              result_object = L5_results)

store_results(path = save_path, 
              save_name = "distance_matrix_L5", 
              result_object = distance_matrix_L5)

store_results(path = save_path, 
              save_name = "distance_mean_centriod_L5", 
              result_object = distance_mean_centriod_L5)

store_results(path = save_path, 
              save_name = "distance_farthest_centriod_L5", 
              result_object = distance_farthest_centriod_L5)

store_results(path = save_path, 
              save_name = "distance_random_centriod_L5", 
              result_object = distance_random_centriod_L5)

# =============================================================================
# STORE RESULTS - L15
# =============================================================================
save_path = "../../../6_Results/10_Covariance_Matrix_And_Distances/Default_Of_Credit_Card_Client_Data/L15"

store_results(path = save_path, 
              save_name = "L15_results", 
              result_object = L15_results)

store_results(path = save_path, 
              save_name = "distance_matrix_L15", 
              result_object = distance_matrix_L15)

store_results(path = save_path, 
              save_name = "distance_mean_centriod_L15", 
              result_object = distance_mean_centriod_L15)

store_results(path = save_path, 
              save_name = "distance_farthest_centriod_L15", 
              result_object = distance_farthest_centriod_L15)

store_results(path = save_path, 
              save_name = "distance_random_centriod_L15", 
              result_object = distance_random_centriod_L15)