# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:59:58 2024

@author: lEO
"""

# Import Libraries
import warnings
from utils import (run_experiments_with_pca_components,
                   plot_all_metrics_vs_pca_components,
                   store_results)

# Handle Warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Run Multiple Component Analysis
# =============================================================================
results = run_experiments_with_pca_components(data_path = r"../../../../1_Data/Processed_Datasets/Statlog_German_Credit_Data/processed_data.xlsx",
                                              target_column = "Class",
                                              components_list = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                                              percentages = [30, 60],
                                              landmark_dir = "../../../../1_Data/Landmark_Sets/Statlog_German_Credit_Data/18_Variance_Retained_for_Statlog_German_Credit_Dataset",
                                              base_output_dir = "../../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/18_Variance_Retained_for_Statlog_German_Credit_Dataset",
                                              output_dir = "../../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/18_Variance_Retained_for_Statlog_German_Credit_Dataset",
                                              results_save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data",
                                              dataset_to_use = "dataset1",
                                              homology_dimension = 2,
                                              add_optional_path = True,
                                              test_size = 0.2,
                                              random_state = 42,
                                              experiment_name = "18_Variance_Retained_for_Statlog_German_Credit_Dataset")

# =============================================================================
# Store Results
# =============================================================================
save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data"

store_results(path = save_path, 
              save_name = "model_results", 
              result_object = results)

# =============================================================================
# Plot PCA Component Results
# =============================================================================
plot_results_knn = plot_all_metrics_vs_pca_components(all_results = results,
                                                      model_key = "knn",
                                                      save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                      separate_plots = True)

plot_results_knn_all = plot_all_metrics_vs_pca_components(all_results = results,
                                                          model_key = "knn",
                                                          save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                          separate_plots = False)

plot_results_svm = plot_all_metrics_vs_pca_components(all_results = results,
                                                      model_key = "svm",
                                                      save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                      separate_plots = True)

plot_results_svm_all = plot_all_metrics_vs_pca_components(all_results = results,
                                                          model_key = "svm",
                                                          save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                          separate_plots = False)

plot_results_xgb = plot_all_metrics_vs_pca_components(all_results = results,
                                                      model_key = "xgb",
                                                      save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                      separate_plots = True)

plot_results_xgb_all = plot_all_metrics_vs_pca_components(all_results = results,
                                                          model_key = "xgb",
                                                          save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                          separate_plots = False)

plot_results_logistic = plot_all_metrics_vs_pca_components(all_results = results,
                                                      model_key = "logistic",
                                                      save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                      separate_plots = True)

plot_results_logistic_all = plot_all_metrics_vs_pca_components(all_results = results,
                                                          model_key = "logistic",
                                                          save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                          separate_plots = False)

plot_results_random_forest = plot_all_metrics_vs_pca_components(all_results = results,
                                                      model_key = "random_forest",
                                                      save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                      separate_plots = True)

plot_results_random_forest_all = plot_all_metrics_vs_pca_components(all_results = results,
                                                          model_key = "random_forest",
                                                          save_path = "../../../../6_Results/Archives/18_Variance_Retained_for_Statlog_German_Credit_Dataset/Statlog_German_Credit_Data/viz",
                                                          separate_plots = False)