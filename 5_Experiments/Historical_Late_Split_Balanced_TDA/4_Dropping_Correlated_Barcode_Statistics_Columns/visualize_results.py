# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:38:02 2025

@author: leona
"""

# =============================================================================
# Import Libraries
# =============================================================================
import joblib
from utils import improved_visualize_model_results

# =============================================================================
# STATLOG GERMAN CREDIT DATASET - Results from Persistent Homology using Default Parameters
# =============================================================================
Model_Results = joblib.load("../../../6_Results/Historical_Late_Split_Balanced_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/Statlog_German_Credit_Data/model_results.pkl")

# Visualize Model Results
viz_results = improved_visualize_model_results(
                model_results=Model_Results,
                save_dir="../../../6_Results/Historical_Late_Split_Balanced_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/Statlog_German_Credit_Data/model_viz",
                compare_datasets=True,
                export_metrics=True,
                plot_precision_recall=False,
                colormap="viridis",
                hide_axis_labels=True
)

# =============================================================================
# DEFAULT OF CREDIT CARD CLIENT DATASET - Results from Persistent Homology using Default Parameters
# =============================================================================
Model_Results_DCCCD = joblib.load("../../../6_Results/Historical_Late_Split_Balanced_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/Default_Of_Credit_Card_Client_Data/model_results.pkl")

# Visualize Model Results
viz_results = improved_visualize_model_results(
                model_results=Model_Results_DCCCD,
                save_dir="../../../6_Results/Historical_Late_Split_Balanced_TDA/4_Dropping_Correlated_Barcode_Statistics_Columns/Default_Of_Credit_Card_Client_Data/model_viz",
                compare_datasets=True,
                export_metrics=True,
                plot_precision_recall=False,
                colormap="viridis",
                hide_axis_labels=True
)
