# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:38:02 2025

@author: leona
"""

# =============================================================================
# Import Libraries
# =============================================================================
import joblib
from utils import (improved_visualize_model_results, 
                   visualize_cross_validation_detailed)

# =============================================================================
# STATLOG GERMAN CREDIT DATASET - Results from Persistent Homology using Default Parameters
# =============================================================================
Model_Results = joblib.load("../../6_Results/4_PH_Tuned_Parameters/Statlog_German_Credit_Data/model_results.pkl")
CV_Results = joblib.load("../../6_Results/4_PH_Tuned_Parameters/Statlog_German_Credit_Data/CV_results.pkl")

# Visualize Model Results
viz_results = improved_visualize_model_results(
                model_results=Model_Results,
                save_dir="../../6_Results/4_PH_Tuned_Parameters/Statlog_German_Credit_Data/model_viz",
                compare_datasets=True,
                export_metrics=True,
                plot_precision_recall=False,
                colormap="viridis",
                hide_axis_labels=False
)

# Visualize CV Results.
cv_results = visualize_cross_validation_detailed(
                cross_val_results=CV_Results,
                save_dir="../../6_Results/4_PH_Tuned_Parameters/Statlog_German_Credit_Data/cv_viz",
                colormap="viridis",
                compare_models=True
)

# =============================================================================
# DEFAULT OF CREDIT CARD CLIENT DATASET - Results from Persistent Homology using Default Parameters
# =============================================================================
Model_Results_DCCCD = joblib.load("../../6_Results/4_PH_Tuned_Parameters/Default_Of_Credit_Card_Client_Data/model_results.pkl")
CV_Results_DCCCD = joblib.load("../../6_Results/4_PH_Tuned_Parameters/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

# Visualize Model Results
viz_results = improved_visualize_model_results(
                model_results=Model_Results_DCCCD,
                save_dir="../../6_Results/4_PH_Tuned_Parameters/Default_Of_Credit_Card_Client_Data/model_viz",
                compare_datasets=True,
                export_metrics=True,
                plot_precision_recall=False,
                colormap="viridis",
                hide_axis_labels=False
)

# Visualize CV Results.
cv_results = visualize_cross_validation_detailed(
                cross_val_results=CV_Results_DCCCD,
                save_dir="../../6_Results/4_PH_Tuned_Parameters/Default_Of_Credit_Card_Client_Data/cv_viz",
                colormap="viridis",
                compare_models=True
)