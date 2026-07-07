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
# DEFAULT OF CREDIT CARD CLIENT DATASET - Results from Persistent Homology using Default Parameters
# =============================================================================
Model_Results_DCCCD = joblib.load("../../6_Results/12_Equivalent_Sample_Size_For_Each_Dataset/Default_Of_Credit_Card_Client_Data/model_results.pkl")
CV_Results_DCCCD = joblib.load("../../6_Results/12_Equivalent_Sample_Size_For_Each_Dataset/Default_Of_Credit_Card_Client_Data/CV_results.pkl")

# Visualize Model Results
viz_results = improved_visualize_model_results(
                model_results=Model_Results_DCCCD,
                save_dir="../../6_Results/12_Equivalent_Sample_Size_For_Each_Dataset/Default_Of_Credit_Card_Client_Data/model_viz",
                compare_datasets=True,
                export_metrics=True,
                plot_precision_recall=False,
                colormap="viridis",
                hide_axis_labels=False
)

# Visualize CV Results.
cv_results = visualize_cross_validation_detailed(
                cross_val_results=CV_Results_DCCCD,
                save_dir="../../6_Results/12_Equivalent_Sample_Size_For_Each_Dataset/Default_Of_Credit_Card_Client_Data/cv_viz",
                colormap="viridis",
                compare_models=True
)