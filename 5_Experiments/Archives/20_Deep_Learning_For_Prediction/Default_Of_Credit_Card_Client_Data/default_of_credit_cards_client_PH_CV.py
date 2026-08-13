# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:59:58 2024

@author: lEO
"""

import os
import joblib
import warnings
from utils import (store_results, 
                   perform_cross_validation_tda)

# =============================================================================
# Handle Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
data_paths = ["../../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/14_Mixed_Classes_Training_With_Imbalanced_Datasets/data_L5.csv",
              "../../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/14_Mixed_Classes_Training_With_Imbalanced_Datasets/data_L15.csv"]

# =============================================================================
# Load Python Object
# =============================================================================
try:
    path = "../../../../6_Results/Archives/14_Mixed_Classes_Training_With_Imbalanced_Datasets/Default_Of_Credit_Card_Client_Data"
    evaluation_results = joblib.load(f"{os.path.abspath(path)}/model_results.pkl")
except FileNotFoundError:
    print("Evaluation results file not found!")
    evaluation_results = {}

# =============================================================================
# Cross Validation
# =============================================================================
model_results = perform_cross_validation_tda(datasets = data_paths, 
                                             model_results = evaluation_results)

# =============================================================================
# Store Python Object
# =============================================================================
save_path = "../../../../6_Results/Archives/14_Mixed_Classes_Training_With_Imbalanced_Datasets/Default_Of_Credit_Card_Client_Data"

store_results(path = save_path, 
              save_name = "CV_results", 
              result_object = model_results)
