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

warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
data_paths = ["../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/19_Linear_Regression_For_Prediction/data_L30.csv", 
              "../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/19_Linear_Regression_For_Prediction/data_L60.csv"]

# =============================================================================
# Load Python Object
# =============================================================================
try:
    path = "../../../6_Results/19_Linear_Regression_For_Prediction/Statlog_German_Credit_Data"
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
save_path = "../../../6_Results/19_Linear_Regression_For_Prediction/Statlog_German_Credit_Data"

store_results(path = save_path, 
              save_name = "CV_results", 
              result_object = model_results)
