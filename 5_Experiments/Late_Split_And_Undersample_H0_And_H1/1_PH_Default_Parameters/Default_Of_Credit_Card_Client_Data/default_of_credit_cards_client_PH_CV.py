# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:59:58 2024

@author: lEO
"""

import os
import sys
import joblib
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from utils import (store_results, 
                   perform_cross_validation_tda)

# =============================================================================
# Handle Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
data_dir = os.path.join(
    ROOT,
    "1_Data",
    "TDA_Datasets",
    "Late_Split_And_Undersample_H0_And_H1",
    "1_PH_Default_Parameters",
    "Default_Of_Credit_Card_Client_Data",
)
data_paths = [os.path.join(data_dir, "data_L5.csv"),
              os.path.join(data_dir, "data_L15.csv")]

# =============================================================================
# Load Python Object
# =============================================================================
try:
    path = os.path.join(
        ROOT,
        "6_Results",
        "Late_Split_And_Undersample_H0_And_H1",
        "1_PH_Default_Parameters",
        "Default_Of_Credit_Card_Client_Data",
    )
    evaluation_results = joblib.load(os.path.join(path, "model_results.pkl"))
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
save_path = os.path.join(
    ROOT,
    "6_Results",
    "Late_Split_And_Undersample_H0_And_H1",
    "1_PH_Default_Parameters",
    "Default_Of_Credit_Card_Client_Data",
)

store_results(path = save_path, 
              save_name = "CV_results", 
              result_object = model_results)
