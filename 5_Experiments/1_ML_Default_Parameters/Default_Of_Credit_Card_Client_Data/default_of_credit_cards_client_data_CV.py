# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:32:37 2024

@author: lEO
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import (cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import MinMaxScaler
from utils import store_results
import joblib
import warnings

# =============================================================================
# Remove Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
X_train = pd.read_excel("../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/X_train.xlsx").iloc[:, 1:]
y_train = pd.read_excel("../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/y_train.xlsx").iloc[:, 1]

# =============================================================================
# Normalize the features
# =============================================================================
scaler_data = MinMaxScaler()
X_columns = X_train.columns
X_train = pd.DataFrame(scaler_data.fit_transform(X_train), columns = X_columns)

# =============================================================================
# Get Python Object
# =============================================================================
training_results = joblib.load(os.path.abspath("../../../6_Results/1_ML_Default_Parameters/Default_Of_Credit_Card_Client_Data/model_results.pkl"))

# =============================================================================
# Getting Best Models
# =============================================================================
all_models = {
    "svm": training_results["svm"]["model"],
    "knn": training_results["knn"]["model"],
    "xgb": training_results["xgb"]["model"],
    "logistic": training_results["logistic"]["model"],
    "random_forest": training_results["random_forest"]["model"],
}

# =============================================================================
# Stratified K-Fold
# =============================================================================
# Define stratified k-fold     
stratifiedkfold = StratifiedKFold(n_splits = 10,
                                  shuffle = True,
                                  random_state = 42)

# =============================================================================
# Cross Validation
# =============================================================================
results = {}

for model_name, model in all_models.items():
    print(f"Performing cross-validation for {model_name}...")
    if model_name == "knn":
        # Ensure input data is in the correct format
        X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.to_numpy().ravel() if isinstance(y_train, pd.DataFrame) else y_train
    
    # Perform cross-validation and compute F1 scores
    cross_val = cross_val_score(estimator = model, 
                                X = X_train,
                                y = y_train,  # Ensure y_train is a 1D array
                                cv = stratifiedkfold,
                                n_jobs = -1)  # Use parallel processing for speed
    
    # Store results
    results[model_name] = {
        "cross_val_scores": cross_val,
        "mean_accracy": np.mean(cross_val),
        "std_accuracy": np.std(cross_val)
    }
    print(f"{model_name}: Mean Accuracy = {results[model_name]['mean_accracy']:.4f}, Std Accuracy = {results[model_name]['std_accuracy']:.4f}")

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../6_Results/1_ML_Default_Parameters/Default_Of_Credit_Card_Client_Data"

store_results(path = save_path, 
              save_name = "CV_results", 
              result_object = results)