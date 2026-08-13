# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:32:37 2024

@author: lEO
"""

# =============================================================================
# Import Libraries
# =============================================================================
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (StratifiedKFold,
                                     GridSearchCV)
from utils import store_results
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             classification_report,
                             confusion_matrix)
import warnings

# =============================================================================
# Initialize a Random State 
# =============================================================================
state = 0

# =============================================================================
# Remove Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
X_resampled = pd.read_excel("../../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/X_resampled.xlsx").iloc[:, 1:]
X_train = pd.read_excel("../../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/X_train.xlsx").iloc[:, 1:]
X_test = pd.read_excel("../../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/X_test.xlsx").iloc[:, 1:]
y_resampled = pd.read_excel("../../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/y_resampled.xlsx").iloc[:, 1]
y_train = pd.read_excel("../../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/y_train.xlsx").iloc[:, 1]
y_test = pd.read_excel("../../../../1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/y_test.xlsx").iloc[:, 1]

# =============================================================================
# Value counts for the target
# =============================================================================
count_target_resampled_train = y_resampled.value_counts()

# =============================================================================
# Normalize the features
# =============================================================================
scaler_data = MinMaxScaler()
X_columns = X_resampled.columns
X_resampled = pd.DataFrame(scaler_data.fit_transform(X_resampled), columns = X_columns)
X_train = pd.DataFrame(scaler_data.transform(X_train), columns = X_columns)
X_test = pd.DataFrame(scaler_data.transform(X_test), columns = X_columns)

# =============================================================================
# Model Building - Logistic Regression, Gaussian Naive Bayes, KNN, SVM
# =============================================================================
def train_dataset(X_resampled,
                  y_resampled,
                  X_test,
                  y_test):
    
    # Step 2: Define hyperparameter grids for each model
    param_grids = {
        "svm": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        },
        "knn": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
            "p": [1, 2]  # Manhattan (L1) or Euclidean (L2)
        },
        "xgb": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "logistic": {
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10]
        },
    }
    
    # Step 3: Train models and perform hyperparameter tuning
    results = {}
    
    for model_name, param_grid in param_grids.items():
        # Select model
        if model_name == "svm":
            model = SVC()
        elif model_name == "knn":
            model = KNeighborsClassifier()
        elif model_name == "xgb":
            model = XGBClassifier(use_label_encoder = False, 
                                  eval_metric = "logloss")
        elif model_name == "logistic":
            model = LogisticRegression()
        elif model_name == "random_forest":
            model = RandomForestClassifier()
            
        # Define stratified k-fold     
        stratifiedkfold = StratifiedKFold(n_splits = 5,
                                          shuffle = True,
                                          random_state = 42)
        
        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv = stratifiedkfold,
            scoring = "f1",
            n_jobs = -1
        )
        grid_search.fit(X_resampled, y_resampled)
        print(f"\n\nFinished grid search on {model_name}")
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        
        # Store results (separate training and test results)
        # Store results (separate cross-validation and test results)
        results[model_name] = {
                "best_model": grid_search.best_estimator_,
                "best_params": grid_search.best_params_,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred)
            }
        print("\n\nSTEP Completed")
        print("Finished storing results.")
    
    return results

# =============================================================================
# MODEL EVALUATION RESULTS
# =============================================================================
training_results = train_dataset(X_resampled,
                                 y_resampled,
                                 X_test,
                                 y_test)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../../6_Results/Default_Parameters/2_ML_Tuned_Parameters/Default_Of_Credit_Card_Client_Data"

store_results(path = save_path, 
              save_name = "model_results", 
              result_object = training_results)

# # =============================================================================
# # Selecting the best features
# # =============================================================================
# get_best_features = pd.DataFrame({"Features": X_resampled.columns,
#                                   "Scores": model.feature_importances_})

# # Save the DataFrame (Feature Info) as an Excel file
# get_best_features.to_excel('../Python_Objects/statlog+german+credit+data/Feature_Info/best_features_from_model.xlsx', index = True)
