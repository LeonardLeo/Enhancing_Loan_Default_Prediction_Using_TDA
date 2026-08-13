# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 2026

Dataset: Taiwanese Bankruptcy Prediction

@author: lEO
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# =============================================================================
# Import Libraries
# =============================================================================
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from utils import store_results
import warnings

# =============================================================================
# Remove Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
_base = "../../../1_Data/Processed_Datasets/Taiwan_Bankruptcy/1_ML_Default_Parameters"
X_resampled = joblib.load(f"{_base}/X_resampled")
X_train = joblib.load(f"{_base}/X_train")
X_test = joblib.load(f"{_base}/X_test")
_y_resampled = joblib.load(f"{_base}/y_resampled")
_y_train = joblib.load(f"{_base}/y_train")
_y_test = joblib.load(f"{_base}/y_test")
y_resampled = _y_resampled.iloc[:, 0] if isinstance(_y_resampled, pd.DataFrame) else _y_resampled.squeeze()
y_train = _y_train.iloc[:, 0] if isinstance(_y_train, pd.DataFrame) else _y_train.squeeze()
y_test = _y_test.iloc[:, 0] if isinstance(_y_test, pd.DataFrame) else _y_test.squeeze()

# =============================================================================
# Value counts for the target
# =============================================================================
count_target_resampled_train = y_resampled.value_counts()

# =============================================================================
# Normalize the features
# =============================================================================
scaler_data = MinMaxScaler()
X_columns = X_resampled.columns
X_resampled = pd.DataFrame(scaler_data.fit_transform(X_resampled), columns=X_columns)
X_train = pd.DataFrame(scaler_data.transform(X_train), columns=X_columns)
X_test = pd.DataFrame(scaler_data.transform(X_test), columns=X_columns)

# =============================================================================
# Model Building - Logistic Regression, KNN, SVM, XGB, Random Forest
# =============================================================================
def train_dataset(X_resampled, y_resampled, X_test, y_test):
    param_grids = {
        "svm": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "degree": [2, 3, 4],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 10],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "leaf_size": [10, 20, 30, 50],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        },
        "xgb": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "logistic": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear", "lbfgs", "sag", "saga", "newton-cg"],
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "max_iter": [100, 200, 500],
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
        },
    }

    results = {}
    for model_name, param_grid in param_grids.items():
        if model_name == "svm":
            model = SVC()
        elif model_name == "knn":
            model = KNeighborsClassifier()
        elif model_name == "xgb":
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        elif model_name == "logistic":
            model = LogisticRegression()
        elif model_name == "random_forest":
            model = RandomForestClassifier()

        stratifiedkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=stratifiedkfold,
            scoring="f1",
            n_jobs=-1,
        )
        grid_search.fit(X_resampled, y_resampled)
        print(f"\n\nFinished grid search on {model_name}")

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        results[model_name] = {
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        print("\n\nSTEP Completed")
        print("Finished storing results.")

    return results

# =============================================================================
# MODEL EVALUATION RESULTS
# =============================================================================
training_results = train_dataset(X_resampled, y_resampled, X_test, y_test)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../6_Results/2_ML_Tuned_Parameters/Taiwan_Bankruptcy"
store_results(path=save_path, save_name="model_results", result_object=training_results)
