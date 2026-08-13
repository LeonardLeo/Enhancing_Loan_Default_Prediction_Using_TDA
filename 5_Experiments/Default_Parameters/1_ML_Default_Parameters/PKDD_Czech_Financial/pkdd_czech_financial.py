# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 2026

@author: lEO
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, SelectFpr
from imblearn.over_sampling import ADASYN
import warnings

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (
    eda,
    data_preprocessing_pipeline,
    save_python_object_using_joblib,
    store_results,
    train_dataset,
)

# =============================================================================
# Remove Warnings
# =============================================================================
warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
# =============================================================================
dataset = pd.read_csv(
    os.path.abspath("../../../../1_Data/Processed_Datasets/PKDD_Czech_Financial/processed_data.csv")
)

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
initial_eda = eda(dataset)
save_python_object_using_joblib(
    python_object=initial_eda,
    dataset_to_use="pkdd_czech",
    save_item="eda",
    save_name="initial_EDA",
    experiment_name="1_ML_Default_Parameters",
)

# =============================================================================
# Preprocessing data
# =============================================================================
# Missing values (numeric → median; categorical → explicit missing token)
for col in dataset.select_dtypes(include=[np.number]).columns:
    if dataset[col].isnull().any():
        dataset[col] = dataset[col].fillna(dataset[col].median())
for col in dataset.select_dtypes(include=["object"]).columns:
    dataset[col] = dataset[col].fillna("missing").astype(str)

dummy_col = [
    "frequency",
    "type",
    "sex",
    "A2",
    "A3",
    "A12",
    "A15",
    "preloan_card_type",
]
dataset = data_preprocessing_pipeline(
    dataset,
    log_col=["amount", "payments", "tx_amount_sum", "tx_amount_mean"],
    dummy_col=dummy_col,
)

# =============================================================================
# Save Clean Dataset
# =============================================================================
save_python_object_using_joblib(
    python_object=dataset,
    dataset_to_use="pkdd_czech",
    save_item="processed",
    save_name="processed_data",
    experiment_name="1_ML_Default_Parameters",
)

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
final_eda = eda(dataset)
save_python_object_using_joblib(
    python_object=final_eda,
    dataset_to_use="pkdd_czech",
    save_item="eda",
    save_name="final_EDA",
    experiment_name="1_ML_Default_Parameters",
)

# =============================================================================
# Select dependent and independent variables
# =============================================================================
X = dataset.drop("target", axis=1)
y = dataset["target"]

# =============================================================================
# Split dataset into training and test data
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# =============================================================================
# Value counts for the target
# =============================================================================
count_target_category_train = y_train.value_counts()
count_target_category_test = y_test.value_counts()

# =============================================================================
# Feature selection
# =============================================================================
selector = SelectFpr(score_func=f_classif)
X_train = pd.DataFrame(
    selector.fit_transform(X_train, y_train), columns=selector.get_feature_names_out()
)
X_test = pd.DataFrame(
    selector.transform(X_test), columns=selector.get_feature_names_out()
)
feature_info = pd.DataFrame(
    {
        "Features": selector.feature_names_in_,
        "Scores": np.around(selector.scores_, 2),
        "P-Value": np.around(selector.pvalues_, 2),
    }
)
save_python_object_using_joblib(
    python_object=feature_info,
    dataset_to_use="pkdd_czech",
    save_item="feature_info",
    save_name="feature_selection_info",
    experiment_name="1_ML_Default_Parameters",
)

# =============================================================================
# Resampling minority class
# =============================================================================
resampler_train = ADASYN(random_state=0)
X_resampled, y_resampled = resampler_train.fit_resample(X_train, y_train)

# =============================================================================
# Save Features and Label
# =============================================================================
save_python_object_using_joblib(
    python_object=X_resampled,
    dataset_to_use="pkdd_czech",
    save_item="processed",
    save_name="X_resampled",
    experiment_name="1_ML_Default_Parameters",
)
save_python_object_using_joblib(
    python_object=X_train,
    dataset_to_use="pkdd_czech",
    save_item="processed",
    save_name="X_train",
    experiment_name="1_ML_Default_Parameters",
)
save_python_object_using_joblib(
    python_object=X_test,
    dataset_to_use="pkdd_czech",
    save_item="processed",
    save_name="X_test",
    experiment_name="1_ML_Default_Parameters",
)
save_python_object_using_joblib(
    python_object=y_resampled,
    dataset_to_use="pkdd_czech",
    save_item="processed",
    save_name="y_resampled",
    experiment_name="1_ML_Default_Parameters",
)
save_python_object_using_joblib(
    python_object=y_train,
    dataset_to_use="pkdd_czech",
    save_item="processed",
    save_name="y_train",
    experiment_name="1_ML_Default_Parameters",
)
save_python_object_using_joblib(
    python_object=y_test,
    dataset_to_use="pkdd_czech",
    save_item="processed",
    save_name="y_test",
    experiment_name="1_ML_Default_Parameters",
)

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
# MODEL BUILDING
# =============================================================================
training_results = train_dataset(X_resampled, y_resampled, X_test, y_test)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
save_path = "../../../../6_Results/Default_Parameters/1_ML_Default_Parameters/PKDD_Czech_Financial"
store_results(path=save_path, save_name="model_results", result_object=training_results)
