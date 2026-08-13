# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:59:58 2024

@author: lEO
"""

import sys
from pathlib import Path

import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (eda,
                   drop_correlated_features,
                   visualize_correlation_drop_maps,
                   train_multiple_dataset_tda_drop_correlated,
                   store_results,
                   store_data_as_csv_or_json,
                   rename_barcode_statistics_columns,
                   tda_artefact_dir,
                   tda_results_dir)

# Handle warnings
warnings.filterwarnings("ignore")

# =============================================================================
# GET DATASET
# =============================================================================
TDA_DIR = tda_artefact_dir("TDA_Datasets", "Historical_Late_Split_Balanced_TDA", "1_PH_Default_Parameters", "Statlog_German_Credit_Data")
data_L30 = pd.read_csv(TDA_DIR / "data_L30.csv")
data_L60 = pd.read_csv(TDA_DIR / "data_L60.csv")

data_L30 = rename_barcode_statistics_columns(data_L30)
data_L60 = rename_barcode_statistics_columns(data_L60)

# =============================================================================
# SHUFFLE, SPLIT DATASET AND SCALE
# =============================================================================
# Shuffle
data_L30 = data_L30.sample(frac=1, random_state=42).reset_index(drop=True)
data_L60 = data_L60.sample(frac=1, random_state=42).reset_index(drop=True)

# Split
X_data_L30 = data_L30.drop(columns="label")
X_data_L60 = data_L60.drop(columns="label")
y_data_L30 = data_L30["label"]
y_data_L60 = data_L60["label"]

X_train_data_L30, X_test_data_L30, y_train_data_L30, y_test_data_L30 = train_test_split(X_data_L30, 
                                                                                        y_data_L30, 
                                                                                        test_size=0.2, 
                                                                                        random_state=42, 
                                                                                        stratify=y_data_L30)

X_train_data_L60, X_test_data_L60, y_train_data_L60, y_test_data_L60 = train_test_split(X_data_L60, 
                                                                                        y_data_L60, 
                                                                                        test_size=0.2, 
                                                                                        random_state=42, 
                                                                                        stratify=y_data_L60)

# Normalize
scaler_data_L30 = MinMaxScaler()
scaler_data_L60 = MinMaxScaler()

X_train_data_L30 = pd.DataFrame(scaler_data_L30.fit_transform(X_train_data_L30), columns = scaler_data_L30.feature_names_in_)
X_train_data_L60 = pd.DataFrame(scaler_data_L60.fit_transform(X_train_data_L60), columns = scaler_data_L60.feature_names_in_)
X_test_data_L30 = pd.DataFrame(scaler_data_L30.transform(X_test_data_L30), columns = scaler_data_L30.feature_names_in_)
X_test_data_L60 = pd.DataFrame(scaler_data_L60.transform(X_test_data_L60), columns = scaler_data_L60.feature_names_in_)

# =============================================================================
# EDA
# =============================================================================
eda_data_L30 = eda(data_L30)
eda_data_L60 = eda(data_L60)

eda_X_train_L30 = eda(X_train_data_L30)
eda_X_train_L60 = eda(X_train_data_L60)

eda_X_test_L30 = eda(X_test_data_L30)
eda_X_test_L60 = eda(X_test_data_L60)

# =============================================================================
# DROP CORRELATED VARIABLES
# =============================================================================
# Using 'high_variance' and 'target_corr' strategy for data_L30
data_L30_var, dropped_data_L30 = drop_correlated_features(X_train_data_L30,
                                                        drop_columns=[0, 4, 8],
                                                        threshold=0.80, 
                                                        feature_label = True,
                                                        target=y_train_data_L30,
                                                        strategy='high_variance')

data_L30_target, data_L30_dropped = drop_correlated_features(X_train_data_L30, 
                                                        drop_columns=[0, 4, 8],
                                                        threshold=0.80, 
                                                        feature_label = True,
                                                        strategy='target_corr', 
                                                        target=y_train_data_L30)

# Using 'high_variance' and 'target_corr' strategy for data_L60
data_L60_var, dropped_data_L60 = drop_correlated_features(X_train_data_L60,
                                                        drop_columns=[0, 4, 8],
                                                        threshold=0.80, 
                                                        feature_label = True,
                                                        target=y_train_data_L60,
                                                        strategy='high_variance')

data_L60_target, data_L60_dropped = drop_correlated_features(X_train_data_L60, 
                                                        drop_columns=[0, 4, 8],
                                                        threshold=0.80, 
                                                        feature_label = True,
                                                        strategy='target_corr', 
                                                        target=y_train_data_L60)

# Data Columns
data_L30_var_columns = data_L30_var.drop("label", axis = 1).columns
data_L30_target_columns = data_L30_target.drop("label", axis = 1).columns
data_L60_var_columns = data_L60_var.drop("label", axis = 1).columns
data_L60_target_columns = data_L60_target.drop("label", axis = 1).columns

# Test Data
data_L30_var_test = X_test_data_L30[data_L30_var_columns]
data_L30_target_test = X_test_data_L30[data_L30_target_columns]
data_L60_var_test = X_test_data_L60[data_L60_var_columns]
data_L60_target_test = X_test_data_L60[data_L60_target_columns]

# =============================================================================
# STORE DATAFRAMES RESULTS
# =============================================================================
save_path_var = str(tda_artefact_dir("TDA_Datasets", "Historical_Late_Split_Balanced_TDA", "4_Dropping_Correlated_Barcode_Statistics_Columns", "Statlog_German_Credit_Data", "Using_High_Variance_For_Correlation"))
save_path_target = str(tda_artefact_dir("TDA_Datasets", "Historical_Late_Split_Balanced_TDA", "4_Dropping_Correlated_Barcode_Statistics_Columns", "Statlog_German_Credit_Data", "Using_Target_Variable_For_Correlation"))
save_path_results = str(tda_results_dir("Historical_Late_Split_Balanced_TDA", "4_Dropping_Correlated_Barcode_Statistics_Columns", "Statlog_German_Credit_Data"))

store_data_as_csv_or_json(path = save_path_var, 
                          csv = True,
                          save_as = ["data_L30_var", 
                                     "data_L60_var"],
                          data_object = [data_L30_var, 
                                         data_L60_var])

store_data_as_csv_or_json(path = save_path_target, 
                          csv = True,
                          save_as = ["data_L30_target",  
                                     "data_L60_target"],
                          data_object = [data_L30_target, 
                                         data_L60_target])

store_data_as_csv_or_json(path = save_path_results, 
                          csv = False,
                          save_as = ["data_L30_var_drop", 
                                     "data_L30_target_drop", 
                                     "data_L60_var_drop", 
                                     "data_L60_target_drop"],
                          data_object = [dropped_data_L30, 
                                         data_L30_dropped, 
                                         dropped_data_L60, 
                                         data_L60_dropped])

# =============================================================================
# VISUALIZE DROPPED CORRELATED COLUMNS
# =============================================================================
visualize_correlation_drop_maps(
    drop_maps=[dropped_data_L30, dropped_data_L60, data_L30_dropped, data_L60_dropped],
    corr_matrices=[
        X_train_data_L30.corr().abs(),
        X_train_data_L60.corr().abs(),
        X_train_data_L30.corr().abs(),
        X_train_data_L60.corr().abs()
    ],
    dataset_labels=[
        "Data L30 (High Variance)",
        "Data L60 (High Variance)",
        "Data L30 (Target Corr)",
        "Data L60 (Target Corr)"
    ],
    save_path=f"{save_path_results}/correlation_graphs",
    save_png=True,
    save_html=True,
    use_pyvis=True
)

# =============================================================================
# TRAIN MACHINE LEARNING MODEL WITH DEFAULT PARAMETERS
# =============================================================================
data_to_use = {"data_L30": {"data": data_L30_target, # You can switch this to data_L30_var for High Variance strategy
                           "X_test": data_L30_target_test,
                           "y_test": y_test_data_L30},
               
               "data_L60": {"data": data_L60_target, # You can switch this to data_L30_var for High Variance strategy
                            "X_test": data_L60_target_test,
                            "y_test": y_test_data_L60}}

model_results = train_multiple_dataset_tda_drop_correlated(data_objects = data_to_use,
                                                           test_size = 0.2,
                                                           random_state = 42,
                                                           xgb = {"eval_metric":"logloss"})

print(model_results)

# =============================================================================
# STORE MODEL RESULTS
# =============================================================================
store_results(path = save_path_results, 
              save_name = "model_results", 
              result_object = model_results)
