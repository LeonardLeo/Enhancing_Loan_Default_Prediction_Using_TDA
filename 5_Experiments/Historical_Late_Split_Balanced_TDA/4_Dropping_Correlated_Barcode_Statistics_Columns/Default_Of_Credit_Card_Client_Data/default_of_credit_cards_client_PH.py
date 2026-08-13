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
TDA_DIR = tda_artefact_dir("TDA_Datasets", "Historical_Late_Split_Balanced_TDA", "1_PH_Default_Parameters", "Default_Of_Credit_Card_Client_Data")
data_L5 = pd.read_csv(TDA_DIR / "data_L5.csv")
data_L15 = pd.read_csv(TDA_DIR / "data_L15.csv")

data_L5 = rename_barcode_statistics_columns(data_L5)
data_L15 = rename_barcode_statistics_columns(data_L15)

# =============================================================================
# SHUFFLE, SPLIT DATASET AND SCALE
# =============================================================================
# Shuffle
data_L5 = data_L5.sample(frac=1, random_state=42).reset_index(drop=True)
data_L15 = data_L15.sample(frac=1, random_state=42).reset_index(drop=True)

# Split
X_data_L5 = data_L5.drop(columns="label")
X_data_L15 = data_L15.drop(columns="label")
y_data_L5 = data_L5["label"]
y_data_L15 = data_L15["label"]

X_train_data_L5, X_test_data_L5, y_train_data_L5, y_test_data_L5 = train_test_split(X_data_L5, 
                                                                                    y_data_L5, 
                                                                                    test_size=0.2, 
                                                                                    random_state=42, 
                                                                                    stratify=y_data_L5)

X_train_data_L15, X_test_data_L15, y_train_data_L15, y_test_data_L15 = train_test_split(X_data_L15, 
                                                                                        y_data_L15, 
                                                                                        test_size=0.2, 
                                                                                        random_state=42, 
                                                                                        stratify=y_data_L15)

# Normalize
scaler_data_L5 = MinMaxScaler()
scaler_data_L15 = MinMaxScaler()

X_train_data_L5 = pd.DataFrame(scaler_data_L5.fit_transform(X_train_data_L5), columns = scaler_data_L5.feature_names_in_)
X_train_data_L15 = pd.DataFrame(scaler_data_L15.fit_transform(X_train_data_L15), columns = scaler_data_L15.feature_names_in_)
X_test_data_L5 = pd.DataFrame(scaler_data_L5.transform(X_test_data_L5), columns = scaler_data_L5.feature_names_in_)
X_test_data_L15 = pd.DataFrame(scaler_data_L15.transform(X_test_data_L15), columns = scaler_data_L15.feature_names_in_)

# =============================================================================
# EDA
# =============================================================================
eda_data_L5 = eda(data_L5)
eda_data_L15 = eda(data_L15)

eda_X_train_L5 = eda(X_train_data_L5)
eda_X_train_L15 = eda(X_train_data_L15)

eda_X_test_L5 = eda(X_test_data_L5)
eda_X_test_L15 = eda(X_test_data_L15)

# =============================================================================
# DROP CORRELATED VARIABLES
# =============================================================================
# Using 'high_variance' and 'target_corr' strategy for data_L5
data_L5_var, dropped_data_L5 = drop_correlated_features(X_train_data_L5,
                                                        drop_columns=[0, 4, 8],
                                                        threshold=0.80, 
                                                        feature_label = True,
                                                        target=y_train_data_L5,
                                                        strategy='high_variance')

data_L5_target, data_L5_dropped = drop_correlated_features(X_train_data_L5, 
                                                        drop_columns=[0, 4, 8],
                                                        threshold=0.80, 
                                                        feature_label = True,
                                                        strategy='target_corr', 
                                                        target=y_train_data_L5)

# Using 'high_variance' and 'target_corr' strategy for data_L15
data_L15_var, dropped_data_L15 = drop_correlated_features(X_train_data_L15,
                                                        drop_columns=[0, 4, 8],
                                                        threshold=0.80, 
                                                        feature_label = True,
                                                        target=y_train_data_L15,
                                                        strategy='high_variance')

data_L15_target, data_L15_dropped = drop_correlated_features(X_train_data_L15, 
                                                        drop_columns=[0, 4, 8],
                                                        threshold=0.80, 
                                                        feature_label = True,
                                                        strategy='target_corr', 
                                                        target=y_train_data_L15)

# Data Columns
data_L5_var_columns = data_L5_var.drop("label", axis = 1).columns
data_L5_target_columns = data_L5_target.drop("label", axis = 1).columns
data_L15_var_columns = data_L15_var.drop("label", axis = 1).columns
data_L15_target_columns = data_L15_target.drop("label", axis = 1).columns

# Test Data
data_L5_var_test = X_test_data_L5[data_L5_var_columns]
data_L5_target_test = X_test_data_L5[data_L5_target_columns]
data_L15_var_test = X_test_data_L15[data_L15_var_columns]
data_L15_target_test = X_test_data_L15[data_L15_target_columns]

# =============================================================================
# STORE DATAFRAMES RESULTS
# =============================================================================
save_path_var = str(tda_artefact_dir("TDA_Datasets", "Historical_Late_Split_Balanced_TDA", "4_Dropping_Correlated_Barcode_Statistics_Columns", "Default_Of_Credit_Card_Client_Data", "Using_High_Variance_For_Correlation"))
save_path_target = str(tda_artefact_dir("TDA_Datasets", "Historical_Late_Split_Balanced_TDA", "4_Dropping_Correlated_Barcode_Statistics_Columns", "Default_Of_Credit_Card_Client_Data", "Using_Target_Variable_For_Correlation"))
save_path_results = str(tda_results_dir("Historical_Late_Split_Balanced_TDA", "4_Dropping_Correlated_Barcode_Statistics_Columns", "Default_Of_Credit_Card_Client_Data"))

store_data_as_csv_or_json(path = save_path_var, 
                          csv = True,
                          save_as = ["data_L5_var", 
                                     "data_L15_var"],
                          data_object = [data_L5_var, 
                                         data_L15_var])

store_data_as_csv_or_json(path = save_path_target, 
                          csv = True,
                          save_as = ["data_L5_target",  
                                     "data_L15_target"],
                          data_object = [data_L5_target, 
                                         data_L15_target])

store_data_as_csv_or_json(path = save_path_results, 
                          csv = False,
                          save_as = ["data_L5_var_drop", 
                                     "data_L5_target_drop", 
                                     "data_L15_var_drop", 
                                     "data_L15_target_drop"],
                          data_object = [dropped_data_L5, 
                                         data_L5_dropped, 
                                         dropped_data_L15, 
                                         data_L15_dropped])

# =============================================================================
# VISUALIZE DROPPED CORRELATED COLUMNS
# =============================================================================
visualize_correlation_drop_maps(
    drop_maps=[dropped_data_L5, dropped_data_L15, data_L5_dropped, data_L15_dropped],
    corr_matrices=[
        X_train_data_L5.corr().abs(),
        X_train_data_L15.corr().abs(),
        X_train_data_L5.corr().abs(),
        X_train_data_L15.corr().abs()
    ],
    dataset_labels=[
        "Data L5 (High Variance)",
        "Data L15 (High Variance)",
        "Data L5 (Target Corr)",
        "Data L15 (Target Corr)"
    ],
    save_path=f"{save_path_results}/correlation_graphs",
    save_png=True,
    save_html=True,
    use_pyvis=True
)

# =============================================================================
# TRAIN MACHINE LEARNING MODEL WITH DEFAULT PARAMETERS
# =============================================================================
data_to_use = {"data_L5": {"data": data_L5_target, # You can switch this to data_L5_var for High Variance strategy
                           "X_test": data_L5_target_test,
                           "y_test": y_test_data_L5},
               
               "data_L15": {"data": data_L15_target, # You can switch this to data_L5_var for High Variance strategy
                            "X_test": data_L15_target_test,
                            "y_test": y_test_data_L15}}

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
