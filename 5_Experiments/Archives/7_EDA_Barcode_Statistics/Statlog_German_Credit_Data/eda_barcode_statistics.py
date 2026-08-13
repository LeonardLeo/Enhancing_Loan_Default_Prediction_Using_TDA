# -*- coding: utf-8 -*-
"""
Created on Thu May  1 14:50:30 2025

@author: leona
"""

import os
import pandas as pd
from utils import eda, store_results
import warnings

warnings.filterwarnings("ignore")

#%%
# =============================================================================
# STATLOG GERMAN CREDIT DATASET
# =============================================================================

# =============================================================================
# EDA for Each Class in Each Frame
# =============================================================================
data = ["../../../../1_Data/Barcode_Statistics/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/barcode_stats_default_L30.csv",
        "../../../../1_Data/Barcode_Statistics/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/barcode_stats_default_L60.csv",
        "../../../../1_Data/Barcode_Statistics/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/barcode_stats_non-default_L30.csv",
        "../../../../1_Data/Barcode_Statistics/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/barcode_stats_non-default_L60.csv"]

results_1 = {}
data_1 = {}
for each_data in data:
    data_name = os.path.basename(each_data)
    dataset = pd.read_csv(each_data)
    data_1[data_name] = dataset
    results_1[data_name] = eda(dataset = dataset,
                               graphs = True,
                               hist_figsize = (30, 20))

# =============================================================================
# Store Python Object
# =============================================================================
save_path = "../../../../6_Results/Archives/7_EDA_Barcode_Statistics/Statlog_German_Credit_Data"

store_results(path = save_path, 
              save_name = "eda_each_class_BS", 
              result_object = results_1)

#%%
# =============================================================================
# EDA for Each Class in DataFrame
# =============================================================================
data = ["../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/data_L30.csv",
        "../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Statlog_German_Credit_Data/data_L60.csv"]

results_2 = {}
data_2 = {}
for each_data in data:
    data_name = os.path.basename(each_data)
    dataset = pd.read_csv(each_data)
    data_2[data_name] = dataset
    results_2[data_name] = eda(dataset = dataset,
                               graphs = True,
                               hist_figsize = (30, 20))

# =============================================================================
# Store Python Object
# =============================================================================
save_path = "../../../../6_Results/Archives/7_EDA_Barcode_Statistics/Statlog_German_Credit_Data"

store_results(path = save_path, 
              save_name = "eda_entire_BS", 
              result_object = results_2)

#%%
# =============================================================================
# EDA for Each Class in Each Frame - H0
# =============================================================================
data = ["../../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_default_L30.csv",
        "../../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_default_L60.csv",
        "../../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_non-default_L30.csv",
        "../../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_non-default_L60.csv"]

results_H0_1 = {}
data_H0_1 = {}
for each_data in data:
    data_name = os.path.basename(each_data)
    dataset = pd.read_csv(each_data)
    data_H0_1[data_name] = dataset
    results_H0_1[data_name] = eda(dataset = dataset,
                               graphs = True,
                               hist_figsize = (30, 20))

# =============================================================================
# Store Python Object - H0
# =============================================================================
save_path = "../../../../6_Results/Archives/7_EDA_Barcode_Statistics/Statlog_German_Credit_Data"

store_results(path = save_path, 
              save_name = "eda_each_class_BS_H0", 
              result_object = results_H0_1)

#%%
# =============================================================================
# EDA for Each Class in DataFrame - H0
# =============================================================================
data = ["../../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/data_L30.csv",
        "../../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/data_L60.csv"]

results_H0_2 = {}
data_H0_2 = {}
for each_data in data:
    data_name = os.path.basename(each_data)
    dataset = pd.read_csv(each_data)
    data_H0_2[data_name] = dataset
    results_H0_2[data_name] = eda(dataset = dataset,
                               graphs = True,
                               hist_figsize = (30, 20))

# =============================================================================
# Store Python Object - H0
# =============================================================================
save_path = "../../../../6_Results/Archives/7_EDA_Barcode_Statistics/Statlog_German_Credit_Data"

store_results(path = save_path, 
              save_name = "eda_entire_BS_H0", 
              result_object = results_H0_2)
