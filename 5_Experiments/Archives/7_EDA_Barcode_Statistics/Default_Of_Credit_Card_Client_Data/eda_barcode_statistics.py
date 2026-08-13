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
# EDA for Each Class in Each Frame
# =============================================================================
data = ["../../../../1_Data/Barcode_Statistics/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/barcode_stats_default_L5.csv",
        "../../../../1_Data/Barcode_Statistics/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/barcode_stats_default_L15.csv",
        "../../../../1_Data/Barcode_Statistics/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/barcode_stats_non-default_L5.csv",
        "../../../../1_Data/Barcode_Statistics/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/barcode_stats_non-default_L15.csv"]

results_3 = {}
data_3 = {}
for each_data in data:
    data_name = os.path.basename(each_data)
    dataset = pd.read_csv(each_data)
    data_3[data_name] = dataset
    results_3[data_name] = eda(dataset = dataset,
                               graphs = True,
                               hist_figsize = (30, 20))

# =============================================================================
# Store Python Object
# =============================================================================
save_path = "../../../../6_Results/Archives/7_EDA_Barcode_Statistics/Default_Of_Credit_Card_Client_Data"

store_results(path = save_path, 
              save_name = "eda_each_class_BS_DCCD", 
              result_object = results_3)

#%%
# =============================================================================
# EDA for Each Class in DataFrame
# =============================================================================
data = ["../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/data_L5.csv",
        "../../../../1_Data/TDA_Datasets/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/Default_Of_Credit_Card_Client_Data/data_L15.csv"]

results_4 = {}
data_4 = {}
for each_data in data:
    data_name = os.path.basename(each_data)
    dataset = pd.read_csv(each_data)
    data_4[data_name] = dataset
    results_4[data_name] = eda(dataset = dataset,
                               graphs = True,
                               hist_figsize = (30, 20))

# =============================================================================
# Store Python Object
# =============================================================================
save_path = "../../../../6_Results/Archives/7_EDA_Barcode_Statistics/Default_Of_Credit_Card_Client_Data"

store_results(path = save_path, 
              save_name = "eda_entire_BS_DCCD", 
              result_object = results_4)

#%%
# =============================================================================
# EDA for Each Class in Each Frame
# =============================================================================
data = ["../../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_default_L5.csv",
        "../../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_default_L15.csv",
        "../../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_non-default_L5.csv",
        "../../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_non-default_L15.csv"]

results_H0_3 = {}
data_H0_3 = {}
for each_data in data:
    data_name = os.path.basename(each_data)
    dataset = pd.read_csv(each_data)
    data_H0_3[data_name] = dataset
    results_H0_3[data_name] = eda(dataset = dataset,
                               graphs = True,
                               hist_figsize = (30, 20))

# =============================================================================
# Store Python Object
# =============================================================================
save_path = "../../../../6_Results/Archives/7_EDA_Barcode_Statistics/Default_Of_Credit_Card_Client_Data"

store_results(path = save_path, 
              save_name = "eda_each_class_BS_DCCD_H0_Only", 
              result_object = results_H0_3)

#%%
# =============================================================================
# EDA for Each Class in DataFrame
# =============================================================================
data = ["../../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/data_L5.csv",
        "../../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/data_L15.csv"]

results_H0_4 = {}
data_H0_4 = {}
for each_data in data:
    data_name = os.path.basename(each_data)
    dataset = pd.read_csv(each_data)
    data_H0_4[data_name] = dataset
    results_H0_4[data_name] = eda(dataset = dataset,
                               graphs = True,
                               hist_figsize = (30, 20))

# =============================================================================
# Store Python Object
# =============================================================================
save_path = "../../../../6_Results/Archives/7_EDA_Barcode_Statistics/Default_Of_Credit_Card_Client_Data"

store_results(path = save_path, 
              save_name = "eda_entire_BS_DCCD_H0_Only", 
              result_object = results_H0_4)