# -*- coding: utf-8 -*-
"""
Created on Wed May  7 23:10:12 2025

@author: leona
"""

# %%
# Import Libraries

import os
import pandas as pd
from utils import perform_pca_analysis

# %%
# =============================================================================
# GET DATASET
# =============================================================================

# =============================================================================
# Class Barcode Statistics - Statlog (3_PH_Default_Parameters)
# =============================================================================
dcccd_barcode_stats_default_L5 = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/barcode_stats_default_L5.csv"))
dcccd_barcode_stats_default_L15 = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/barcode_stats_default_L15.csv"))
dcccd_barcode_stats_non_default_L5 = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/barcode_stats_non-default_L5.csv"))
dcccd_barcode_stats_non_default_L15 = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/barcode_stats_non-default_L15.csv"))

#%%
# =============================================================================
# Entire Barcode Statistics - Statlog (3_PH_Default_Parameters)
# =============================================================================
dcccd_data_L5 = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L5.csv"))
dcccd_data_L15 = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L15.csv"))





#%%
# =============================================================================
# Barcode Statistics - Statlog (4_PH_Tuned_Parameters)
# =============================================================================
dcccd_barcode_stats_default_L5_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/4_PH_Tuned_Parameters/barcode_stats_default_L5.csv"))
dcccd_barcode_stats_default_L15_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/4_PH_Tuned_Parameters/barcode_stats_default_L15.csv"))
dcccd_barcode_stats_non_default_L5_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/4_PH_Tuned_Parameters/barcode_stats_non-default_L5.csv"))
dcccd_barcode_stats_non_default_L15_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/4_PH_Tuned_Parameters/barcode_stats_non-default_L15.csv"))

#%%
# =============================================================================
# Entire Barcode Statistics - Statlog (4_PH_Tuned_Parameters)
# =============================================================================
dcccd_data_L5_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/4_PH_Tuned_Parameters/data_L5.csv"))
dcccd_data_L15_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/4_PH_Tuned_Parameters/data_L15.csv"))





#%%
# =============================================================================
# Barcode Statistics - Statlog (6_Experiment_Impact_of_H0_Only)
# =============================================================================
dcccd_barcode_stats_default_L5_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_default_L5.csv"))
dcccd_barcode_stats_default_L15_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_default_L15.csv"))
dcccd_barcode_stats_non_default_L5_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_non-default_L5.csv"))
dcccd_barcode_stats_non_default_L15_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_non-default_L15.csv"))

#%%
# =============================================================================
# Entire Barcode Statistics - Statlog (6_Experiment_Impact_of_H0_Only)
# =============================================================================
dcccd_data_L5_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/data_L5.csv"))
dcccd_data_L15_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/6_Experiment_Impact_of_H0_Only/data_L15.csv"))





# %%
# Dataset as a dictionary

all_dcccd_datasets = {
    "dcccd_barcode_stats_default_L5": dcccd_barcode_stats_default_L5,
    "dcccd_barcode_stats_default_L15": dcccd_barcode_stats_default_L15,
    "dcccd_barcode_stats_non_default_L5": dcccd_barcode_stats_non_default_L5,
    "dcccd_barcode_stats_non_default_L15": dcccd_barcode_stats_non_default_L15,
    "dcccd_data_L5": dcccd_data_L5,
    "dcccd_data_L15": dcccd_data_L15,

    "dcccd_barcode_stats_default_L5_4_PH_Tuned_Parameters": dcccd_barcode_stats_default_L5_4_PH_Tuned_Parameters,
    "dcccd_barcode_stats_default_L15_4_PH_Tuned_Parameters": dcccd_barcode_stats_default_L15_4_PH_Tuned_Parameters,
    "dcccd_barcode_stats_non_default_L5_4_PH_Tuned_Parameters": dcccd_barcode_stats_non_default_L5_4_PH_Tuned_Parameters,
    "dcccd_barcode_stats_non_default_L15_4_PH_Tuned_Parameters": dcccd_barcode_stats_non_default_L15_4_PH_Tuned_Parameters,
    "dcccd_data_L5_4_PH_Tuned_Parameters": dcccd_data_L5_4_PH_Tuned_Parameters,
    "dcccd_data_L15_4_PH_Tuned_Parameters": dcccd_data_L15_4_PH_Tuned_Parameters,

    "dcccd_barcode_stats_default_L5_6_Experiment_Impact_of_H0_Only": dcccd_barcode_stats_default_L5_6_Experiment_Impact_of_H0_Only,
    "dcccd_barcode_stats_default_L15_6_Experiment_Impact_of_H0_Only": dcccd_barcode_stats_default_L15_6_Experiment_Impact_of_H0_Only,
    "dcccd_barcode_stats_non_default_L5_6_Experiment_Impact_of_H0_Only": dcccd_barcode_stats_non_default_L5_6_Experiment_Impact_of_H0_Only,
    "dcccd_barcode_stats_non_default_L15_6_Experiment_Impact_of_H0_Only": dcccd_barcode_stats_non_default_L15_6_Experiment_Impact_of_H0_Only,
    "dcccd_data_L5_6_Experiment_Impact_of_H0_Only": dcccd_data_L5_6_Experiment_Impact_of_H0_Only,
    "dcccd_data_L15_6_Experiment_Impact_of_H0_Only": dcccd_data_L15_6_Experiment_Impact_of_H0_Only
}





# %%
# =============================================================================
# PCA ON DATASETS
# =============================================================================
save_path = "../../../6_Results/8_Dimensionality_Reduction_On_Barcode_Statistics/Default_Of_Credit_Card_Client_Data/PCA_Results"
os.makedirs(os.path.abspath(save_path), exist_ok=True)

reduced_datasets, pca_metadata, summary_stats = perform_pca_analysis(
    all_dcccd_datasets, 
    output_dir=os.path.abspath(save_path),
    n_components=2,
    target_column = "label"
)
reduced_datasets.keys(), summary_stats.keys()
