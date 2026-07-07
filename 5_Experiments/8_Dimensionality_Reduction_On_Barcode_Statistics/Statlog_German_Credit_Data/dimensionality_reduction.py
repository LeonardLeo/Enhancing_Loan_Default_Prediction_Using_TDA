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
sgcd_barcode_stats_default_L30 = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/3_PH_Default_Parameters/barcode_stats_default_L30.csv"))
sgcd_barcode_stats_default_L60 = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/3_PH_Default_Parameters/barcode_stats_default_L60.csv"))
sgcd_barcode_stats_non_default_L30 = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/3_PH_Default_Parameters/barcode_stats_non-default_L30.csv"))
sgcd_barcode_stats_non_default_L60 = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/3_PH_Default_Parameters/barcode_stats_non-default_L60.csv"))

#%%
# =============================================================================
# Entire Barcode Statistics - Statlog (3_PH_Default_Parameters)
# =============================================================================
sgcd_data_L30 = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/3_PH_Default_Parameters/data_L30.csv"))
sgcd_data_L60 = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/3_PH_Default_Parameters/data_L60.csv"))





#%%
# =============================================================================
# Barcode Statistics - Statlog (4_PH_Tuned_Parameters)
# =============================================================================
sgcd_barcode_stats_default_L30_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/4_PH_Tuned_Parameters/barcode_stats_default_L30.csv"))
sgcd_barcode_stats_default_L60_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/4_PH_Tuned_Parameters/barcode_stats_default_L60.csv"))
sgcd_barcode_stats_non_default_L30_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/4_PH_Tuned_Parameters/barcode_stats_non-default_L30.csv"))
sgcd_barcode_stats_non_default_L60_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/4_PH_Tuned_Parameters/barcode_stats_non-default_L60.csv"))

#%%
# =============================================================================
# Entire Barcode Statistics - Statlog (4_PH_Tuned_Parameters)
# =============================================================================
sgcd_data_L30_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/4_PH_Tuned_Parameters/data_L30.csv"))
sgcd_data_L60_4_PH_Tuned_Parameters = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/4_PH_Tuned_Parameters/data_L60.csv"))





#%%
# =============================================================================
# Barcode Statistics - Statlog (6_Experiment_Impact_of_H0_Only)
# =============================================================================
sgcd_barcode_stats_default_L30_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_default_L30.csv"))
sgcd_barcode_stats_default_L60_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_default_L60.csv"))
sgcd_barcode_stats_non_default_L30_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_non-default_L30.csv"))
sgcd_barcode_stats_non_default_L60_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/Barcode_Statistics/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/barcode_stats_non-default_L60.csv"))

#%%
# =============================================================================
# Entire Barcode Statistics - Statlog (6_Experiment_Impact_of_H0_Only)
# =============================================================================
sgcd_data_L30_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/data_L30.csv"))
sgcd_data_L60_6_Experiment_Impact_of_H0_Only = pd.read_csv(os.path.abspath("../../../1_Data/TDA_Datasets/Statlog_German_Credit_Data/6_Experiment_Impact_of_H0_Only/data_L60.csv"))





# %%
# Dataset as a dictionary

all_sgcd_datasets = {
    "sgcd_barcode_stats_default_L30": sgcd_barcode_stats_default_L30,
    "sgcd_barcode_stats_default_L60": sgcd_barcode_stats_default_L60,
    "sgcd_barcode_stats_non_default_L30": sgcd_barcode_stats_non_default_L30,
    "sgcd_barcode_stats_non_default_L60": sgcd_barcode_stats_non_default_L60,
    "sgcd_data_L30": sgcd_data_L30,
    "sgcd_data_L60": sgcd_data_L60,

    "sgcd_barcode_stats_default_L30_4_PH_Tuned_Parameters": sgcd_barcode_stats_default_L30_4_PH_Tuned_Parameters,
    "sgcd_barcode_stats_default_L60_4_PH_Tuned_Parameters": sgcd_barcode_stats_default_L60_4_PH_Tuned_Parameters,
    "sgcd_barcode_stats_non_default_L30_4_PH_Tuned_Parameters": sgcd_barcode_stats_non_default_L30_4_PH_Tuned_Parameters,
    "sgcd_barcode_stats_non_default_L60_4_PH_Tuned_Parameters": sgcd_barcode_stats_non_default_L60_4_PH_Tuned_Parameters,
    "sgcd_data_L30_4_PH_Tuned_Parameters": sgcd_data_L30_4_PH_Tuned_Parameters,
    "sgcd_data_L60_4_PH_Tuned_Parameters": sgcd_data_L60_4_PH_Tuned_Parameters,

    "sgcd_barcode_stats_default_L30_6_Experiment_Impact_of_H0_Only": sgcd_barcode_stats_default_L30_6_Experiment_Impact_of_H0_Only,
    "sgcd_barcode_stats_default_L60_6_Experiment_Impact_of_H0_Only": sgcd_barcode_stats_default_L60_6_Experiment_Impact_of_H0_Only,
    "sgcd_barcode_stats_non_default_L30_6_Experiment_Impact_of_H0_Only": sgcd_barcode_stats_non_default_L30_6_Experiment_Impact_of_H0_Only,
    "sgcd_barcode_stats_non_default_L60_6_Experiment_Impact_of_H0_Only": sgcd_barcode_stats_non_default_L60_6_Experiment_Impact_of_H0_Only,
    "sgcd_data_L30_6_Experiment_Impact_of_H0_Only": sgcd_data_L30_6_Experiment_Impact_of_H0_Only,
    "sgcd_data_L60_6_Experiment_Impact_of_H0_Only": sgcd_data_L60_6_Experiment_Impact_of_H0_Only
}





# %%
# =============================================================================
# PCA ON DATASETS
# =============================================================================
save_path = "../../../6_Results/8_Dimensionality_Reduction_On_Barcode_Statistics/Statlog_German_Credit_Data/PCA_Results"
os.makedirs(save_path, exist_ok=True)

reduced_datasets, pca_metadata, summary_stats = perform_pca_analysis(
    all_sgcd_datasets, 
    output_dir=os.path.abspath(save_path),
    n_components=2,
    target_column = "label"
)
reduced_datasets.keys(), summary_stats.keys()
