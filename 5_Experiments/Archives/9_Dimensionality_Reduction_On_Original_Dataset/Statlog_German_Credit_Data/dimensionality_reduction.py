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
# Step 1: Load and normalize the dataset
data = pd.read_excel("../../../../1_Data/Processed_Datasets/Statlog_German_Credit_Data/processed_data.xlsx")

# Dataset as a dictionary
dataset_dict = {"Statlog_German_Credit_Data": data}

# %%
# =============================================================================
# PCA ON DATASETS
# =============================================================================
save_path = "../../../../6_Results/Archives/9_Dimensionality_Reduction_On_Original_Dataset/Statlog_German_Credit_Data/PCA_Results"
os.makedirs(save_path, exist_ok=True)

reduced_datasets, pca_metadata, summary_stats = perform_pca_analysis(
    dataset_dict, 
    output_dir=os.path.abspath(save_path),
    n_components=2,
    target_column = "label"
)
reduced_datasets.keys(), summary_stats.keys()
