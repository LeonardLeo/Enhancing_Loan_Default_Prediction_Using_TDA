# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 2026

Dataset: Polish Companies Bankruptcy (3 year)

@author: lEO
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import os
import pandas as pd
from utils import data_preprocessing_pipeline, perform_pca_analysis

# =============================================================================
# GET DATASET
# =============================================================================
data = pd.read_csv(
    os.path.abspath("../../../../1_Data/Processed_Datasets/Polish_Bankruptcy_3Year/processed_data.csv")
)

# =============================================================================
# Preprocessing data
# =============================================================================
# Missing values → median (Polish ARFF attributes contain NaNs)
for col in data.columns:
    if col != "target" and data[col].isnull().any():
        data[col] = data[col].fillna(data[col].median())

data = data_preprocessing_pipeline(data)

# Dataset as a dictionary
dataset_dict = {"Polish_Bankruptcy_3Year": data}

# =============================================================================
# PCA ON DATASETS
# =============================================================================
# ALERT: Statlog Exp9 uses target_column="label" on processed_data.xlsx even though
# the target column in processed data is "Class". These registry datasets use "target".
save_path = "../../../../6_Results/Archives/9_Dimensionality_Reduction_On_Original_Dataset/Polish_Bankruptcy_3Year/PCA_Results"
os.makedirs(save_path, exist_ok=True)

reduced_datasets, pca_metadata, summary_stats = perform_pca_analysis(
    dataset_dict,
    output_dir=os.path.abspath(save_path),
    n_components=2,
    target_column="target",
)
reduced_datasets.keys(), summary_stats.keys()
