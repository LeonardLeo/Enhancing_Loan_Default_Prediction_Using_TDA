# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 2026

Dataset: South German Credit

@author: lEO
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import os
import pandas as pd
from utils import data_preprocessing_pipeline, perform_pca_analysis

# =============================================================================
# GET DATASET
# =============================================================================
data = pd.read_csv(
    os.path.abspath("../../../1_Data/Processed_Datasets/South_German_Credit/processed_data.csv")
)

# =============================================================================
# Preprocessing data
# =============================================================================
data = data_preprocessing_pipeline(
    data,
    log_col=["hoehe", "laufzeit"],
)

# Dataset as a dictionary
dataset_dict = {"South_German_Credit": data}

# =============================================================================
# PCA ON DATASETS
# =============================================================================
# ALERT: Statlog Exp9 uses target_column="label" on processed_data.xlsx even though
# the target column in processed data is "Class". These registry datasets use "target".
save_path = "../../../6_Results/9_Dimensionality_Reduction_On_Original_Dataset/South_German_Credit/PCA_Results"
os.makedirs(save_path, exist_ok=True)

reduced_datasets, pca_metadata, summary_stats = perform_pca_analysis(
    dataset_dict,
    output_dir=os.path.abspath(save_path),
    n_components=2,
    target_column="target",
)
reduced_datasets.keys(), summary_stats.keys()
