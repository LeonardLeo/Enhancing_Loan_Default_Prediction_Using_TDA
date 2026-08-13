# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 2026

Dataset: PKDD'99 Czech Financial

@author: lEO
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import os
import numpy as np
import pandas as pd
from utils import data_preprocessing_pipeline, perform_pca_analysis

# =============================================================================
# GET DATASET
# =============================================================================
data = pd.read_csv(
    os.path.abspath("../../../../1_Data/Processed_Datasets/PKDD_Czech_Financial/processed_data.csv")
)

# =============================================================================
# Preprocessing data
# =============================================================================
# Missing values (numeric → median; categorical → explicit missing token)
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().any():
        data[col] = data[col].fillna(data[col].median())
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].fillna("missing").astype(str)

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
data = data_preprocessing_pipeline(
    data,
    log_col=["amount", "payments", "tx_amount_sum", "tx_amount_mean"],
    dummy_col=dummy_col,
)

# Dataset as a dictionary
dataset_dict = {"PKDD_Czech_Financial": data}

# =============================================================================
# PCA ON DATASETS
# =============================================================================
# ALERT: Statlog Exp9 uses target_column="label" on processed_data.xlsx even though
# the target column in processed data is "Class". These registry datasets use "target".
save_path = "../../../../6_Results/Archives/9_Dimensionality_Reduction_On_Original_Dataset/PKDD_Czech_Financial/PCA_Results"
os.makedirs(save_path, exist_ok=True)

reduced_datasets, pca_metadata, summary_stats = perform_pca_analysis(
    dataset_dict,
    output_dir=os.path.abspath(save_path),
    n_components=2,
    target_column="target",
)
reduced_datasets.keys(), summary_stats.keys()
