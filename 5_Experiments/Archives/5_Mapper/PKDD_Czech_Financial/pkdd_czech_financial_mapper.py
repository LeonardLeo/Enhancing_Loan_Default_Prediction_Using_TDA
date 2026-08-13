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

# =============================================================================
# Import Libraries
# =============================================================================
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import data_preprocessing_pipeline, build_mapper_viz

warnings.filterwarnings("ignore")

# =============================================================================
# Get Dataset
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

# =============================================================================
# Select dependent and independent variables
# =============================================================================
X = data.drop(columns=["target"])
y = data["target"]

# =============================================================================
# Split dataset into training and test data
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# Normalize the features (train-only fit)
# =============================================================================
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
train_features = X_train_scaled.to_numpy()

# =============================================================================
# Build Mapper visualizations on training features
# =============================================================================
save_path = "../../../../6_Results/Archives/5_Mapper/PKDD_Czech_Financial"
os.makedirs(save_path, exist_ok=True)

build_mapper_viz(
    data=train_features,
    resampled_data_label=y_train.reset_index(drop=True),
    resolution=[50, 70, 90],
    percentage_overlap=[0.25, 0.4, 0.6],
    clustering_grid={
        "kmeans": [
            {"n_clusters": 2, "random_state": 42, "n_init": 10},
            {"n_clusters": 3, "random_state": 42, "n_init": 10},
            {"n_clusters": 4, "random_state": 42, "n_init": 10},
        ]
    },
    lens_methods=["pca"],
    lens_params={"pca": {"n_components": 1}},
    color_functions=["lens", "labels"],
    color_function_name=["PCA Lens", "Default Status"],
    output_dir=save_path,
)
