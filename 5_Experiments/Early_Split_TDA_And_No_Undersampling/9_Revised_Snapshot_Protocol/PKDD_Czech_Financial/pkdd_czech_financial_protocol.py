# -*- coding: utf-8 -*-
"""
Early Split TDA And No Undersampling / 9_Revised_Snapshot_Protocol
Dataset: PKDD'99 Czech Financial

Revised snapshot protocol: fixed points per snapshot (not a class percent), default 60 train / 15 test snapshots.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings

import numpy as np
import pandas as pd

# This file lives four folders below the repository root (where utils.py is).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import (
    data_preprocessing_pipeline,
    set_revised_snapshot_arm,
    design_for_dataset,
    run_split_ml,
    run_full_ml,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

DATASET_KEY = "pkdd_czech"
PROTOCOL_BUCKET = "Early_Split_TDA_And_No_Undersampling"
EXPERIMENT = "9_Revised_Snapshot_Protocol"
FOLDER = "PKDD_Czech_Financial"
SPLIT_TIMING = "early"
UNDERSAMPLE = False
PCA_N_COMPONENTS = 10
RUN_ZANIAR_SWEEP = False   # True = also run train {60,80,100} x test {15,22,30}

# Tell the Experiment 9 helpers which arm this file belongs to.
set_revised_snapshot_arm(PROTOCOL_BUCKET, SPLIT_TIMING, UNDERSAMPLE)

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_csv(os.path.join(REPO_ROOT, "1_Data", "Processed_Datasets", "PKDD_Czech_Financial", "processed_data.csv"))
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().any():
        data[col] = data[col].fillna(data[col].median())
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].fillna("missing").astype(str)
data = data_preprocessing_pipeline(
    data,
    log_col=["amount", "payments", "tx_amount_sum", "tx_amount_mean"],
    dummy_col=["frequency", "type", "sex", "A2", "A3", "A12", "A15", "preloan_card_type"],
)
X = data.drop(columns=["target"])
y = data["target"].astype(int)
X = X.select_dtypes(include=[np.number]).copy()

print("Loaded", FOLDER, "rows:", len(X), "columns:", X.shape[1])
print("split timing:", SPLIT_TIMING, " undersample:", UNDERSAMPLE, " PCA rank:", PCA_N_COMPONENTS)

# =============================================================================
# Stage 1 - Design: intrinsic dimension, joint points-per-snapshot, reuse
# =============================================================================
design = design_for_dataset(DATASET_KEY, X=X, y=y)
print(
    "Chosen points per snapshot =", design["chosen_t"],
    " effective train snapshots =", design.get("effective_defaults", {}).get("train_l"),
    " test snapshots =", design.get("effective_defaults", {}).get("test_l"),
)

# =============================================================================
# Stage 2 - Draw snapshots, Ripser, overlap, classifiers
# =============================================================================
run_split_ml({DATASET_KEY: design}, [DATASET_KEY], sweep=RUN_ZANIAR_SWEEP)

# =============================================================================
# Stage 3 - Optional full-table (non-split) arm on DCCCD only
# =============================================================================
print("Full-table non-split arm is DCCCD-only; skipped for this dataset.")
