# -*- coding: utf-8 -*-
"""
Early split, no undersample, using both H0 and H1 / 9_Revised_Snapshot_Protocol
Dataset: South German Credit

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

DATASET_KEY = "south_german_credit"
PROTOCOL_BUCKET = "Early_Split_No_Undersample_H0_And_H1"
EXPERIMENT = "9_Revised_Snapshot_Protocol"
FOLDER = "South_German_Credit"
SPLIT_TIMING = "early"
UNDERSAMPLE = False
PCA_N_COMPONENTS = 10
RUN_ZANIAR_SWEEP = False   # True = also run train {60,80,100} x test {15,22,30}

# Tell the Experiment 9 helpers which arm this file belongs to.
set_revised_snapshot_arm(PROTOCOL_BUCKET, SPLIT_TIMING, UNDERSAMPLE)

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_csv(os.path.join(REPO_ROOT, "1_Data", "Processed_Datasets", "South_German_Credit", "processed_data.csv"))
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
data = data_preprocessing_pipeline(data, log_col=["hoehe", "laufzeit"])
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
