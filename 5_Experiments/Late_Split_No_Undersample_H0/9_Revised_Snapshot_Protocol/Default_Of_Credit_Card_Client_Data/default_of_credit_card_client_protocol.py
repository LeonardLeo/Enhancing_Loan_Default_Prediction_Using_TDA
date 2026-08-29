# -*- coding: utf-8 -*-
"""
Late split, no undersample, using just H0 / 9_Revised_Snapshot_Protocol
Dataset: Default of Credit Card Client

This experiment does not run Ripser.

Revised snapshot protocol: fixed points per snapshot (not a class percent), default 60 train / 15 test snapshots.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings

import pandas as pd

# This file lives four folders below the repository root (where utils.py is).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import (
    set_revised_snapshot_arm,
    design_for_dataset,
    run_h0_slice_revised_protocol,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

DATASET_KEY = "credit_card_default"
PROTOCOL_BUCKET = "Late_Split_No_Undersample_H0"
EXPERIMENT = "9_Revised_Snapshot_Protocol"
FOLDER = "Default_Of_Credit_Card_Client_Data"
SPLIT_TIMING = "late"
UNDERSAMPLE = False
PCA_N_COMPONENTS = 7
RUN_ZANIAR_SWEEP = False   # True = also run train {60,80,100} x test {15,22,30}

# Tell the Experiment 9 helpers which arm this file belongs to.
set_revised_snapshot_arm(PROTOCOL_BUCKET, SPLIT_TIMING, UNDERSAMPLE)

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_excel(os.path.join(REPO_ROOT, "1_Data", "Processed_Datasets", "Default_Of_Credit_Card_Client_Data", "processed_data.xlsx"))
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
X = data.drop(columns=["default payment next month"])
y = data["default payment next month"].astype(int)

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
# Stage 2 - Slice H0 columns from the sibling Experiment 9 barcodes and retrain
# =============================================================================
run_h0_slice_revised_protocol(DATASET_KEY, design)
