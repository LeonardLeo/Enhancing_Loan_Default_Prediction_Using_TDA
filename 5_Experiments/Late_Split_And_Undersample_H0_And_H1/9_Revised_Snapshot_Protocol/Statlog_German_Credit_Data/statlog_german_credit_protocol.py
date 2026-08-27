# -*- coding: utf-8 -*-
"""
Late split and undersample (the original historical run), using both H0 and H1 / 9_Revised_Snapshot_Protocol
Dataset: Statlog German Credit

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
    run_split_ml,
    run_full_ml,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

DATASET_KEY = "statlog_german"
PROTOCOL_BUCKET = "Late_Split_And_Undersample_H0_And_H1"
EXPERIMENT = "9_Revised_Snapshot_Protocol"
FOLDER = "Statlog_German_Credit_Data"
SPLIT_TIMING = "late"
UNDERSAMPLE = True
PCA_N_COMPONENTS = 15
RUN_ZANIAR_SWEEP = False   # True = also run train {60,80,100} x test {15,22,30}

# Tell the Experiment 9 helpers which arm this file belongs to.
set_revised_snapshot_arm(PROTOCOL_BUCKET, SPLIT_TIMING, UNDERSAMPLE)

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_excel(os.path.join(REPO_ROOT, "1_Data", "Processed_Datasets", "Statlog_German_Credit_Data", "processed_data.xlsx"))
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
X = data.drop(columns=["Class"])
y = data["Class"].astype(int)

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
