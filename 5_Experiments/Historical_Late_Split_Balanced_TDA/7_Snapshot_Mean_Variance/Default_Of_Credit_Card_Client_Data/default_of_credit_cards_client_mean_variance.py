# -*- coding: utf-8 -*-
"""
Historical Late Split, Balanced TDA / 7_Snapshot_Mean_Variance
Dataset: Default of Credit Card Client

This experiment does not run Ripser. It records the mean and variance of each barcode-statistic column.
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
    store_results,
    summarize_snapshot_statistics,
    flatten_snapshot_mean_variance,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

PROTOCOL_BUCKET = "Historical_Late_Split_Balanced_TDA"
EXPERIMENT = "7_Snapshot_Mean_Variance"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Default_Of_Credit_Card_Client_Data"

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
os.makedirs(save_path, exist_ok=True)

all_summaries = {}
flat_rows = []

# =============================================================================
# Mean and variance - L5
# =============================================================================
path_L5 = os.path.join(src_dir, "data_L5.csv")
if os.path.exists(path_L5):
    summary_path_L5 = summarize_snapshot_statistics(path_L5)
    key_path_L5 = path_L5.replace(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets") + os.sep, "")
    all_summaries[key_path_L5] = summary_path_L5
    flat_rows.extend(flatten_snapshot_mean_variance(summary_path_L5, key_path_L5))
    print("OK", os.path.basename(path_L5), "n=", summary_path_L5["n_snapshots"])
else:
    print("Missing (run this arm's experiment 1 first):", path_L5)
# =============================================================================
# Mean and variance - L15
# =============================================================================
path_L15 = os.path.join(src_dir, "data_L15.csv")
if os.path.exists(path_L15):
    summary_path_L15 = summarize_snapshot_statistics(path_L15)
    key_path_L15 = path_L15.replace(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets") + os.sep, "")
    all_summaries[key_path_L15] = summary_path_L15
    flat_rows.extend(flatten_snapshot_mean_variance(summary_path_L15, key_path_L15))
    print("OK", os.path.basename(path_L15), "n=", summary_path_L15["n_snapshots"])
else:
    print("Missing (run this arm's experiment 1 first):", path_L15)
if not flat_rows:
    raise FileNotFoundError("No Experiment 1 barcode files for " + PROTOCOL_BUCKET + "/" + FOLDER)

pd.DataFrame(flat_rows).to_csv(os.path.join(save_path, "snapshot_mean_variance.csv"), index=False)
store_results(path=save_path, save_name="snapshot_mean_variance_full", result_object=all_summaries)
