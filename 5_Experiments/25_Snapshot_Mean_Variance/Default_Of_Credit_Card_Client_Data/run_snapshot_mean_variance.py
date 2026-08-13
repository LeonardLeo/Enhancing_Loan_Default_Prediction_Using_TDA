# -*- coding: utf-8 -*-
"""
Experiment 25 — Snapshot Mean and Variance
Dataset: Default of Credit Card Clients (UCI)

Question
--------
Across the barcode-statistic snapshots produced by Experiment 3, how stable
is each topological summary (mean birth, mean death, ...)?  Large variance
means successive snapshots disagree; small variance means the barcode
vector is a stable description of that class cloud.

What this script does (in order)
--------------------------------
1. Load Experiment 3 matrices (data_L5.csv / data_L15.csv).
2. For every barcode column, compute mean and sample variance.
3. Also store the 24-number mean vector as a proxy for the landscape
   average (not a full persistence landscape — see the report).
4. Write a flat CSV (one row per feature) plus a pickle of the full dict.

Prerequisite: Experiment 3 must have written those CSVs.
Results: 6_Results/25_Snapshot_Mean_Variance/Default_Of_Credit_Card_Client_Data/
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import store_results, summarize_snapshot_statistics

warnings.filterwarnings("ignore")

# =============================================================================
# Dataset settings (this folder only)
# =============================================================================
FOLDER = "Default_Of_Credit_Card_Client_Data"
SAVE_PATH = f"../../../6_Results/25_Snapshot_Mean_Variance/{FOLDER}"
TDA_DIR = f"../../../1_Data/TDA_Datasets/{FOLDER}/3_PH_Default_Parameters"
SOURCES = [
    os.path.abspath(f"{TDA_DIR}/data_L5.csv"),
    os.path.abspath(f"{TDA_DIR}/data_L15.csv"),
]

# =============================================================================
# Stage 1 — Summarise each Exp 3 barcode matrix
# =============================================================================
all_summaries = {}
flat_rows = []
missing = []

for path in SOURCES:
    if not os.path.exists(path):
        missing.append(path)
        print(f"Missing (run Experiment 3 first): {path}")
        continue

    summary = summarize_snapshot_statistics(path)
    key = f"{FOLDER}/3_PH_Default_Parameters/{os.path.basename(path)}"
    all_summaries[key] = summary

    for feat, mean_v in summary["global_mean"].items():
        flat_rows.append(
            {
                "source": key,
                "feature": feat,
                "mean": mean_v,
                "variance": summary["global_variance"][feat],
                "n_snapshots": summary["n_snapshots"],
            }
        )
    print(
        f"OK {os.path.basename(path)}: "
        f"n={summary['n_snapshots']}, features={len(summary['feature_columns'])}"
    )

# =============================================================================
# Stage 2 — Save (or stop with a clear message)
# =============================================================================
if not flat_rows:
    print(
        f"\nNo Experiment 3 barcode files under {TDA_DIR}. "
        f"Run the matching Exp 3 script, then re-run this file."
    )
else:
    os.makedirs(os.path.abspath(SAVE_PATH), exist_ok=True)
    pd.DataFrame(flat_rows).to_csv(
        os.path.abspath(f"{SAVE_PATH}/snapshot_mean_variance.csv"), index=False
    )
    store_results(
        path=SAVE_PATH,
        save_name="snapshot_mean_variance_full",
        result_object=all_summaries,
    )
    if missing:
        print(f"Processed available files; {len(missing)} file(s) still missing.")
