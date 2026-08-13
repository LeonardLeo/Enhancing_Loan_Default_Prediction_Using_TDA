# -*- coding: utf-8 -*-
"""
Experiment 8 — PCA on barcode statistics
Dataset: Taiwanese Bankruptcy Prediction

Consumes Experiment 3 combined matrices only (data_L*.csv).
Tuned / H0-only tables are included when present.

Results: 6_Results/Archives/8_Dimensionality_Reduction_On_Barcode_Statistics/Taiwan_Bankruptcy/
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import perform_pca_analysis

warnings.filterwarnings("ignore")

FOLDER = "Taiwan_Bankruptcy"
PERCENTAGES = [10, 20]
SAVE_PATH = "../../../../6_Results/Archives/8_Dimensionality_Reduction_On_Barcode_Statistics/" + FOLDER + "/PCA_Results"

candidates = []
for pct in PERCENTAGES:
    candidates.append(
        ("exp3_combined_L%s" % pct,
         "../../../../1_Data/TDA_Datasets/%s/3_PH_Default_Parameters/data_L%s.csv" % (FOLDER, pct))
    )
    for cls in ("default", "non-default"):
        candidates.append(
            ("exp3_%s_L%s" % (cls, pct),
             "../../../../1_Data/Barcode_Statistics/%s/3_PH_Default_Parameters/barcode_stats_%s_L%s.csv" % (FOLDER, cls, pct))
        )

datasets = {}
for name, path in candidates:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        datasets[name] = pd.read_csv(abs_path)
        print(f"Loaded {name}: {datasets[name].shape}")
    else:
        print(f"Skip (run Experiment 3 first): {abs_path}")

if not datasets:
    print("No barcode tables found. Run Experiment 3, then re-run this script.")
else:
    os.makedirs(os.path.abspath(SAVE_PATH), exist_ok=True)
    perform_pca_analysis(
        datasets,
        output_dir=os.path.abspath(SAVE_PATH),
        n_components=2,
        target_column="label",
    )
