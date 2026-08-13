# -*- coding: utf-8 -*-
"""
Experiment 15 — Working with k in KNN
Dataset: Taiwanese Bankruptcy Prediction

Consumes Experiment 3 barcode matrices (does not rebuild landmarks).
Sweeps k on those TDA features and stores the curve.

Results: 6_Results/15_Working_With_K_in_KNN/Taiwan_Bankruptcy/
"""

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import store_results, train_multiple_knn_datasets

warnings.filterwarnings("ignore")

FOLDER = "Taiwan_Bankruptcy"
PERCENTAGES = [10, 20]
SAVE_PATH = "../../../6_Results/15_Working_With_K_in_KNN/" + FOLDER

paths = []
for pct in PERCENTAGES:
    path = os.path.abspath(
        f"../../../1_Data/TDA_Datasets/{FOLDER}/3_PH_Default_Parameters/data_L{pct}.csv"
    )
    if os.path.exists(path):
        paths.append(path)
    else:
        print(f"Missing Exp 3 matrix: {path}")

if not paths:
    print("Run Experiment 3 first so data_L*.csv exist, then re-run this script.")
else:
    model_results = train_multiple_knn_datasets(
        path_datasets=paths,
        y_col_name="label",
        test_size=0.2,
        random_state=42,
        base_output_path=SAVE_PATH,
    )
    print(model_results)
    store_results(path=SAVE_PATH, save_name="model_results", result_object=model_results)
