# -*- coding: utf-8 -*-
"""
Experiment 19 — Linear regression as a classifier
Dataset: Polish Companies Bankruptcy (3-year)

Consumes Experiment 3 barcode matrices.  Statlog's original Exp 19
recomputed H0-only barcodes; here we keep the H0 columns (g*_0) from
the existing Exp 3 table so we do not rebuild 500 landmark files.

Results: 6_Results/19_Linear_Regression_For_Prediction/Polish_Bankruptcy_3Year/
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import store_results, train_multiple_dataset_tda_linear_regression

warnings.filterwarnings("ignore")

FOLDER = "Polish_Bankruptcy_3Year"
PERCENTAGES = [10, 20]
SAVE_PATH = "../../../6_Results/19_Linear_Regression_For_Prediction/" + FOLDER
TMP_DIR = os.path.abspath("../../../1_Data/TDA_Datasets/" + FOLDER + "/19_Linear_Regression_For_Prediction")

paths = []
os.makedirs(TMP_DIR, exist_ok=True)
for pct in PERCENTAGES:
    src = os.path.abspath(
        f"../../../1_Data/TDA_Datasets/{FOLDER}/3_PH_Default_Parameters/data_L{pct}.csv"
    )
    if not os.path.exists(src):
        print(f"Missing Exp 3 matrix: {src}")
        continue
    df = pd.read_csv(src)
    keep = [c for c in df.columns if c == "label" or c.endswith("_0")]
    dest = os.path.join(TMP_DIR, f"data_L{pct}.csv")
    df[keep].to_csv(dest, index=False)
    paths.append(dest)
    print(f"H0-only slice L{pct}: {df[keep].shape} -> {dest}")

if not paths:
    print("Run Experiment 3 first so data_L*.csv exist, then re-run this script.")
else:
    model_results = train_multiple_dataset_tda_linear_regression(
        path_datasets=paths,
        y_col_name="label",
        test_size=0.2,
        random_state=42,
    )
    print(model_results)
    store_results(path=SAVE_PATH, save_name="model_results", result_object=model_results)
