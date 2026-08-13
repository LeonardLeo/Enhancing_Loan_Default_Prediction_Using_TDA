# -*- coding: utf-8 -*-
"""
Experiment 7 — EDA of barcode statistics
Dataset: PKDD'99 Czech Financial

Does not rebuild landmarks. Describes Experiment 3 barcode tables.
Missing Experiment 6 (H0-only) files are skipped with a note.

Results: 6_Results/Archives/7_EDA_Barcode_Statistics/PKDD_Czech_Financial/
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import eda, store_results

warnings.filterwarnings("ignore")

FOLDER = "PKDD_Czech_Financial"
SAVE_PATH = "../../../../6_Results/Archives/7_EDA_Barcode_Statistics/" + FOLDER
PERCENTAGES = [10, 20]


def eda_existing(paths, save_name):
    """Run utils.eda on every path that already exists."""
    results = {}
    for path in paths:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            print(f"Skip (not generated yet): {abs_path}")
            continue
        name = os.path.basename(abs_path)
        print(f"EDA {name} ...")
        results[name] = eda(dataset=pd.read_csv(abs_path), graphs=False)
    if results:
        store_results(path=SAVE_PATH, save_name=save_name, result_object=results)
    else:
        print(f"Nothing to save for {save_name}.")


eda_existing(
    [
        f"../../../../1_Data/Barcode_Statistics/{FOLDER}/3_PH_Default_Parameters/barcode_stats_{cls}_L{pct}.csv"
        for pct in PERCENTAGES
        for cls in ("default", "non-default")
    ],
    "eda_each_class_BS",
)
eda_existing(
    [
        f"../../../../1_Data/TDA_Datasets/{FOLDER}/3_PH_Default_Parameters/data_L{pct}.csv"
        for pct in PERCENTAGES
    ],
    "eda_entire_BS",
)
eda_existing(
    [
        f"../../../../1_Data/Barcode_Statistics/{FOLDER}/6_Experiment_Impact_of_H0_Only/barcode_stats_{cls}_L{pct}.csv"
        for pct in PERCENTAGES
        for cls in ("default", "non-default")
    ],
    "eda_each_class_BS_H0",
)
eda_existing(
    [
        f"../../../../1_Data/TDA_Datasets/{FOLDER}/6_Experiment_Impact_of_H0_Only/data_L{pct}.csv"
        for pct in PERCENTAGES
    ],
    "eda_entire_BS_H0",
)
