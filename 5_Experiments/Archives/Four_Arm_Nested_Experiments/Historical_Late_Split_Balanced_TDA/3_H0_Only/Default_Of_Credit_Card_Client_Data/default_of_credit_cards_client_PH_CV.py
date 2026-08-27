# -*- coding: utf-8 -*-
"""Cross-validation consumer. Artefact paths go through utils.tda_artefact_dir."""
import sys
import warnings
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (
    late_split_barcode_paths,
    store_results,
    perform_cross_validation_tda,
    tda_results_dir,
)

warnings.filterwarnings("ignore")

data_paths = late_split_barcode_paths(
    "Historical_Late_Split_Balanced_TDA",
    "Default_Of_Credit_Card_Client_Data",
    [5, 15],
    "3_H0_Only",
)
save_path = str(tda_results_dir(
    "Historical_Late_Split_Balanced_TDA",
    "3_H0_Only",
    "Default_Of_Credit_Card_Client_Data",
))

try:
    evaluation_results = joblib.load(str(Path(save_path) / "model_results.pkl"))
except FileNotFoundError:
    print("Evaluation results file not found!")
    evaluation_results = {}

model_results = perform_cross_validation_tda(
    datasets=data_paths,
    model_results=evaluation_results,
)
store_results(path=save_path, save_name="CV_results", result_object=model_results)
