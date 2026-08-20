# -*- coding: utf-8 -*-
"""
No_Undersampling / 3_H0_Only
Dataset: Default of Credit Card Client

This file is the method document for this dataset. Heavy Ripser / IO helpers
live in utils.py; the pipeline itself is written here in order.

Protocol
--------
- Split timing : late
- Undersample  : False
- PCA rank     : 7  (historical Exp 3 rank for this table)
- Snapshot size percents : [5.0, 15.0]
- Number of snapshots    : 500
This experiment does not run Ripser. It loads Experiment 1 barcode tables,
keeps only H0 (dimension-0) columns, then trains the five default classifiers.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (
    _percent_token,
    _write_h0_slice,
    store_results,
    tda_artefact_dir,
    tda_results_dir,
    train_multiple_dataset_tda,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'credit_card_default'
PROTOCOL_BUCKET = 'No_Undersampling'
EXPERIMENT = "3_H0_Only"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = 'Default_Of_Credit_Card_Client_Data'

SPLIT_TIMING = 'late'
UNDERSAMPLE = False
LANDMARK_PERCENTAGES = [5.0, 15.0]
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =============================================================================
# Load barcode tables and keep H0 columns
# =============================================================================
paths = []
dest_root = tda_artefact_dir("TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
for pct in LANDMARK_PERCENTAGES:
    token = _percent_token(pct)
    src = tda_artefact_dir(
        "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER, f"data_L{token}.csv"
    )
    dest = dest_root / f"data_L{token}.csv"
    print(f"Filter H0: {src} -> {dest}")
    _write_h0_slice(src, dest)
    paths.append(str(dest))

# =============================================================================
# Train models
# =============================================================================
model_results = train_multiple_dataset_tda(
    path_datasets=paths,
    y_col_name="label",
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    xgb={"eval_metric": "logloss"},
)

print(model_results)

# =============================================================================
# Store results
# =============================================================================
save_path = tda_results_dir(PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
store_results(path=str(save_path), save_name="model_results", result_object=model_results)
