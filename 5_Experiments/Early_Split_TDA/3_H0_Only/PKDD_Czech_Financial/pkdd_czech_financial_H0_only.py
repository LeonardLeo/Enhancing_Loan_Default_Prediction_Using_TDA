# -*- coding: utf-8 -*-
"""
Early_Split_TDA / 3_H0_Only
Dataset: PKDD'99 Czech Financial

This file is the method document for this dataset. Heavy Ripser / IO helpers
live in utils.py; the pipeline itself is written here in order.

Protocol
--------
- Split timing : early
- Undersample  : True
- PCA rank     : 10  (historical Exp 3 rank for this table)
- Snapshot size percents : [10.0, 20.0]
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
    early_split_barcode_pairs,
    train_multiple_dataset_tda_presplit,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'pkdd_czech'
PROTOCOL_BUCKET = 'Early_Split_TDA'
EXPERIMENT = "3_H0_Only"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = 'PKDD_Czech_Financial'

SPLIT_TIMING = 'early'
UNDERSAMPLE = True
LANDMARK_PERCENTAGES = [10.0, 20.0]
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =============================================================================
# Load barcode tables and keep H0 columns
# =============================================================================
pairs = {}
dest_root = tda_artefact_dir("TDA_Datasets", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
for pct in LANDMARK_PERCENTAGES:
    token = _percent_token(pct)
    src_train = tda_artefact_dir(
        "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER, "train", f"data_L{token}.csv"
    )
    src_test = tda_artefact_dir(
        "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER, "test", f"data_L{token}.csv"
    )
    dest_train = dest_root / "train" / f"data_L{token}.csv"
    dest_test = dest_root / "test" / f"data_L{token}.csv"
    print(f"Filter H0: {src_train} -> {dest_train}")
    _write_h0_slice(src_train, dest_train)
    _write_h0_slice(src_test, dest_test)
    pairs[f"data_L{token}"] = {"train": str(dest_train), "test": str(dest_test)}

# =============================================================================
# Train models
# =============================================================================
model_results = train_multiple_dataset_tda_presplit(
    train_test_pairs=pairs,
    y_col_name="label",
    random_state=RANDOM_STATE,
    xgb={"eval_metric": "logloss"},
)

print(model_results)

# =============================================================================
# Store results
# =============================================================================
save_path = tda_results_dir(PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
store_results(path=str(save_path), save_name="model_results", result_object=model_results)
