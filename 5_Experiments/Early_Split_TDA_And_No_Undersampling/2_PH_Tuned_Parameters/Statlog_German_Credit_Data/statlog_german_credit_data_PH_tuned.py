# -*- coding: utf-8 -*-
"""
Early_Split_TDA_And_No_Undersampling / 2_PH_Tuned_Parameters
Dataset: Statlog German Credit

This file is the method document for this dataset. Heavy Ripser / IO helpers
live in utils.py; the pipeline itself is written here in order.

Protocol
--------
- Split timing : early
- Undersample  : False
- PCA rank     : 15  (historical Exp 3 rank for this table)
- Snapshot size percents : [30.0, 60.0]
- Number of snapshots    : 500
This experiment does not run Ripser. It loads Experiment 1 barcode tables
and retrains the five classifiers with GridSearchCV (F1 scoring).
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
    DEFAULT_TDA_TUNED_MODEL_CONFIGS,
    store_results,
    tda_results_dir,
    early_split_barcode_pairs,
    train_models_on_multiple_presplit_datasets,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'statlog_german'
PROTOCOL_BUCKET = 'Early_Split_TDA_And_No_Undersampling'
EXPERIMENT = "2_PH_Tuned_Parameters"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = 'Statlog_German_Credit_Data'

SPLIT_TIMING = 'early'
UNDERSAMPLE = False
LANDMARK_PERCENTAGES = [30.0, 60.0]
RANDOM_STATE = 42
TEST_SIZE = 0.2
SCORING = "f1"
N_SPLITS_KFOLD = 5

# GridSearchCV grids (utils.DEFAULT_TDA_TUNED_MODEL_CONFIGS):
#   SVM            C in {0.1, 1, 10}, kernel in {linear, rbf}, gamma in {scale, auto}
#   KNN            n_neighbors in {3, 5, 7}, weights in {uniform, distance}, p in {1, 2}
#   XGBoost        n_estimators in {50, 100, 200}, learning_rate in {0.01, 0.1, 0.2}, max_depth in {3, 5, 7}
#   Logistic       C in {0.1, 1, 10}, solver in {liblinear, lbfgs}
#   Random Forest  n_estimators in {50, 100, 200}, max_depth in {3, 5, 10, None}, min_samples_split in {2, 5, 10}

# =============================================================================
# Load barcode tables (Experiment 1, already split by customer)
# =============================================================================
pairs = early_split_barcode_pairs(
    PROTOCOL_BUCKET, FOLDER, LANDMARK_PERCENTAGES, SOURCE_EXPERIMENT
)
for name, paths in pairs.items():
    print(f"Train table: {paths['train']}")
    print(f"Test table : {paths['test']}")

# =============================================================================
# Train models (tuned)
# =============================================================================
model_results = train_models_on_multiple_presplit_datasets(
    train_test_pairs=pairs,
    model_configs=DEFAULT_TDA_TUNED_MODEL_CONFIGS,
    target_column="label",
    scoring_metric=SCORING,
    scale_features=True,
    random_state=RANDOM_STATE,
    n_splits_kfold=N_SPLITS_KFOLD,
)

print(model_results)

# =============================================================================
# Store results
# =============================================================================
save_path = tda_results_dir(PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
store_results(path=str(save_path), save_name="model_results", result_object=model_results)
