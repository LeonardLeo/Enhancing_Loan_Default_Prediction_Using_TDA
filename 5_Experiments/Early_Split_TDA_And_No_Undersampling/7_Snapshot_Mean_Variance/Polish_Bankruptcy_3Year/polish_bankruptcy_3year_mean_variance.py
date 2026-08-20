# -*- coding: utf-8 -*-
"""
Early_Split_TDA_And_No_Undersampling / 7_Snapshot_Mean_Variance
Dataset: Polish Companies Bankruptcy (3 year)

This file is the method document for this dataset. Heavy Ripser / IO helpers
live in utils.py; the pipeline itself is written here in order.

Protocol
--------
- Split timing : early
- Undersample  : False
- PCA rank     : 10  (historical Exp 3 rank for this table)
- Snapshot size percents : [10.0, 20.0]
- Number of snapshots    : 500
This experiment does not run Ripser. It loads Experiment 1 barcode tables
and records the mean and variance of each barcode-statistic column.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (
    _percent_token,
    store_results,
    summarize_snapshot_statistics,
    tda_artefact_dir,
    tda_results_dir,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'polish_bankruptcy'
PROTOCOL_BUCKET = 'Early_Split_TDA_And_No_Undersampling'
EXPERIMENT = "7_Snapshot_Mean_Variance"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = 'Polish_Bankruptcy_3Year'

SPLIT_TIMING = 'early'
UNDERSAMPLE = False
LANDMARK_PERCENTAGES = [10.0, 20.0]

# =============================================================================
# Load barcode tables
# =============================================================================
sources = []
for split in ("train", "test"):
    for pct in LANDMARK_PERCENTAGES:
        sources.append(
            tda_artefact_dir(
                "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER,
                split, f"data_L{_percent_token(pct)}.csv",
            )
        )

# =============================================================================
# Mean and variance across snapshots
# =============================================================================
all_summaries = {}
flat_rows = []
missing = []
for path in sources:
    if not path.exists():
        missing.append(str(path))
        print(f"Missing (run this arm's experiment 1 first): {path}")
        continue
    summary = summarize_snapshot_statistics(str(path))
    key = f"{PROTOCOL_BUCKET}/{SOURCE_EXPERIMENT}/{FOLDER}/{path.name}"
    if path.parent.name in {"train", "test"}:
        key = f"{PROTOCOL_BUCKET}/{SOURCE_EXPERIMENT}/{FOLDER}/{path.parent.name}/{path.name}"
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
    print(f"OK {path.name}: n={summary['n_snapshots']}")

if not flat_rows:
    raise FileNotFoundError(
        f"No experiment-1 barcode files for {PROTOCOL_BUCKET}/{FOLDER}. Missing: {missing}"
    )

# =============================================================================
# Store results
# =============================================================================
save_path = tda_results_dir(PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path.mkdir(parents=True, exist_ok=True)
pd.DataFrame(flat_rows).to_csv(save_path / "snapshot_mean_variance.csv", index=False)
store_results(path=str(save_path), save_name="snapshot_mean_variance_full", result_object=all_summaries)
