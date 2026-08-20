# -*- coding: utf-8 -*-
"""
No_Undersampling / 6_Sampling_Ratio_Audit
Dataset: South German Credit

This file is the method document for this dataset. Heavy Ripser / IO helpers
live in utils.py; the pipeline itself is written here in order.

Protocol
--------
- Split timing : late
- Undersample  : False
- PCA rank     : 10  (historical Exp 3 rank for this table)
- Snapshot size percents : [10.0, 20.0]
- Number of snapshots    : 500
This experiment does not run Ripser. It audits reuse on the class pools
after this arm's split and optional undersample.

Reuse ratio:
    R = (points_per_snapshot * n_snapshots) / class_count
with points_per_snapshot = floor(class_count * snapshot_size_percent / 100).
"""

# =============================================================================
# Import Libraries
# =============================================================================
import math
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (
    compute_sampling_ratio_audit,
    load_processed_features,
    store_results,
    tda_results_dir,

)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'south_german_credit'
PROTOCOL_BUCKET = 'No_Undersampling'
EXPERIMENT = "6_Sampling_Ratio_Audit"
FOLDER = 'South_German_Credit'

SPLIT_TIMING = 'late'
UNDERSAMPLE = False
PCA_N_COMPONENTS = 10
LANDMARK_PERCENTAGES = [10.0, 20.0]
N_SNAPSHOTS = 500               # historical snapshot count (500)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =============================================================================
# Load data
# =============================================================================
X, y, cfg = load_processed_features(DATASET_KEY)

# =============================================================================
# Customer split (labels only — this audit does not need PCA)
# =============================================================================
# Late split: the barcode-row hold-out happens after snapshots, so the
# audit uses the full class pool (after optional undersample).
splits = {"full": y}


def pool_counts(labels):
    n_pos = int((labels == cfg.positive_label).sum())
    n_neg = int((labels != cfg.positive_label).sum())
    if UNDERSAMPLE:
        n1 = n2 = min(n_pos, n_neg)
    else:
        n1, n2 = n_pos, n_neg
    return n_pos, n_neg, n1, n2

# =============================================================================
# Reuse-ratio audit
# =============================================================================
# R = (points_per_snapshot * n_snapshots) / class_count
# points_per_snapshot = floor(class_count * snapshot_size_percent / 100)
rows = []
payload = {
    "dataset": FOLDER,
    "protocol_bucket": PROTOCOL_BUCKET,
    "split_timing": SPLIT_TIMING,
    "undersample": UNDERSAMPLE,
    "n_snapshots_historical": N_SNAPSHOTS,
    "landmarks": {},
}
for split_name, labels in splits.items():
    raw_pos, raw_neg, n1, n2 = pool_counts(labels)
    payload[f"{split_name}_raw_n_pos"] = raw_pos
    payload[f"{split_name}_raw_n_neg"] = raw_neg
    payload[f"{split_name}_n1"] = n1
    payload[f"{split_name}_n2"] = n2
    for pct in LANDMARK_PERCENTAGES:
        for class_name, n_class in (("class1", n1), ("class2", n2)):
            points_per_snapshot = max(2, int(n_class * pct / 100.0))
            revised_n_snapshots = max(2, int(math.ceil(n_class / points_per_snapshot))) if points_per_snapshot else 1
            reuse_historical = (points_per_snapshot * N_SNAPSHOTS) / n_class if n_class else float("nan")
            reuse_revised = (points_per_snapshot * revised_n_snapshots) / n_class if n_class else float("nan")
            print(
                f"{split_name} {class_name} percent={pct:g}: "
                f"n={n_class}  points_per_snapshot={points_per_snapshot}  "
                f"R_historical={reuse_historical:.3f}  R_revised={reuse_revised:.3f}"
            )
            for rule, n_snap in (("historical_l500", N_SNAPSHOTS), ("revised_ceil_n_over_t", revised_n_snapshots)):
                audit = compute_sampling_ratio_audit(
                    n1=n1, n2=n2, t=points_per_snapshot, l=n_snap, landmark_percent=pct
                )
                audit.update(
                    {
                        "dataset": FOLDER,
                        "protocol_bucket": PROTOCOL_BUCKET,
                        "split": split_name,
                        "class": class_name,
                        "n_class": n_class,
                        "points_per_snapshot": points_per_snapshot,
                        "n_snapshots": n_snap,
                        "reuse_ratio": (points_per_snapshot * n_snap) / n_class if n_class else float("nan"),
                        "l_rule": rule,
                        "undersample": UNDERSAMPLE,
                    }
                )
                rows.append(audit)
                payload["landmarks"][f"{split_name}_L{pct:g}_{class_name}_{rule}"] = audit

# =============================================================================
# Store results
# =============================================================================
save_path = tda_results_dir(PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path.mkdir(parents=True, exist_ok=True)
frame = pd.DataFrame(rows)
frame.to_csv(save_path / "sampling_ratio_audit.csv", index=False)
store_results(path=str(save_path), save_name="sampling_ratio_audit", result_object=payload)
print(f"Saved {save_path / 'sampling_ratio_audit.csv'}")
