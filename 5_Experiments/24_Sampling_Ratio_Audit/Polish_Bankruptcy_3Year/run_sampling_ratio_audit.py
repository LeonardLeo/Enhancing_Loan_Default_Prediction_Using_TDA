# -*- coding: utf-8 -*-
"""
Experiment 24 — Sampling Ratio Audit
Dataset: Polish Companies Bankruptcy (3-year)

Question
--------
When Experiment 3 draws `l` snapshots of `t` customers from a class of size
`n1`, how often is the same person reused?  The checklist target is

    R = (t * l) / n1  ≈  1  (or smaller).

R ≫ 1 means snapshots are heavily overlapping, so barcode rows are not
independent draws.

What this script does (in order)
--------------------------------
1. Load the processed table (same file Exp 3 starts from).
2. Count defaults vs non-defaults.
3. Set n1 = n2 = minority count (Exp 3 undersamples to this size).
4. For each landmark percent used in Exp 3, compute t = floor(n1 * L / 100).
5. Score two snapshot budgets:
     - historical l = 500  (what Exp 3 actually used)
     - revised     l = ceil(n1 / t)       (the value that makes R ≈ 1)
6. Write a CSV people can read without opening Python.

Results: 6_Results/24_Sampling_Ratio_Audit/Polish_Bankruptcy_3Year/
"""

# =============================================================================
# Import Libraries
# =============================================================================
import math
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import compute_sampling_ratio_audit, store_results

warnings.filterwarnings("ignore")

# =============================================================================
# Dataset settings (this folder only)
# =============================================================================
FOLDER = "Polish_Bankruptcy_3Year"
TARGET_COLUMN = "target"
POSITIVE_LABEL = 1
LANDMARK_PERCENTS = [10, 20]
HISTORICAL_L = 500  # snapshot count used in Experiment 3
SAVE_PATH = f"../../../6_Results/24_Sampling_Ratio_Audit/{FOLDER}"

# =============================================================================
# Stage 1 — Load processed data
# =============================================================================
dataset = pd.read_csv(os.path.abspath("../../../1_Data/Processed_Datasets/Polish_Bankruptcy_3Year/processed_data.csv"))
y = dataset[TARGET_COLUMN]
n_pos = int((y == POSITIVE_LABEL).sum())
n_neg = int((y != POSITIVE_LABEL).sum())
n1 = n2 = min(n_pos, n_neg)  # Exp 3 balances by undersampling the majority

print(f"Loaded {FOLDER}: {len(dataset)} rows")
print(f"  defaults={n_pos}, non-defaults={n_neg}, balanced n1=n2={n1}")

# =============================================================================
# Stage 2 — Audit each landmark percent
# =============================================================================
rows = []
payload = {
    "dataset": FOLDER,
    "raw_n_pos": n_pos,
    "raw_n_neg": n_neg,
    "balanced_n1": n1,
    "balanced_n2": n2,
    "landmarks": {},
}

for pct in LANDMARK_PERCENTS:
    t = max(3, int(n1 * pct / 100))
    revised_l = max(2, int(math.ceil(n1 / t))) if t else 1

    for label, l_value in (("historical_l500", HISTORICAL_L), ("revised_ceil_n1_over_t", revised_l)):
        audit = compute_sampling_ratio_audit(
            n1=n1, n2=n2, t=t, l=l_value, landmark_percent=pct
        )
        audit["dataset"] = FOLDER
        audit["l_rule"] = label
        rows.append(audit)
        payload["landmarks"][f"L{pct:g}_{label}"] = audit
        print(
            f"  L{pct:g} / {label}: t={t}, l={l_value}, "
            f"R=(t*l)/n1={audit['naive_tl_over_n1']:.2f}, "
            f"OK={audit['suggested_naive_near_or_below_1']}"
        )

# =============================================================================
# Stage 3 — Save
# =============================================================================
os.makedirs(os.path.abspath(SAVE_PATH), exist_ok=True)
pd.DataFrame(rows).to_csv(
    os.path.abspath(f"{SAVE_PATH}/sampling_ratio_audit.csv"), index=False
)
store_results(path=SAVE_PATH, save_name="sampling_ratio_audit", result_object=payload)
print(f"Saved {SAVE_PATH}/sampling_ratio_audit.csv")
