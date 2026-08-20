# -*- coding: utf-8 -*-
"""
No Undersampling / 6_Sampling_Ratio_Audit
Dataset: Default of Credit Card Client

This experiment does not run Ripser. It audits reuse ratio = (points per snapshot x number of snapshots) / class count.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# This file lives four folders below the repository root (where utils.py is).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import (
    compute_sampling_ratio_audit,
    store_results,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

import math

PROTOCOL_BUCKET = "No_Undersampling"
EXPERIMENT = "6_Sampling_Ratio_Audit"
FOLDER = "Default_Of_Credit_Card_Client_Data"
LANDMARK_PERCENTAGES = [5.0, 15.0]
N_SNAPSHOTS = 500

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_excel(os.path.join(REPO_ROOT, "1_Data", "Processed_Datasets", "Default_Of_Credit_Card_Client_Data", "processed_data.xlsx"))
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
X = data.drop(columns=["default payment next month"])
y = data["default payment next month"].astype(int)

# =============================================================================
# Scale features (MinMax so every column sits in [0, 1])
# =============================================================================
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# =============================================================================
# PCA on the full table (late-split geometry)
# =============================================================================
pca = PCA(n_components=7, random_state=42)
X_reduced = pd.DataFrame(
    pca.fit_transform(X_normalized),
    columns=[f"PCA_{num}" for num in range(1, 7 + 1)],
    index=X.index,
)
print(f"Variance retained with PCA components: {pca.explained_variance_ratio_.sum():.2%}")

# =============================================================================
# Keep both classes at their original sizes (no undersampling)
# =============================================================================
reduced_data = X_reduced.copy()
reduced_data["Class"] = y.values
default_data = reduced_data[reduced_data["Class"] == 1].reset_index(drop=True)
non_default_data = reduced_data[reduced_data["Class"] == 0].reset_index(drop=True)

print("default rows:", len(default_data), " non-default rows:", len(non_default_data))
pools = {
    "default": default_data.drop(columns=["Class"]),
    "non-default": non_default_data.drop(columns=["Class"]),
}

n1 = len(default_data)
n2 = len(non_default_data)
splits = {"full_table": (n1, n2)}

# =============================================================================
# Reuse-ratio audit
# =============================================================================
# R = (points_per_snapshot * n_snapshots) / class_count
# points_per_snapshot = floor(class_count * snapshot_size_percent / 100)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
os.makedirs(save_path, exist_ok=True)
rows = []
payload = {
    "dataset": FOLDER,
    "protocol_bucket": PROTOCOL_BUCKET,
    "undersample": False,
    "n_snapshots_historical": N_SNAPSHOTS,
    "landmarks": {},
}
for split_name, (n1, n2) in splits.items():
    payload[f"{split_name}_n1"] = n1
    payload[f"{split_name}_n2"] = n2
    for pct in LANDMARK_PERCENTAGES:
        for class_name, n_class in (("class1", n1), ("class2", n2)):
            points_per_snapshot = max(2, int(n_class * pct / 100.0))
            revised_n_snapshots = max(2, int(math.ceil(n_class / points_per_snapshot))) if points_per_snapshot else 1
            for rule, n_snap in (("historical_l500", N_SNAPSHOTS), ("revised_ceil_n_over_t", revised_n_snapshots)):
                reuse = (points_per_snapshot * n_snap) / n_class if n_class else float("nan")
                print(f"{split_name} {class_name} percent={pct:g} n={n_class} points_per_snapshot={points_per_snapshot} n_snapshots={n_snap} R={reuse:.3f}")
                audit = compute_sampling_ratio_audit(
                    n1=n1, n2=n2, t=points_per_snapshot, l=n_snap, landmark_percent=pct
                )
                audit.update({
                    "dataset": FOLDER,
                    "protocol_bucket": PROTOCOL_BUCKET,
                    "split": split_name,
                    "class": class_name,
                    "n_class": n_class,
                    "points_per_snapshot": points_per_snapshot,
                    "n_snapshots": n_snap,
                    "reuse_ratio": reuse,
                    "l_rule": rule,
                    "undersample": False,
                })
                rows.append(audit)
                payload["landmarks"][f"{split_name}_L{pct:g}_{class_name}_{rule}"] = audit

frame = pd.DataFrame(rows)
frame.to_csv(os.path.join(save_path, "sampling_ratio_audit.csv"), index=False)
store_results(path=save_path, save_name="sampling_ratio_audit", result_object=payload)
print("Saved", os.path.join(save_path, "sampling_ratio_audit.csv"))
