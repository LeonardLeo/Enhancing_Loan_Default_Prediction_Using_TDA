# -*- coding: utf-8 -*-
"""
Experiment 26 — Intrinsic Dimension Estimation
Dataset: Polish Companies Bankruptcy (3 year)

Question
--------
How many degrees of freedom does this point cloud really have?

We estimate intrinsic dimension **twice**:
  1. BEFORE PCA — geometry of the scaled credit/bankruptcy table.
  2. AFTER PCA  — geometry of the space Ripser actually samples in Exp 3.

PCA rank (10 components here) is *not* intrinsic dimension. Headline
estimator: Two-NN (Facco et al.). We also run Levina–Bickel (hand-coded)
and scikit-dimension (TwoNN, MLE, MiND_ML, lPCA) so a reviewer can see
the hand-coded formula agrees with the published package
(Bac et al., arXiv:2109.02596). dadapy is not added: it is the same
Two-NN estimator under another name.

What this script does (in order)
--------------------------------
1. Load processed tabular features; drop the target column.
2. Median-fill numeric holes; dummy-encode leftover categoricals.
3. MinMax-scale.
4. Estimate b on the scaled table (before PCA).
5. Fit the same PCA Exp 3 uses (10 components) and estimate b again.
6. Record how many components would be needed to keep ~90% variance
   (the design target for the four new tables).

This experiment does *not* need Experiment 3 artefacts.
Results: 6_Results/Statistics/1_Intrinsic_Dimension_Estimation/Polish_Bankruptcy_3Year/
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import estimate_intrinsic_dimension_suite, store_results

warnings.filterwarnings("ignore")

# =============================================================================
# Dataset settings (this folder only)
# =============================================================================
FOLDER = "Polish_Bankruptcy_3Year"
TARGET_COLUMN = "target"
PCA_COMPONENTS = 10  # Exp 3 ambient dimension for this table
MAX_POINTS = 5000
VARIANCE_TARGET = 0.90
SAVE_PATH = ROOT / "6_Results" / "Statistics" / "1_Intrinsic_Dimension_Estimation" / FOLDER
PROCESSED = ROOT / "1_Data" / "Processed_Datasets" / FOLDER / "processed_data.csv"

# =============================================================================
# Stage 1 — Load and keep numeric predictors
# =============================================================================
dataset = pd.read_csv(PROCESSED)
drop_cols = [c for c in ("Unnamed: 0", TARGET_COLUMN) if c in dataset.columns]
features = dataset.drop(columns=drop_cols)
for col in features.select_dtypes(include=[np.number]).columns:
    if features[col].isnull().any():
        features[col] = features[col].fillna(features[col].median())
if features.select_dtypes(include=["object"]).shape[1]:
    features = pd.get_dummies(features, drop_first=False)
features = features.select_dtypes(include=[np.number]).fillna(0)

print(f"Loaded {FOLDER}: {len(dataset)} rows, {features.shape[1]} numeric features")

# =============================================================================
# Stage 2 — Scale
# =============================================================================
X_scaled = MinMaxScaler().fit_transform(features)

# =============================================================================
# Stage 3 — ID before PCA, after Exp-3 PCA, and n for 90% variance
# =============================================================================
suite = estimate_intrinsic_dimension_suite(
    X_scaled,
    pca_components=PCA_COMPONENTS,
    n_samples=MAX_POINTS,
    random_state=42,
    variance_target=VARIANCE_TARGET,
)

var90 = suite["n_components_for_target_variance"]
two_nn_before = suite["before_pca"]["handcoded_two_nn"].get("intrinsic_dim_two_nn")
two_nn_after = suite["after_pca"]["handcoded_two_nn"].get("intrinsic_dim_two_nn")
levina_before = suite["before_pca"]["handcoded_levina_bickel"].get("intrinsic_dim_levina_bickel")
levina_after = suite["after_pca"]["handcoded_levina_bickel"].get("intrinsic_dim_levina_bickel")
skdim_before = suite["before_pca"]["skdim"].get("estimators", {})
skdim_after = suite["after_pca"]["skdim"].get("estimators", {})
row = {
    "dataset": FOLDER,
    "n_features": suite["n_features"],
    "pca_components_exp3": suite["pca_components_used_in_TDA"],
    "variance_retained_exp3_pca": suite["variance_retained_pca"],
    "n_components_for_90pct": var90["n_components"],
    "variance_at_90pct_n": var90["variance_at_n"],
    "two_nn_before_pca": two_nn_before,
    "two_nn_after_pca": two_nn_after,
    "levina_bickel_before_pca": levina_before,
    "levina_bickel_after_pca": levina_after,
    "skdim_TwoNN_before_pca": skdim_before.get("TwoNN") if isinstance(skdim_before.get("TwoNN"), float) else None,
    "skdim_TwoNN_after_pca": skdim_after.get("TwoNN") if isinstance(skdim_after.get("TwoNN"), float) else None,
    "skdim_MLE_before_pca": skdim_before.get("MLE_LevinaBickel") if isinstance(skdim_before.get("MLE_LevinaBickel"), float) else None,
    "skdim_MLE_after_pca": skdim_after.get("MLE_LevinaBickel") if isinstance(skdim_after.get("MLE_LevinaBickel"), float) else None,
    "skdim_MiND_ML_before_pca": skdim_before.get("MiND_ML") if isinstance(skdim_before.get("MiND_ML"), float) else None,
    "skdim_MiND_ML_after_pca": skdim_after.get("MiND_ML") if isinstance(skdim_after.get("MiND_ML"), float) else None,
    "skdim_lPCA_before_pca": skdim_before.get("lPCA") if isinstance(skdim_before.get("lPCA"), float) else None,
    "skdim_lPCA_after_pca": skdim_after.get("lPCA") if isinstance(skdim_after.get("lPCA"), float) else None,
}

print(
    f"{FOLDER}: TwoNN before PCA={row['two_nn_before_pca']:.3f}, "
    f"after Exp3 PCA({row['pca_components_exp3']})={row['two_nn_after_pca']:.3f}, "
    f"var kept={row['variance_retained_exp3_pca']:.1%}, "
    f"n for 90%={row['n_components_for_90pct']}"
)

# =============================================================================
# Stage 4 — Save
# =============================================================================
SAVE_PATH.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row]).to_csv(SAVE_PATH / "intrinsic_dimension_estimates.csv", index=False)
store_results(
    path=str(SAVE_PATH),
    save_name="intrinsic_dimension_estimates",
    result_object={"row": row, "suite": suite},
)
