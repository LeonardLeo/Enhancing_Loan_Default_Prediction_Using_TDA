import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Experiment 26 — Intrinsic dimension estimation (b).

Estimators:
  - Two-NN (Facco et al.)
  - Levina–Bickel MLE

Applied to PCA-reduced processed features (same PCA dims as main TDA pipeline)
so we can compare b to the PCA component counts (7 for DCCCD, 15 for SGCD).
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils import (
    estimate_intrinsic_dimension_two_nn,
    estimate_intrinsic_dimension_levina_bickel,
    store_data_as_csv_or_json,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "6_Results" / "26_Intrinsic_Dimension_Estimation"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "name": "Default_Of_Credit_Card_Client_Data",
        "path": ROOT / "1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/processed_data.xlsx",
        "target": "default payment next month",
        "pca_components": 7,
        "max_points": 5000,
    },
    {
        "name": "Statlog_German_Credit_Data",
        "path": ROOT / "1_Data/Processed_Datasets/Statlog_German_Credit_Data/processed_data.xlsx",
        "target": "Class",
        "pca_components": 15,
        "max_points": None,
    },
]

rows = []
payload = {}

for cfg in DATASETS:
    df = pd.read_excel(cfg["path"])
    drop = [c for c in ["Unnamed: 0", cfg["target"]] if c in df.columns]
    X = df.drop(columns=drop)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Raw scaled space
    two_nn_raw = estimate_intrinsic_dimension_two_nn(
        X_scaled, n_samples=cfg["max_points"], random_state=42
    )
    lb_raw = estimate_intrinsic_dimension_levina_bickel(
        X_scaled, k=10, n_samples=cfg["max_points"], random_state=42
    )

    # PCA space used for diagrams
    pca = PCA(n_components=cfg["pca_components"], random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    two_nn_pca = estimate_intrinsic_dimension_two_nn(
        X_pca, n_samples=cfg["max_points"], random_state=42
    )
    lb_pca = estimate_intrinsic_dimension_levina_bickel(
        X_pca, k=10, n_samples=cfg["max_points"], random_state=42
    )

    entry = {
        "dataset": cfg["name"],
        "pca_components_used_in_TDA": cfg["pca_components"],
        "variance_retained_pca": float(pca.explained_variance_ratio_.sum()),
        "two_nn_raw": two_nn_raw,
        "levina_bickel_raw": lb_raw,
        "two_nn_pca": two_nn_pca,
        "levina_bickel_pca": lb_pca,
        "warning_if_b_near_7": (
            abs(two_nn_pca.get("intrinsic_dim_two_nn", 0) - 7) < 1.5
            or abs(lb_pca.get("intrinsic_dim_levina_bickel", 0) - 7) < 1.5
        ),
    }
    payload[cfg["name"]] = entry
    rows.append(
        {
            "dataset": cfg["name"],
            "two_nn_raw": two_nn_raw.get("intrinsic_dim_two_nn"),
            "levina_bickel_raw": lb_raw.get("intrinsic_dim_levina_bickel"),
            "two_nn_pca": two_nn_pca.get("intrinsic_dim_two_nn"),
            "levina_bickel_pca": lb_pca.get("intrinsic_dim_levina_bickel"),
            "pca_components": cfg["pca_components"],
            "variance_retained_pca": entry["variance_retained_pca"],
        }
    )
    print(
        f"{cfg['name']}: TwoNN(pca)={two_nn_pca.get('intrinsic_dim_two_nn'):.3f}, "
        f"LB(pca)={lb_pca.get('intrinsic_dim_levina_bickel'):.3f}"
    )

pd.DataFrame(rows).to_csv(OUT / "intrinsic_dimension_estimates.csv", index=False)
store_data_as_csv_or_json(
    path=str(OUT),
    csv=False,
    save_as=["intrinsic_dimension_estimates"],
    data_object=[payload],
)
print(f"\nSaved to {OUT}")
