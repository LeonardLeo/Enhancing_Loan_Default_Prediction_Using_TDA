import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Experiment 27 — Null hypothesis test (Algorithm 2, Robinson & Turner arXiv:1310.7467).

Tests whether default vs non-default barcode-statistic snapshots appear to arise
from the same process, using the joint loss F_{p,q} and a permutation p-value.

NOTE: We apply F_{p,q} to barcode-statistic *vectors* (Euclidean/Minkowski),
which is a computationally tractable proxy for pairwise persistence-diagram
distances. Results should be reported with that caveat; diagram-level
bottleneck/Wasserstein distances can be swapped in later using landmark CSVs.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils import permutation_test_algorithm2, store_data_as_csv_or_json

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "6_Results" / "27_Null_Hypothesis_Algorithm2"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ROOT / "1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L5.csv",
    ROOT / "1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L15.csv",
    ROOT / "1_Data/TDA_Datasets/Statlog_German_Credit_Data/3_PH_Default_Parameters/data_L30.csv",
    ROOT / "1_Data/TDA_Datasets/Statlog_German_Credit_Data/3_PH_Default_Parameters/data_L60.csv",
]

# Cap group size for permutation cost; stratified random subset if larger
MAX_PER_GROUP = 100
N_PERM = 200
PQ_SETTINGS = [(2, 2), (1, 1), (2, 1)]

rows = []
payload = {}

for path in SOURCES:
    if not path.exists():
        print(f"⚠️ Missing (skip): {path}")
        continue

    df = pd.read_csv(path)
    feats = [c for c in df.columns if c != "label"]
    g1 = df[df["label"] == 1][feats].values
    g2 = df[df["label"] == 0][feats].values

    rng = np.random.default_rng(42)
    if len(g1) > MAX_PER_GROUP:
        g1 = g1[rng.choice(len(g1), MAX_PER_GROUP, replace=False)]
    if len(g2) > MAX_PER_GROUP:
        g2 = g2[rng.choice(len(g2), MAX_PER_GROUP, replace=False)]

    key = f"{path.parent.parent.name}/{path.name}"
    payload[key] = {}
    for p, q in PQ_SETTINGS:
        result = permutation_test_algorithm2(
            g1, g2, n_permutations=N_PERM, p=p, q=q, random_state=42
        )
        payload[key][f"F_{p}_{q}"] = result
        rows.append(
            {
                "source": key,
                "p": p,
                "q": q,
                "observed_F_pq": result["observed_F_pq"],
                "p_value": result["p_value"],
                "n1": result["n1"],
                "n2": result["n2"],
                "null_mean": result["null_mean"],
            }
        )
        print(
            f"{key} F_{p},{q}: observed={result['observed_F_pq']:.4f}, "
            f"p={result['p_value']:.4f}"
        )

if rows:
    pd.DataFrame(rows).to_csv(OUT / "algorithm2_permutation_results.csv", index=False)

store_data_as_csv_or_json(
    path=str(OUT),
    csv=False,
    save_as=["algorithm2_permutation_results"],
    data_object=[payload],
)
print(f"\nSaved to {OUT}")
print(
    "Reference: Robinson & Turner, Hypothesis Testing for TDA, arXiv:1310.7467 "
    "(Algorithm 2; barcode-vector proxy for diagram distances)."
)
