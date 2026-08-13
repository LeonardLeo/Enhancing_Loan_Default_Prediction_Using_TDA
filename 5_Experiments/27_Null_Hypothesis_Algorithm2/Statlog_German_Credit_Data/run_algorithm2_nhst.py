# -*- coding: utf-8 -*-
"""
Experiment 27 — Null Hypothesis Test (Robinson & Turner Algorithm 2)
Dataset: Statlog German Credit (UCI)

Question
--------
Do default vs non-default barcode snapshots look like they come from the
*same* distribution?  Algorithm 2 (arXiv:1310.7467) builds a joint-loss
statistic F_{p,q} and a permutation p-value.

Limitation (please cite when publishing)
----------------------------------------
We apply F_{p,q} to 24-dimensional *barcode-statistic vectors*, not to
bottleneck / Wasserstein distances between full persistence diagrams.
Treat the p-values as a tractable proxy.

What this script does (in order)
--------------------------------
1. Load Experiment 3 matrices (data_L30.csv / data_L60.csv).
2. Split rows by label (1 = default, 0 = non-default).
3. Cap each group at 100 rows so the permutation loop stays affordable.
4. Run Algorithm 2 for (p,q) in {(2,2), (1,1), (2,1)}.
5. Write one CSV row per (file, p, q).

Prerequisite: Experiment 3 barcode CSVs.
Results: 6_Results/27_Null_Hypothesis_Algorithm2/Statlog_German_Credit_Data/
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import permutation_test_algorithm2, store_results

warnings.filterwarnings("ignore")

# =============================================================================
# Dataset settings (this folder only)
# =============================================================================
FOLDER = "Statlog_German_Credit_Data"
POSITIVE_LABEL = 1
SAVE_PATH = f"../../../6_Results/27_Null_Hypothesis_Algorithm2/{FOLDER}"
TDA_DIR = f"../../../1_Data/TDA_Datasets/{FOLDER}/3_PH_Default_Parameters"
SOURCES = [
    os.path.abspath(f"{TDA_DIR}/data_L30.csv"),
    os.path.abspath(f"{TDA_DIR}/data_L60.csv"),
]
MAX_PER_GROUP = 100
N_PERM = 200
PQ_SETTINGS = [(2, 2), (1, 1), (2, 1)]

# =============================================================================
# Stage 1 — Test each Exp 3 barcode matrix
# =============================================================================
rows = []
payload = {}
missing = []
rng = np.random.default_rng(42)

for path in SOURCES:
    if not os.path.exists(path):
        missing.append(path)
        print(f"Missing (run Experiment 3 first): {path}")
        continue

    df = pd.read_csv(path)
    feats = [c for c in df.columns if c != "label"]
    g1 = df[df["label"] == POSITIVE_LABEL][feats].to_numpy()
    g2 = df[df["label"] != POSITIVE_LABEL][feats].to_numpy()

    if len(g1) > MAX_PER_GROUP:
        g1 = g1[rng.choice(len(g1), MAX_PER_GROUP, replace=False)]
    if len(g2) > MAX_PER_GROUP:
        g2 = g2[rng.choice(len(g2), MAX_PER_GROUP, replace=False)]

    key = f"{FOLDER}/{os.path.basename(path)}"
    payload[key] = {"barcode_vector_proxy": True, "tests": {}}

    for p, q in PQ_SETTINGS:
        result = permutation_test_algorithm2(
            g1, g2, n_permutations=N_PERM, p=p, q=q, random_state=42
        )
        payload[key]["tests"][f"F_{p}_{q}"] = result
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
                "barcode_vector_proxy": True,
            }
        )
        print(
            f"{os.path.basename(path)} F_{p},{q}: "
            f"observed={result['observed_F_pq']:.4f}, p={result['p_value']:.4f}"
        )

# =============================================================================
# Stage 2 — Save
# =============================================================================
if not rows:
    print(
        f"\nNo Experiment 3 barcode files under {TDA_DIR}. "
        f"Run the matching Exp 3 script, then re-run this file."
    )
else:
    os.makedirs(os.path.abspath(SAVE_PATH), exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.abspath(f"{SAVE_PATH}/algorithm2_permutation_results.csv"), index=False
    )
    store_results(
        path=SAVE_PATH,
        save_name="algorithm2_permutation_results",
        result_object=payload,
    )
    print(
        "Reference: Robinson & Turner, arXiv:1310.7467 "
        "(Algorithm 2; barcode-vector proxy)."
    )
    if missing:
        print(f"Processed available files; {len(missing)} file(s) still missing.")
