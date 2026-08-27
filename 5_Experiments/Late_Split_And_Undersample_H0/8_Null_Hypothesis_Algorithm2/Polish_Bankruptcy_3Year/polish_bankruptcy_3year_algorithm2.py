# -*- coding: utf-8 -*-
"""
Late split and undersample (the original historical run), using just H0 / 8_Null_Hypothesis_Algorithm2
Dataset: Polish Companies Bankruptcy (3 year)

This experiment does not run Ripser. It runs Robinson-Turner Algorithm 2 (permutation test) on barcode vectors.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings

import numpy as np
import pandas as pd

# This file lives four folders below the repository root (where utils.py is).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import (
    permutation_test_algorithm2,
    store_results,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

PROTOCOL_BUCKET = "Late_Split_And_Undersample_H0"
EXPERIMENT = "8_Null_Hypothesis_Algorithm2"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Polish_Bankruptcy_3Year"
MAX_PER_GROUP = 100
N_PERMUTATIONS = 200
PQ_PAIRS = ((2, 2), (1, 1), (2, 1))

src_dir = os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets", PROTOCOL_BUCKET, SOURCE_EXPERIMENT, FOLDER)
save_path = os.path.join(REPO_ROOT, "6_Results", PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
os.makedirs(save_path, exist_ok=True)

rows = []
payload = {}
rng = np.random.default_rng(42)

# =============================================================================
# Algorithm 2 - L10
# =============================================================================
path_L10 = os.path.join(src_dir, "data_L10.csv")
if os.path.exists(path_L10):
    table_path_L10 = pd.read_csv(path_L10)
    feats_path_L10 = [c for c in table_path_L10.columns if c != "label"]
    group1_path_L10 = table_path_L10[table_path_L10["label"] == 1][feats_path_L10].to_numpy()
    group2_path_L10 = table_path_L10[table_path_L10["label"] != 1][feats_path_L10].to_numpy()
    if len(group1_path_L10) > MAX_PER_GROUP:
        group1_path_L10 = group1_path_L10[rng.choice(len(group1_path_L10), MAX_PER_GROUP, replace=False)]
    if len(group2_path_L10) > MAX_PER_GROUP:
        group2_path_L10 = group2_path_L10[rng.choice(len(group2_path_L10), MAX_PER_GROUP, replace=False)]
    key_path_L10 = path_L10.replace(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets") + os.sep, "")
    payload[key_path_L10] = {"barcode_vector_proxy": True, "tests": {}}
    # The paper reports three (p, q) pairs: F_2,2 then F_1,1 then F_2,1.
    for p, q in PQ_PAIRS:
        result = permutation_test_algorithm2(
            group1_path_L10, group2_path_L10, n_permutations=N_PERMUTATIONS, p=p, q=q, random_state=42
        )
        payload[key_path_L10]["tests"][f"F_{p}_{q}"] = result
        rows.append({
            "source": key_path_L10,
            "protocol_bucket": PROTOCOL_BUCKET,
            "p": p,
            "q": q,
            "observed_F_pq": result["observed_F_pq"],
            "p_value": result["p_value"],
            "n1": result["n1"],
            "n2": result["n2"],
            "null_mean": result["null_mean"],
            "barcode_vector_proxy": True,
        })
        print(f"{os.path.basename(path_L10)} F_{p},{q}: observed={result['observed_F_pq']:.4f}, p={result['p_value']:.4f}")
else:
    print("Missing (run this arm's experiment 1 first):", path_L10)
# =============================================================================
# Algorithm 2 - L20
# =============================================================================
path_L20 = os.path.join(src_dir, "data_L20.csv")
if os.path.exists(path_L20):
    table_path_L20 = pd.read_csv(path_L20)
    feats_path_L20 = [c for c in table_path_L20.columns if c != "label"]
    group1_path_L20 = table_path_L20[table_path_L20["label"] == 1][feats_path_L20].to_numpy()
    group2_path_L20 = table_path_L20[table_path_L20["label"] != 1][feats_path_L20].to_numpy()
    if len(group1_path_L20) > MAX_PER_GROUP:
        group1_path_L20 = group1_path_L20[rng.choice(len(group1_path_L20), MAX_PER_GROUP, replace=False)]
    if len(group2_path_L20) > MAX_PER_GROUP:
        group2_path_L20 = group2_path_L20[rng.choice(len(group2_path_L20), MAX_PER_GROUP, replace=False)]
    key_path_L20 = path_L20.replace(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets") + os.sep, "")
    payload[key_path_L20] = {"barcode_vector_proxy": True, "tests": {}}
    # The paper reports three (p, q) pairs: F_2,2 then F_1,1 then F_2,1.
    for p, q in PQ_PAIRS:
        result = permutation_test_algorithm2(
            group1_path_L20, group2_path_L20, n_permutations=N_PERMUTATIONS, p=p, q=q, random_state=42
        )
        payload[key_path_L20]["tests"][f"F_{p}_{q}"] = result
        rows.append({
            "source": key_path_L20,
            "protocol_bucket": PROTOCOL_BUCKET,
            "p": p,
            "q": q,
            "observed_F_pq": result["observed_F_pq"],
            "p_value": result["p_value"],
            "n1": result["n1"],
            "n2": result["n2"],
            "null_mean": result["null_mean"],
            "barcode_vector_proxy": True,
        })
        print(f"{os.path.basename(path_L20)} F_{p},{q}: observed={result['observed_F_pq']:.4f}, p={result['p_value']:.4f}")
else:
    print("Missing (run this arm's experiment 1 first):", path_L20)
if not rows:
    raise FileNotFoundError("No Experiment 1 barcode files for Algorithm 2 on " + PROTOCOL_BUCKET + "/" + FOLDER)

pd.DataFrame(rows).to_csv(os.path.join(save_path, "algorithm2_permutation_results.csv"), index=False)
store_results(path=save_path, save_name="algorithm2_permutation_results", result_object=payload)
