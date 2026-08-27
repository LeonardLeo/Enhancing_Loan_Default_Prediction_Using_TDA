# -*- coding: utf-8 -*-
"""
Early split, no undersample, using just H0 / 8_Null_Hypothesis_Algorithm2
Dataset: Default of Credit Card Client

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

PROTOCOL_BUCKET = "Early_Split_No_Undersample_H0"
EXPERIMENT = "8_Null_Hypothesis_Algorithm2"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = "Default_Of_Credit_Card_Client_Data"
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
# Algorithm 2 - L5 TRAIN
# =============================================================================
path_train_L5 = os.path.join(src_dir, "train", "data_L5.csv")
if os.path.exists(path_train_L5):
    table_path_train_L5 = pd.read_csv(path_train_L5)
    feats_path_train_L5 = [c for c in table_path_train_L5.columns if c != "label"]
    group1_path_train_L5 = table_path_train_L5[table_path_train_L5["label"] == 1][feats_path_train_L5].to_numpy()
    group2_path_train_L5 = table_path_train_L5[table_path_train_L5["label"] != 1][feats_path_train_L5].to_numpy()
    if len(group1_path_train_L5) > MAX_PER_GROUP:
        group1_path_train_L5 = group1_path_train_L5[rng.choice(len(group1_path_train_L5), MAX_PER_GROUP, replace=False)]
    if len(group2_path_train_L5) > MAX_PER_GROUP:
        group2_path_train_L5 = group2_path_train_L5[rng.choice(len(group2_path_train_L5), MAX_PER_GROUP, replace=False)]
    key_path_train_L5 = path_train_L5.replace(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets") + os.sep, "")
    payload[key_path_train_L5] = {"barcode_vector_proxy": True, "tests": {}}
    # The paper reports three (p, q) pairs: F_2,2 then F_1,1 then F_2,1.
    for p, q in PQ_PAIRS:
        result = permutation_test_algorithm2(
            group1_path_train_L5, group2_path_train_L5, n_permutations=N_PERMUTATIONS, p=p, q=q, random_state=42
        )
        payload[key_path_train_L5]["tests"][f"F_{p}_{q}"] = result
        rows.append({
            "source": key_path_train_L5,
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
        print(f"{os.path.basename(path_train_L5)} F_{p},{q}: observed={result['observed_F_pq']:.4f}, p={result['p_value']:.4f}")
else:
    print("Missing (run this arm's experiment 1 first):", path_train_L5)
# =============================================================================
# Algorithm 2 - L5 TEST
# =============================================================================
path_test_L5 = os.path.join(src_dir, "test", "data_L5.csv")
if os.path.exists(path_test_L5):
    table_path_test_L5 = pd.read_csv(path_test_L5)
    feats_path_test_L5 = [c for c in table_path_test_L5.columns if c != "label"]
    group1_path_test_L5 = table_path_test_L5[table_path_test_L5["label"] == 1][feats_path_test_L5].to_numpy()
    group2_path_test_L5 = table_path_test_L5[table_path_test_L5["label"] != 1][feats_path_test_L5].to_numpy()
    if len(group1_path_test_L5) > MAX_PER_GROUP:
        group1_path_test_L5 = group1_path_test_L5[rng.choice(len(group1_path_test_L5), MAX_PER_GROUP, replace=False)]
    if len(group2_path_test_L5) > MAX_PER_GROUP:
        group2_path_test_L5 = group2_path_test_L5[rng.choice(len(group2_path_test_L5), MAX_PER_GROUP, replace=False)]
    key_path_test_L5 = path_test_L5.replace(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets") + os.sep, "")
    payload[key_path_test_L5] = {"barcode_vector_proxy": True, "tests": {}}
    # The paper reports three (p, q) pairs: F_2,2 then F_1,1 then F_2,1.
    for p, q in PQ_PAIRS:
        result = permutation_test_algorithm2(
            group1_path_test_L5, group2_path_test_L5, n_permutations=N_PERMUTATIONS, p=p, q=q, random_state=42
        )
        payload[key_path_test_L5]["tests"][f"F_{p}_{q}"] = result
        rows.append({
            "source": key_path_test_L5,
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
        print(f"{os.path.basename(path_test_L5)} F_{p},{q}: observed={result['observed_F_pq']:.4f}, p={result['p_value']:.4f}")
else:
    print("Missing (run this arm's experiment 1 first):", path_test_L5)
# =============================================================================
# Algorithm 2 - L15 TRAIN
# =============================================================================
path_train_L15 = os.path.join(src_dir, "train", "data_L15.csv")
if os.path.exists(path_train_L15):
    table_path_train_L15 = pd.read_csv(path_train_L15)
    feats_path_train_L15 = [c for c in table_path_train_L15.columns if c != "label"]
    group1_path_train_L15 = table_path_train_L15[table_path_train_L15["label"] == 1][feats_path_train_L15].to_numpy()
    group2_path_train_L15 = table_path_train_L15[table_path_train_L15["label"] != 1][feats_path_train_L15].to_numpy()
    if len(group1_path_train_L15) > MAX_PER_GROUP:
        group1_path_train_L15 = group1_path_train_L15[rng.choice(len(group1_path_train_L15), MAX_PER_GROUP, replace=False)]
    if len(group2_path_train_L15) > MAX_PER_GROUP:
        group2_path_train_L15 = group2_path_train_L15[rng.choice(len(group2_path_train_L15), MAX_PER_GROUP, replace=False)]
    key_path_train_L15 = path_train_L15.replace(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets") + os.sep, "")
    payload[key_path_train_L15] = {"barcode_vector_proxy": True, "tests": {}}
    # The paper reports three (p, q) pairs: F_2,2 then F_1,1 then F_2,1.
    for p, q in PQ_PAIRS:
        result = permutation_test_algorithm2(
            group1_path_train_L15, group2_path_train_L15, n_permutations=N_PERMUTATIONS, p=p, q=q, random_state=42
        )
        payload[key_path_train_L15]["tests"][f"F_{p}_{q}"] = result
        rows.append({
            "source": key_path_train_L15,
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
        print(f"{os.path.basename(path_train_L15)} F_{p},{q}: observed={result['observed_F_pq']:.4f}, p={result['p_value']:.4f}")
else:
    print("Missing (run this arm's experiment 1 first):", path_train_L15)
# =============================================================================
# Algorithm 2 - L15 TEST
# =============================================================================
path_test_L15 = os.path.join(src_dir, "test", "data_L15.csv")
if os.path.exists(path_test_L15):
    table_path_test_L15 = pd.read_csv(path_test_L15)
    feats_path_test_L15 = [c for c in table_path_test_L15.columns if c != "label"]
    group1_path_test_L15 = table_path_test_L15[table_path_test_L15["label"] == 1][feats_path_test_L15].to_numpy()
    group2_path_test_L15 = table_path_test_L15[table_path_test_L15["label"] != 1][feats_path_test_L15].to_numpy()
    if len(group1_path_test_L15) > MAX_PER_GROUP:
        group1_path_test_L15 = group1_path_test_L15[rng.choice(len(group1_path_test_L15), MAX_PER_GROUP, replace=False)]
    if len(group2_path_test_L15) > MAX_PER_GROUP:
        group2_path_test_L15 = group2_path_test_L15[rng.choice(len(group2_path_test_L15), MAX_PER_GROUP, replace=False)]
    key_path_test_L15 = path_test_L15.replace(os.path.join(REPO_ROOT, "1_Data", "TDA_Datasets") + os.sep, "")
    payload[key_path_test_L15] = {"barcode_vector_proxy": True, "tests": {}}
    # The paper reports three (p, q) pairs: F_2,2 then F_1,1 then F_2,1.
    for p, q in PQ_PAIRS:
        result = permutation_test_algorithm2(
            group1_path_test_L15, group2_path_test_L15, n_permutations=N_PERMUTATIONS, p=p, q=q, random_state=42
        )
        payload[key_path_test_L15]["tests"][f"F_{p}_{q}"] = result
        rows.append({
            "source": key_path_test_L15,
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
        print(f"{os.path.basename(path_test_L15)} F_{p},{q}: observed={result['observed_F_pq']:.4f}, p={result['p_value']:.4f}")
else:
    print("Missing (run this arm's experiment 1 first):", path_test_L15)
if not rows:
    raise FileNotFoundError("No Experiment 1 barcode files for Algorithm 2 on " + PROTOCOL_BUCKET + "/" + FOLDER)

pd.DataFrame(rows).to_csv(os.path.join(save_path, "algorithm2_permutation_results.csv"), index=False)
store_results(path=save_path, save_name="algorithm2_permutation_results", result_object=payload)
