# -*- coding: utf-8 -*-
"""
Early_Split_TDA_And_No_Undersampling / 8_Null_Hypothesis_Algorithm2
Dataset: South German Credit

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
and runs Robinson–Turner Algorithm 2 (permutation test) on the barcode vectors.
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import (
    _percent_token,
    get_dataset_config,
    permutation_test_algorithm2,
    store_results,
    tda_artefact_dir,
    tda_results_dir,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Protocol knobs (this arm, this dataset)
# =============================================================================
DATASET_KEY = 'south_german_credit'
PROTOCOL_BUCKET = 'Early_Split_TDA_And_No_Undersampling'
EXPERIMENT = "8_Null_Hypothesis_Algorithm2"
SOURCE_EXPERIMENT = "1_PH_Default_Parameters"
FOLDER = 'South_German_Credit'

SPLIT_TIMING = 'early'
UNDERSAMPLE = False
LANDMARK_PERCENTAGES = [10.0, 20.0]
MAX_PER_GROUP = 100
N_PERMUTATIONS = 200
RANDOM_STATE = 42
PQ_PAIRS = ((2, 2), (1, 1), (2, 1))

cfg = get_dataset_config(DATASET_KEY)

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
# Algorithm 2 permutation tests
# =============================================================================
rows = []
payload = {}
rng = np.random.default_rng(RANDOM_STATE)
for path in sources:
    if not path.exists():
        print(f"Missing (run this arm's experiment 1 first): {path}")
        continue
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c != "label"]
    g1 = df[df["label"] == cfg.positive_label][feats].to_numpy()
    g2 = df[df["label"] != cfg.positive_label][feats].to_numpy()
    if len(g1) > MAX_PER_GROUP:
        g1 = g1[rng.choice(len(g1), MAX_PER_GROUP, replace=False)]
    if len(g2) > MAX_PER_GROUP:
        g2 = g2[rng.choice(len(g2), MAX_PER_GROUP, replace=False)]
    rel = f"{path.parent.name}/{path.name}" if path.parent.name in {"train", "test"} else path.name
    key = f"{FOLDER}/{rel}"
    payload[key] = {"barcode_vector_proxy": True, "tests": {}}
    for p, q in PQ_PAIRS:
        result = permutation_test_algorithm2(
            g1, g2, n_permutations=N_PERMUTATIONS, p=p, q=q, random_state=RANDOM_STATE
        )
        payload[key]["tests"][f"F_{p}_{q}"] = result
        rows.append(
            {
                "source": key,
                "protocol_bucket": PROTOCOL_BUCKET,
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
        print(f"{rel} F_{p},{q}: observed={result['observed_F_pq']:.4f}, p={result['p_value']:.4f}")

if not rows:
    raise FileNotFoundError(
        f"No experiment-1 barcode files for Algorithm 2 on {PROTOCOL_BUCKET}/{FOLDER}."
    )

# =============================================================================
# Store results
# =============================================================================
save_path = tda_results_dir(PROTOCOL_BUCKET, EXPERIMENT, FOLDER)
save_path.mkdir(parents=True, exist_ok=True)
frame = pd.DataFrame(rows)
frame.to_csv(save_path / "algorithm2_permutation_results.csv", index=False)
store_results(path=str(save_path), save_name="algorithm2_permutation_results", result_object=payload)
