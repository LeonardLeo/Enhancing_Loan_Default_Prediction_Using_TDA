# -*- coding: utf-8 -*-
"""
Snapshot sample size / 1_Snapshot_Count_Sweep
Dataset: Default of Credit Card Client

x-axis = number of snapshots {15, 30, 45, 60}. Points per snapshot is held
fixed at the dataset-aware default. This is item 1, not item 2 (which instead
holds 60 snapshots and moves points per snapshot).

I report F1 as the headline metric because several tables are class-imbalanced;
accuracy is shown as well. Items 1, 2, and 4 share one compute grid. The shared
pool builder for this dataset is:

    5_Experiments/Snapshot_Sample_Size/0_Shared_Pools/Default_Of_Credit_Card_Client_Data/default_of_credit_card_client_shared_pools.py
"""
# =============================================================================
# Import Libraries
# =============================================================================
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BUCKET = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BUCKET))

from sample_size_lib import (
    CANDIDATE_POINTS_PER_SNAPSHOT,
    CUSTOMER_SPLIT_SEED,
    N_REPEATS,
    N_TEST_SNAPSHOTS,
    N_TRAIN_POOL,
    PROTOCOLS,
    evaluate_nested_prefixes,
    export_experiment_tables,
    resolve_grid,
    reuse_ratio,
    surviving_points_per_snapshot,
)

# =============================================================================
# Protocol knobs
# =============================================================================
DATASET_KEY = 'credit_card_default'
FOLDER = 'Default_Of_Credit_Card_Client_Data'
ITEM = '1'
ITEM_FOLDER = '1_Snapshot_Count_Sweep'

PCA_N_COMPONENTS = 7
CANDIDATES = list(CANDIDATE_POINTS_PER_SNAPSHOT)   # 15, 30, 45, 60
N_SNAPSHOTS_TRAIN_POOL = N_TRAIN_POOL              # draw 60 training snapshots
N_SNAPSHOTS_TEST = N_TEST_SNAPSHOTS                # 15 held-out test snapshots
NESTED_PREFIXES = (15, 30, 45, 60)                 # 15 subset 30 subset 45 subset 60
N_REPEATS_SNAPSHOT_DRAWS = N_REPEATS               # 10
CUSTOMER_SPLIT_RANDOM_STATE = CUSTOMER_SPLIT_SEED  # 0
SKIP_EXISTING = True
CLASSIFIERS = ("svm", "knn", "xgb", "logistic", "random_forest")
# 95% CI = mean +/- 1.96 * SE across the 10 snapshot-draw repeats
# (snapshot-sampling uncertainty, not customer-split uncertainty).

PROTOCOL_ARMS = list(PROTOCOLS.keys())

# =============================================================================
# Consume / build the shared pool, then slice it for this figure
# =============================================================================
for protocol_bucket in PROTOCOL_ARMS:
    spec = PROTOCOLS[protocol_bucket]
    print("=" * 72)
    print(f"{ITEM_FOLDER}  /  {FOLDER}  /  {protocol_bucket}")
    print(f"  split timing : {spec['split_timing']}")
    print(f"  undersample  : {spec['undersample']}")
    print(f"  PCA rank     : {PCA_N_COMPONENTS}")
    print("=" * 72)

    # 1. Customer split (early arms) or full-table pool (late arms)
    # 2. PCA fit (train-only if early, full table if late)
    # 3. Optional undersample
    design, pools = resolve_grid(DATASET_KEY, protocol_bucket)
    grid = surviving_points_per_snapshot(design["binding_class_count"], CANDIDATES)
    surviving = grid["surviving"]
    dropped = [d["points_per_snapshot"] for d in grid["dropped"]]
    default_pps = grid["default_points_per_snapshot"]
    print(f"  surviving points per snapshot = {surviving}  dropped = {dropped}")
    print(f"  default points per snapshot   = {default_pps}")
    print(
        f"  reuse ratio at 60 snapshots   = "
        f"{reuse_ratio(default_pps, N_SNAPSHOTS_TRAIN_POOL, pools['train_minority_count']):.3f}"
    )

    # 4. Draw 60 snapshots, Ripser each, nested prefixes 15 subset 30 subset 45 subset 60
    # 5. Train five classifiers; record F1 and accuracy
    # 6. Repeat snapshot draws 10 times; CI is mean +/- 1.96 SE across repeats
    if ITEM == "1":
        pps_values = [default_pps]
    else:
        pps_values = surviving
    for points_per_snapshot in pps_values:
        for repeat in range(N_REPEATS_SNAPSHOT_DRAWS):
            evaluate_nested_prefixes(
                DATASET_KEY,
                protocol_bucket,
                points_per_snapshot,
                repeat,
                pools=pools,
                skip_existing=SKIP_EXISTING,
            )

# Item 1 keeps rows at the default points per snapshot.
# Item 2 keeps rows at 60 training snapshots.
# Item 4 keeps every surviving (points per snapshot, n snapshots) cell.
export_experiment_tables(ITEM)
print(f"Exported {ITEM_FOLDER} for {FOLDER}.")
