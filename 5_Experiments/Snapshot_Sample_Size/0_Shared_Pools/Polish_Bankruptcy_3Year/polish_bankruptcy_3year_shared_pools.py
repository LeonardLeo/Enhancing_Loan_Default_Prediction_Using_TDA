# -*- coding: utf-8 -*-
"""
Snapshot sample size — shared pool builder
Dataset: Polish Companies Bankruptcy (3 year)

Items 1, 2, and 4 are views of this one compute grid. This file builds the
shared 60-train / 15-test barcode pools. Open the item folders to see how
each figure consumes the pool.

Ripser helpers live in sample_size_lib.py; the steps are written here.
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
    export_all_experiment_tables,
    resolve_grid,
    results_shared_dir,
    reuse_ratio,
    save_json,
    surviving_points_per_snapshot,
)

# =============================================================================
# Protocol knobs
# =============================================================================
DATASET_KEY = 'polish_bankruptcy'
FOLDER = 'Polish_Bankruptcy_3Year'
PCA_N_COMPONENTS = 10              # Exp 3 rank
CANDIDATES = list(CANDIDATE_POINTS_PER_SNAPSHOT)   # 15, 30, 45, 60
N_SNAPSHOTS_TRAIN_POOL = N_TRAIN_POOL              # 60
N_SNAPSHOTS_TEST = N_TEST_SNAPSHOTS                # 15
NESTED_PREFIXES = (15, 30, 45, 60)                 # 15 subset 30 subset 45 subset 60
N_REPEATS_SNAPSHOT_DRAWS = N_REPEATS               # 10
CUSTOMER_SPLIT_RANDOM_STATE = CUSTOMER_SPLIT_SEED  # 0 (fixed; CI is not a new-customer CI)
SKIP_EXISTING = True
HOMOLOGY_DIM = 2

# Four protocol arms. This dataset is run on each of them.
#   Historical_Late_Split_Balanced_TDA     late split, undersample
#   Early_Split_TDA                        early split, undersample
#   No_Undersampling                       late split, no undersample
#   Early_Split_TDA_And_No_Undersampling   early split, no undersample
PROTOCOL_ARMS = list(PROTOCOLS.keys())

# =============================================================================
# For each protocol: split, PCA, choose points-per-snapshot, draw, Ripser, train
# =============================================================================
for protocol_bucket in PROTOCOL_ARMS:
    spec = PROTOCOLS[protocol_bucket]
    print("=" * 72)
    print(f"Shared pool  {FOLDER}  /  {protocol_bucket}")
    print(f"  split timing : {spec['split_timing']}")
    print(f"  undersample  : {spec['undersample']}")
    print(f"  PCA rank     : {PCA_N_COMPONENTS} (fit on train only if early, else full table)")
    print("=" * 72)

    # Customer split + PCA + optional undersample (protocol-honest class pools)
    design, pools = resolve_grid(DATASET_KEY, protocol_bucket)
    binding = design["binding_class_count"]
    grid = surviving_points_per_snapshot(binding, CANDIDATES)
    surviving = grid["surviving"]
    dropped = grid["dropped"]
    default_pps = grid["default_points_per_snapshot"]
    print(f"  binding class count = {binding}")
    print(f"  surviving points per snapshot = {surviving}  (dropped {[d['points_per_snapshot'] for d in dropped]})")
    print(f"  item-1 default points per snapshot = {default_pps} (largest surviving candidate)")
    print(
        f"  reuse at 60 snapshots = "
        f"{reuse_ratio(default_pps, N_SNAPSHOTS_TRAIN_POOL, pools['train_minority_count']):.3f}"
    )

    design_dir = results_shared_dir(protocol_bucket, FOLDER)
    save_json(design_dir / "design.json", {k: v for k, v in design.items()})

    # Draw 60 training snapshots once per (points_per_snapshot, repeat), Ripser each,
    # then reuse barcodes for nested prefixes 15 subset 30 subset 45 subset 60.
    # Fifteen test snapshots are drawn independently and held fixed.
    for points_per_snapshot in surviving:
        for repeat in range(N_REPEATS_SNAPSHOT_DRAWS):
            evaluate_nested_prefixes(
                DATASET_KEY,
                protocol_bucket,
                points_per_snapshot,
                repeat,
                pools=pools,
                skip_existing=SKIP_EXISTING,
            )

export_all_experiment_tables()
print("Shared pools written; item 1 / 2 / 4 tables exported.")
