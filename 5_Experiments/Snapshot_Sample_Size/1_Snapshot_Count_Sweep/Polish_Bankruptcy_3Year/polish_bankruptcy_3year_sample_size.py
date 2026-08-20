# -*- coding: utf-8 -*-
"""
Snapshot sample size / 1_Snapshot_Count_Sweep
Dataset: Polish Companies Bankruptcy (3 year)

This figure holds points per snapshot at the default cloud size (the largest surviving value in 15, 30, 45, 60). The x-axis is the number of snapshots: 15, 30, 45, 60.

The four protocol arms are written one after another in this file so you can
read load -> scale -> PCA -> class split -> snapshots -> Ripser -> train
without jumping into another module. Ripser on 60+15 snapshots (repeated 10
times) is the mechanical part and lives in utils.py, same as generate_landmark_sets
in the original PH scripts.

"""

# =============================================================================
# Import Libraries
# =============================================================================
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# This file lives four folders below the repository root (where utils.py is).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import (
    data_preprocessing_pipeline,
    CANDIDATE_POINTS_PER_SNAPSHOT,
    N_REPEATS,
    N_SNAPSHOTS_GRID,
    N_TEST_SNAPSHOTS,
    N_TRAIN_POOL,
    assemble_split_matrix,
    attach_design_columns,
    compute_barcodes_for_pool,
    draw_snapshot_pool,
    export_experiment_tables,
    pool_dir,
    repeat_metrics_path,
    train_on_prefix,
    write_repeat_metrics,
)

# =============================================================================
# Deal with Warnings
# =============================================================================
warnings.filterwarnings("ignore")

DATASET_KEY = "polish_bankruptcy"
FOLDER = "Polish_Bankruptcy_3Year"
ITEM = "1"
ITEM_FOLDER = "1_Snapshot_Count_Sweep"
PCA_N_COMPONENTS = 10
CANDIDATES = list(CANDIDATE_POINTS_PER_SNAPSHOT)   # 15, 30, 45, 60
N_SNAPSHOTS_TRAIN_POOL = N_TRAIN_POOL              # 60
N_SNAPSHOTS_TEST = N_TEST_SNAPSHOTS                # 15
NESTED_PREFIXES = tuple(N_SNAPSHOTS_GRID)          # 15 subset 30 subset 45 subset 60
N_REPEATS_SNAPSHOT_DRAWS = N_REPEATS               # 10
SKIP_EXISTING = True

# =============================================================================
# Get Dataset
# =============================================================================
data = pd.read_csv(os.path.join(REPO_ROOT, "1_Data", "Processed_Datasets", "Polish_Bankruptcy_3Year", "processed_data.csv"))
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
for col in data.columns:
    if col != "target" and data[col].isnull().any():
        data[col] = data[col].fillna(data[col].median())
data = data_preprocessing_pipeline(data)
X = data.drop(columns=["target"])
y = data["target"].astype(int)
X = X.select_dtypes(include=[np.number]).copy()

print("Loaded", FOLDER, "rows:", len(X), "PCA rank:", PCA_N_COMPONENTS)

# =============================================================================
# PROTOCOL: Historical Late Split, Balanced TDA
# split timing = late    undersample = True
# =============================================================================
print("=" * 72)
print("Historical Late Split, Balanced TDA")
print("=" * 72)
features = X.copy()
labels = y.copy()

scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
pca = PCA(n_components=10, random_state=42)
reduced = pd.DataFrame(
    pca.fit_transform(X_normalized),
    columns=[f"PCA_{i}" for i in range(1, 10 + 1)],
    index=features.index,
)
print("  variance retained (full-table PCA):", f"{pca.explained_variance_ratio_.sum():.2%}")

reduced["Class"] = labels.values
default_data = reduced[reduced["Class"] == 1].reset_index(drop=True)
non_default_data = reduced[reduced["Class"] == 0].reset_index(drop=True)
n_samples = min(len(default_data), len(non_default_data))
default_data = default_data.sample(n=n_samples, random_state=0).reset_index(drop=True)
non_default_data = non_default_data.sample(n=n_samples, random_state=0).reset_index(drop=True)
train_classes = {
    "default": default_data.drop(columns=["Class"]),
    "non-default": non_default_data.drop(columns=["Class"]),
}
test_classes = train_classes

print("  train default / non-default:", len(train_classes["default"]), len(train_classes["non-default"]))
print("  test  default / non-default:", len(test_classes["default"]), len(test_classes["non-default"]))
binding = min(len(train_classes['default']), len(train_classes['non-default']))
print("  binding class count (largest cloud that still fits a without-replacement draw):", binding)

# Drop any candidate in 15, 30, 45, 60 that cannot be drawn from the class pool.
surviving = []
for points_per_snapshot in CANDIDATES:
    if points_per_snapshot >= binding:
        print(f"    drop {points_per_snapshot} points per snapshot: class pool has only {binding} people")
    else:
        surviving.append(points_per_snapshot)
if not surviving:
    surviving = [max(5, binding - 1)]
    print("    documented clip: every candidate was too big, so use class count minus one =", surviving[0])
default_pps = max(surviving)
print("  surviving points per snapshot:", surviving)
print("  item-1 default points per snapshot (largest surviving):", default_pps)

protocol_bucket = "Historical_Late_Split_Balanced_TDA"
for points_per_snapshot in surviving:
    for repeat in range(N_REPEATS_SNAPSHOT_DRAWS):
        out_path = repeat_metrics_path(protocol_bucket, FOLDER, points_per_snapshot, repeat)
        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[skip] metrics {os.path.basename(str(out_path))}")
            continue

        # Draw 60 train snapshots + 15 test snapshots (no replacement inside a snapshot).
        index_sets, prefix_order, seeds = draw_snapshot_pool(
            train_classes=train_classes,
            test_classes=test_classes,
            protocol_bucket=protocol_bucket,
            dataset_folder=FOLDER,
            points_per_snapshot=points_per_snapshot,
            repeat=repeat,
            n_train_snapshots=N_SNAPSHOTS_TRAIN_POOL,
            n_test_snapshots=N_SNAPSHOTS_TEST,
            skip_existing=SKIP_EXISTING,
        )
        # Ripser once per snapshot. Nested prefixes 15 subset 30 subset 45 subset 60 reuse those barcodes.
        meta = compute_barcodes_for_pool(
            train_classes=train_classes,
            test_classes=test_classes,
            index_sets=index_sets,
            prefix_order=prefix_order,
            seeds=seeds,
            dataset_key=DATASET_KEY,
            protocol_bucket=protocol_bucket,
            dataset_folder=FOLDER,
            points_per_snapshot=points_per_snapshot,
            repeat=repeat,
            minority_count=min(len(train_classes["default"]), len(train_classes["non-default"])),
            majority_count=max(len(train_classes["default"]), len(train_classes["non-default"])),
            skip_existing=SKIP_EXISTING,
            n_train_snapshots=N_SNAPSHOTS_TRAIN_POOL,
            n_test_snapshots=N_SNAPSHOTS_TEST,
        )
        cache = pool_dir(protocol_bucket, FOLDER, points_per_snapshot, repeat)
        prefix_order = meta["nested_prefix_order"]
        test_df = assemble_split_matrix(cache, "test", list(range(N_SNAPSHOTS_TEST)))
        rows = []
        for n_snapshots in NESTED_PREFIXES:
            train_df = assemble_split_matrix(cache, "train", prefix_order, n_keep=n_snapshots)
            for metrics in train_on_prefix(train_df, test_df):
                rows.append(
                    attach_design_columns(
                        metrics,
                        dataset_key=DATASET_KEY,
                        display_name="Polish Companies Bankruptcy (3 year)",
                        folder_name=FOLDER,
                        protocol_bucket=protocol_bucket,
                        points_per_snapshot=points_per_snapshot,
                        n_snapshots=n_snapshots,
                        repeat=repeat,
                        minority_count=min(len(train_classes["default"]), len(train_classes["non-default"])),
                        majority_count=max(len(train_classes["default"]), len(train_classes["non-default"])),
                        binding=binding,
                        default_points_per_snapshot=default_pps,
                        n_train_barcode_rows=len(train_df),
                        n_test_barcode_rows=len(test_df),
                    )
                )
        write_repeat_metrics(out_path, rows, skip_existing=False)

# =============================================================================
# PROTOCOL: Early Split TDA
# split timing = early    undersample = True
# =============================================================================
print("=" * 72)
print("Early Split TDA")
print("=" * 72)
features = X.copy()
labels = y.copy()

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=0, stratify=labels
)
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
pca = PCA(n_components=10, random_state=42)
X_train_pca = pd.DataFrame(
    pca.fit_transform(X_train_scaled),
    columns=[f"PCA_{i}" for i in range(1, 10 + 1)],
    index=X_train.index,
)
X_test_pca = pd.DataFrame(
    pca.transform(X_test_scaled),
    columns=[f"PCA_{i}" for i in range(1, 10 + 1)],
    index=X_test.index,
)
print("  variance retained (train-fit PCA):", f"{pca.explained_variance_ratio_.sum():.2%}")

train_frame = X_train_pca.copy()
train_frame["Class"] = y_train.values
default_train = train_frame[train_frame["Class"] == 1].reset_index(drop=True)
non_default_train = train_frame[train_frame["Class"] == 0].reset_index(drop=True)
n_tr = min(len(default_train), len(non_default_train))
default_train = default_train.sample(n=n_tr, random_state=0).reset_index(drop=True)
non_default_train = non_default_train.sample(n=n_tr, random_state=0).reset_index(drop=True)

test_frame = X_test_pca.copy()
test_frame["Class"] = y_test.values
default_test = test_frame[test_frame["Class"] == 1].reset_index(drop=True)
non_default_test = test_frame[test_frame["Class"] == 0].reset_index(drop=True)
n_te = min(len(default_test), len(non_default_test))
default_test = default_test.sample(n=n_te, random_state=0).reset_index(drop=True)
non_default_test = non_default_test.sample(n=n_te, random_state=0).reset_index(drop=True)

train_classes = {
    "default": default_train.drop(columns=["Class"]),
    "non-default": non_default_train.drop(columns=["Class"]),
}
test_classes = {
    "default": default_test.drop(columns=["Class"]),
    "non-default": non_default_test.drop(columns=["Class"]),
}

print("  train default / non-default:", len(train_classes["default"]), len(train_classes["non-default"]))
print("  test  default / non-default:", len(test_classes["default"]), len(test_classes["non-default"]))
binding = min(len(train_classes['default']), len(train_classes['non-default']), len(test_classes['default']), len(test_classes['non-default']))
print("  binding class count (largest cloud that still fits a without-replacement draw):", binding)

# Drop any candidate in 15, 30, 45, 60 that cannot be drawn from the class pool.
surviving = []
for points_per_snapshot in CANDIDATES:
    if points_per_snapshot >= binding:
        print(f"    drop {points_per_snapshot} points per snapshot: class pool has only {binding} people")
    else:
        surviving.append(points_per_snapshot)
if not surviving:
    surviving = [max(5, binding - 1)]
    print("    documented clip: every candidate was too big, so use class count minus one =", surviving[0])
default_pps = max(surviving)
print("  surviving points per snapshot:", surviving)
print("  item-1 default points per snapshot (largest surviving):", default_pps)

protocol_bucket = "Early_Split_TDA"
for points_per_snapshot in surviving:
    for repeat in range(N_REPEATS_SNAPSHOT_DRAWS):
        out_path = repeat_metrics_path(protocol_bucket, FOLDER, points_per_snapshot, repeat)
        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[skip] metrics {os.path.basename(str(out_path))}")
            continue

        # Draw 60 train snapshots + 15 test snapshots (no replacement inside a snapshot).
        index_sets, prefix_order, seeds = draw_snapshot_pool(
            train_classes=train_classes,
            test_classes=test_classes,
            protocol_bucket=protocol_bucket,
            dataset_folder=FOLDER,
            points_per_snapshot=points_per_snapshot,
            repeat=repeat,
            n_train_snapshots=N_SNAPSHOTS_TRAIN_POOL,
            n_test_snapshots=N_SNAPSHOTS_TEST,
            skip_existing=SKIP_EXISTING,
        )
        # Ripser once per snapshot. Nested prefixes 15 subset 30 subset 45 subset 60 reuse those barcodes.
        meta = compute_barcodes_for_pool(
            train_classes=train_classes,
            test_classes=test_classes,
            index_sets=index_sets,
            prefix_order=prefix_order,
            seeds=seeds,
            dataset_key=DATASET_KEY,
            protocol_bucket=protocol_bucket,
            dataset_folder=FOLDER,
            points_per_snapshot=points_per_snapshot,
            repeat=repeat,
            minority_count=min(len(train_classes["default"]), len(train_classes["non-default"])),
            majority_count=max(len(train_classes["default"]), len(train_classes["non-default"])),
            skip_existing=SKIP_EXISTING,
            n_train_snapshots=N_SNAPSHOTS_TRAIN_POOL,
            n_test_snapshots=N_SNAPSHOTS_TEST,
        )
        cache = pool_dir(protocol_bucket, FOLDER, points_per_snapshot, repeat)
        prefix_order = meta["nested_prefix_order"]
        test_df = assemble_split_matrix(cache, "test", list(range(N_SNAPSHOTS_TEST)))
        rows = []
        for n_snapshots in NESTED_PREFIXES:
            train_df = assemble_split_matrix(cache, "train", prefix_order, n_keep=n_snapshots)
            for metrics in train_on_prefix(train_df, test_df):
                rows.append(
                    attach_design_columns(
                        metrics,
                        dataset_key=DATASET_KEY,
                        display_name="Polish Companies Bankruptcy (3 year)",
                        folder_name=FOLDER,
                        protocol_bucket=protocol_bucket,
                        points_per_snapshot=points_per_snapshot,
                        n_snapshots=n_snapshots,
                        repeat=repeat,
                        minority_count=min(len(train_classes["default"]), len(train_classes["non-default"])),
                        majority_count=max(len(train_classes["default"]), len(train_classes["non-default"])),
                        binding=binding,
                        default_points_per_snapshot=default_pps,
                        n_train_barcode_rows=len(train_df),
                        n_test_barcode_rows=len(test_df),
                    )
                )
        write_repeat_metrics(out_path, rows, skip_existing=False)

# =============================================================================
# PROTOCOL: No Undersampling
# split timing = late    undersample = False
# =============================================================================
print("=" * 72)
print("No Undersampling")
print("=" * 72)
features = X.copy()
labels = y.copy()

scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
pca = PCA(n_components=10, random_state=42)
reduced = pd.DataFrame(
    pca.fit_transform(X_normalized),
    columns=[f"PCA_{i}" for i in range(1, 10 + 1)],
    index=features.index,
)
print("  variance retained (full-table PCA):", f"{pca.explained_variance_ratio_.sum():.2%}")

reduced["Class"] = labels.values
default_data = reduced[reduced["Class"] == 1].reset_index(drop=True)
non_default_data = reduced[reduced["Class"] == 0].reset_index(drop=True)
train_classes = {
    "default": default_data.drop(columns=["Class"]),
    "non-default": non_default_data.drop(columns=["Class"]),
}
test_classes = train_classes

print("  train default / non-default:", len(train_classes["default"]), len(train_classes["non-default"]))
print("  test  default / non-default:", len(test_classes["default"]), len(test_classes["non-default"]))
binding = min(len(train_classes['default']), len(train_classes['non-default']))
print("  binding class count (largest cloud that still fits a without-replacement draw):", binding)

# Drop any candidate in 15, 30, 45, 60 that cannot be drawn from the class pool.
surviving = []
for points_per_snapshot in CANDIDATES:
    if points_per_snapshot >= binding:
        print(f"    drop {points_per_snapshot} points per snapshot: class pool has only {binding} people")
    else:
        surviving.append(points_per_snapshot)
if not surviving:
    surviving = [max(5, binding - 1)]
    print("    documented clip: every candidate was too big, so use class count minus one =", surviving[0])
default_pps = max(surviving)
print("  surviving points per snapshot:", surviving)
print("  item-1 default points per snapshot (largest surviving):", default_pps)

protocol_bucket = "No_Undersampling"
for points_per_snapshot in surviving:
    for repeat in range(N_REPEATS_SNAPSHOT_DRAWS):
        out_path = repeat_metrics_path(protocol_bucket, FOLDER, points_per_snapshot, repeat)
        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[skip] metrics {os.path.basename(str(out_path))}")
            continue

        # Draw 60 train snapshots + 15 test snapshots (no replacement inside a snapshot).
        index_sets, prefix_order, seeds = draw_snapshot_pool(
            train_classes=train_classes,
            test_classes=test_classes,
            protocol_bucket=protocol_bucket,
            dataset_folder=FOLDER,
            points_per_snapshot=points_per_snapshot,
            repeat=repeat,
            n_train_snapshots=N_SNAPSHOTS_TRAIN_POOL,
            n_test_snapshots=N_SNAPSHOTS_TEST,
            skip_existing=SKIP_EXISTING,
        )
        # Ripser once per snapshot. Nested prefixes 15 subset 30 subset 45 subset 60 reuse those barcodes.
        meta = compute_barcodes_for_pool(
            train_classes=train_classes,
            test_classes=test_classes,
            index_sets=index_sets,
            prefix_order=prefix_order,
            seeds=seeds,
            dataset_key=DATASET_KEY,
            protocol_bucket=protocol_bucket,
            dataset_folder=FOLDER,
            points_per_snapshot=points_per_snapshot,
            repeat=repeat,
            minority_count=min(len(train_classes["default"]), len(train_classes["non-default"])),
            majority_count=max(len(train_classes["default"]), len(train_classes["non-default"])),
            skip_existing=SKIP_EXISTING,
            n_train_snapshots=N_SNAPSHOTS_TRAIN_POOL,
            n_test_snapshots=N_SNAPSHOTS_TEST,
        )
        cache = pool_dir(protocol_bucket, FOLDER, points_per_snapshot, repeat)
        prefix_order = meta["nested_prefix_order"]
        test_df = assemble_split_matrix(cache, "test", list(range(N_SNAPSHOTS_TEST)))
        rows = []
        for n_snapshots in NESTED_PREFIXES:
            train_df = assemble_split_matrix(cache, "train", prefix_order, n_keep=n_snapshots)
            for metrics in train_on_prefix(train_df, test_df):
                rows.append(
                    attach_design_columns(
                        metrics,
                        dataset_key=DATASET_KEY,
                        display_name="Polish Companies Bankruptcy (3 year)",
                        folder_name=FOLDER,
                        protocol_bucket=protocol_bucket,
                        points_per_snapshot=points_per_snapshot,
                        n_snapshots=n_snapshots,
                        repeat=repeat,
                        minority_count=min(len(train_classes["default"]), len(train_classes["non-default"])),
                        majority_count=max(len(train_classes["default"]), len(train_classes["non-default"])),
                        binding=binding,
                        default_points_per_snapshot=default_pps,
                        n_train_barcode_rows=len(train_df),
                        n_test_barcode_rows=len(test_df),
                    )
                )
        write_repeat_metrics(out_path, rows, skip_existing=False)

# =============================================================================
# PROTOCOL: Early Split TDA And No Undersampling
# split timing = early    undersample = False
# =============================================================================
print("=" * 72)
print("Early Split TDA And No Undersampling")
print("=" * 72)
features = X.copy()
labels = y.copy()

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=0, stratify=labels
)
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
pca = PCA(n_components=10, random_state=42)
X_train_pca = pd.DataFrame(
    pca.fit_transform(X_train_scaled),
    columns=[f"PCA_{i}" for i in range(1, 10 + 1)],
    index=X_train.index,
)
X_test_pca = pd.DataFrame(
    pca.transform(X_test_scaled),
    columns=[f"PCA_{i}" for i in range(1, 10 + 1)],
    index=X_test.index,
)
print("  variance retained (train-fit PCA):", f"{pca.explained_variance_ratio_.sum():.2%}")

train_frame = X_train_pca.copy()
train_frame["Class"] = y_train.values
default_train = train_frame[train_frame["Class"] == 1].reset_index(drop=True)
non_default_train = train_frame[train_frame["Class"] == 0].reset_index(drop=True)
test_frame = X_test_pca.copy()
test_frame["Class"] = y_test.values
default_test = test_frame[test_frame["Class"] == 1].reset_index(drop=True)
non_default_test = test_frame[test_frame["Class"] == 0].reset_index(drop=True)

train_classes = {
    "default": default_train.drop(columns=["Class"]),
    "non-default": non_default_train.drop(columns=["Class"]),
}
test_classes = {
    "default": default_test.drop(columns=["Class"]),
    "non-default": non_default_test.drop(columns=["Class"]),
}

print("  train default / non-default:", len(train_classes["default"]), len(train_classes["non-default"]))
print("  test  default / non-default:", len(test_classes["default"]), len(test_classes["non-default"]))
binding = min(len(train_classes['default']), len(train_classes['non-default']), len(test_classes['default']), len(test_classes['non-default']))
print("  binding class count (largest cloud that still fits a without-replacement draw):", binding)

# Drop any candidate in 15, 30, 45, 60 that cannot be drawn from the class pool.
surviving = []
for points_per_snapshot in CANDIDATES:
    if points_per_snapshot >= binding:
        print(f"    drop {points_per_snapshot} points per snapshot: class pool has only {binding} people")
    else:
        surviving.append(points_per_snapshot)
if not surviving:
    surviving = [max(5, binding - 1)]
    print("    documented clip: every candidate was too big, so use class count minus one =", surviving[0])
default_pps = max(surviving)
print("  surviving points per snapshot:", surviving)
print("  item-1 default points per snapshot (largest surviving):", default_pps)

protocol_bucket = "Early_Split_TDA_And_No_Undersampling"
for points_per_snapshot in surviving:
    for repeat in range(N_REPEATS_SNAPSHOT_DRAWS):
        out_path = repeat_metrics_path(protocol_bucket, FOLDER, points_per_snapshot, repeat)
        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[skip] metrics {os.path.basename(str(out_path))}")
            continue

        # Draw 60 train snapshots + 15 test snapshots (no replacement inside a snapshot).
        index_sets, prefix_order, seeds = draw_snapshot_pool(
            train_classes=train_classes,
            test_classes=test_classes,
            protocol_bucket=protocol_bucket,
            dataset_folder=FOLDER,
            points_per_snapshot=points_per_snapshot,
            repeat=repeat,
            n_train_snapshots=N_SNAPSHOTS_TRAIN_POOL,
            n_test_snapshots=N_SNAPSHOTS_TEST,
            skip_existing=SKIP_EXISTING,
        )
        # Ripser once per snapshot. Nested prefixes 15 subset 30 subset 45 subset 60 reuse those barcodes.
        meta = compute_barcodes_for_pool(
            train_classes=train_classes,
            test_classes=test_classes,
            index_sets=index_sets,
            prefix_order=prefix_order,
            seeds=seeds,
            dataset_key=DATASET_KEY,
            protocol_bucket=protocol_bucket,
            dataset_folder=FOLDER,
            points_per_snapshot=points_per_snapshot,
            repeat=repeat,
            minority_count=min(len(train_classes["default"]), len(train_classes["non-default"])),
            majority_count=max(len(train_classes["default"]), len(train_classes["non-default"])),
            skip_existing=SKIP_EXISTING,
            n_train_snapshots=N_SNAPSHOTS_TRAIN_POOL,
            n_test_snapshots=N_SNAPSHOTS_TEST,
        )
        cache = pool_dir(protocol_bucket, FOLDER, points_per_snapshot, repeat)
        prefix_order = meta["nested_prefix_order"]
        test_df = assemble_split_matrix(cache, "test", list(range(N_SNAPSHOTS_TEST)))
        rows = []
        for n_snapshots in NESTED_PREFIXES:
            train_df = assemble_split_matrix(cache, "train", prefix_order, n_keep=n_snapshots)
            for metrics in train_on_prefix(train_df, test_df):
                rows.append(
                    attach_design_columns(
                        metrics,
                        dataset_key=DATASET_KEY,
                        display_name="Polish Companies Bankruptcy (3 year)",
                        folder_name=FOLDER,
                        protocol_bucket=protocol_bucket,
                        points_per_snapshot=points_per_snapshot,
                        n_snapshots=n_snapshots,
                        repeat=repeat,
                        minority_count=min(len(train_classes["default"]), len(train_classes["non-default"])),
                        majority_count=max(len(train_classes["default"]), len(train_classes["non-default"])),
                        binding=binding,
                        default_points_per_snapshot=default_pps,
                        n_train_barcode_rows=len(train_df),
                        n_test_barcode_rows=len(test_df),
                    )
                )
        write_repeat_metrics(out_path, rows, skip_existing=False)

# =============================================================================
# Keep the rows this figure needs, then write the CSV tables
# =============================================================================
export_experiment_tables("1")
print("Exported", ITEM_FOLDER, "for", FOLDER)
