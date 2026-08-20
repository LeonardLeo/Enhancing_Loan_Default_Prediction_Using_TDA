# -*- coding: utf-8 -*-
"""
Snapshot sample-size study (dated 13/08/2026).

Items 1, 2, and 4 share one compute grid but are different x-factor views:
  1_Snapshot_Count_Sweep          — x = number of snapshots; points per snapshot fixed
  2_Points_Per_Snapshot_Sweep     — x = points per snapshot; number of snapshots fixed at 60
  3_Snapshot_Count_Across_Cloud_Sizes — x = number of snapshots; one curve per cloud size
Item 3 is that study — not a third independent grid.

English identifiers (see docs/Notation.md):
  points_per_snapshot, n_snapshots, minority_count, majority_count, reuse_ratio

Compute:
  For each (dataset, protocol, points_per_snapshot, repeat) generate 60 training
  snapshots once, Ripser once per snapshot, then reuse barcodes for nested
  prefixes 15 ⊂ 30 ⊂ 45 ⊂ 60. Fifteen test snapshots are drawn independently
  and held fixed across the snapshot-count sweep. skip_existing on the barcode
  cache. Smaller datasets first.
"""
from __future__ import annotations

import json
import math
import os
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import (  # noqa: E402
    compute_barcode_statistics,
    dataset_pca_rank,
    get_dataset_config,
    load_processed_features,
)

# -----------------------------------------------------------------------------
# Locked design
# -----------------------------------------------------------------------------
BUCKET = "Snapshot_Sample_Size"
SHARED_EXPERIMENT = "0_Shared_Pools"
CANDIDATE_POINTS_PER_SNAPSHOT: Tuple[int, ...] = (15, 30, 45, 60)
N_SNAPSHOTS_GRID: Tuple[int, ...] = (15, 30, 45, 60)
N_TRAIN_POOL = 60
N_TEST_SNAPSHOTS = 15
N_REPEATS = 10
CUSTOMER_SPLIT_SEED = 0
PCA_RANDOM_STATE = 42  # matches Exp 3 / DatasetConfig geometry
MODEL_RANDOM_STATE = 0
HOMOLOGY_DIM = 2
Z_95 = 1.96

PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "Historical_Late_Split_Balanced_TDA": {
        "split_timing": "late",
        "undersample": True,
        "display": "Historical late split, balanced",
    },
    "Early_Split_TDA": {
        "split_timing": "early",
        "undersample": True,
        "display": "Early split, undersampled",
    },
    "No_Undersampling": {
        "split_timing": "late",
        "undersample": False,
        "display": "Late split, no undersampling",
    },
    "Early_Split_TDA_And_No_Undersampling": {
        "split_timing": "early",
        "undersample": False,
        "display": "Early split, no undersampling",
    },
}

DATASET_RUN_ORDER: Tuple[str, ...] = (
    "pkdd_czech",
    "south_german_credit",
    "statlog_german",
    "taiwan_bankruptcy",
    "polish_bankruptcy",
    "credit_card_default",
)

ITEM_FOLDERS = {
    "1": "1_Snapshot_Count_Sweep",
    "2": "2_Points_Per_Snapshot_Sweep",
    "4": "3_Snapshot_Count_Across_Cloud_Sizes",
}

CLASSIFIER_ORDER = ("svm", "logistic", "knn", "xgb", "random_forest")
CLASSIFIER_DISPLAY = {
    "svm": "SVM",
    "logistic": "Logistic Regression",
    "knn": "KNN",
    "xgb": "XGBoost",
    "random_forest": "Random Forest",
}
HIGHLIGHT_MODELS = {"svm", "logistic"}
# Okabe–Ito (Wong 2011), colourblind-safe. Yellow is replaced by dark gold so every
# hue holds on white. This mapping is locked for every Snapshot Sample Size figure.
# SVM / Logistic take the two strongest hues; the other three are full saturation.
MODEL_COLORS = {
    "svm": "#0072B2",  # blue
    "logistic": "#D55E00",  # vermillion
    "knn": "#009E73",  # bluish green
    "xgb": "#C78D00",  # dark gold
    "random_forest": "#CC79A7",  # reddish purple
}
MODEL_LINEWIDTH = {
    "svm": 2.4,
    "logistic": 2.4,
    "knn": 1.8,
    "xgb": 1.8,
    "random_forest": 1.8,
}
MODEL_MARKERS = {
    "svm": "o",
    "logistic": "o",
    "knn": "s",
    "xgb": "^",
    "random_forest": "D",
}
MODEL_MARKERSIZE = {
    "svm": 8.0,
    "logistic": 8.0,
    "knn": 6.6,
    "xgb": 7.0,
    "random_forest": 6.6,
}
MODEL_LINE_ALPHA = 1.0  # never fade the lines themselves (no alpha < 0.9)
CI_RIBBON_ALPHA = 0.28
# Families of cloud size (item 4) — same Okabe–Ito family, assigned by rank when a
# protocol drops a candidate or uses a documented clip (for example 14 on PKDD).
CLOUD_SIZE_COLOR_CYCLE = ("#0072B2", "#009E73", "#C78D00", "#CC79A7", "#D55E00")
CLOUD_SIZE_MARKER_CYCLE = ("o", "s", "^", "D", "v")

METRIC_DISPLAY = {"f1": "F1", "accuracy": "Accuracy"}
BARCODE_COLUMNS = [f"g{i}_{j}" for j in range(HOMOLOGY_DIM) for i in range(1, 13)]


def win_long_path(path) -> Path:
    raw = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not raw.startswith("\\\\?\\"):
        if raw.startswith("\\\\"):
            raw = "\\\\?\\UNC\\" + raw[2:]
        else:
            raw = "\\\\?\\" + raw
    return Path(raw)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: Any) -> Path:
    path = win_long_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)
    return path


def load_json(path: Path) -> Any:
    with win_long_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reuse_ratio(
    points_per_snapshot: int,
    n_snapshots: int,
    minority_count: int,
) -> float:
    """(points per snapshot × number of snapshots) / minority class count."""
    if minority_count <= 0:
        return float("nan")
    return float(points_per_snapshot * n_snapshots) / float(minority_count)


def snapshot_size_percent_of_class(points_per_snapshot: int, class_count: int) -> float:
    if class_count <= 0:
        return float("nan")
    return 100.0 * float(points_per_snapshot) / float(class_count)


# =============================================================================
# Protocol clouds (honest to each arm; customer split seed is locked at 0)
# =============================================================================
def undersample_xy(
    X: pd.DataFrame,
    y: pd.Series,
    positive_label: int = 1,
    random_state: int = CUSTOMER_SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.Series]:
    data = X.copy()
    data["__y__"] = pd.Series(y).values
    pos = data[data["__y__"] == positive_label]
    neg = data[data["__y__"] != positive_label]
    n = min(len(pos), len(neg))
    if n < 2:
        raise ValueError(f"Cannot undersample: pos={len(pos)} neg={len(neg)}")
    pos = pos.sample(n=n, random_state=random_state)
    neg = neg.sample(n=n, random_state=random_state)
    out = pd.concat([pos, neg], ignore_index=True)
    return out.drop(columns=["__y__"]), out["__y__"].astype(int)


def split_classes(
    X: pd.DataFrame,
    y: pd.Series,
    positive_label: int = 1,
) -> Dict[str, pd.DataFrame]:
    data = X.copy()
    data["__y__"] = pd.Series(y).values
    pos = data[data["__y__"] == positive_label].drop(columns=["__y__"]).reset_index(drop=True)
    neg = data[data["__y__"] != positive_label].drop(columns=["__y__"]).reset_index(drop=True)
    return {"default": pos, "non-default": neg}


def _add_missing_indicators(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    miss_tr = X_train.isna().astype(float)
    miss_te = X_test.isna().astype(float)
    miss_tr.columns = [f"miss_{c}" for c in X_train.columns]
    miss_te.columns = [f"miss_{c}" for c in X_test.columns]
    keep = [c for c in miss_tr.columns if miss_tr[c].sum() > 0]
    if not keep:
        return X_train, X_test
    return (
        pd.concat([X_train, miss_tr[keep]], axis=1),
        pd.concat([X_test, miss_te[keep]], axis=1),
    )


def early_split_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    test_size: float = 0.2,
    random_state: int = CUSTOMER_SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    Xte_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    Xtr_imp, Xte_imp = _add_missing_indicators(Xtr_imp, Xte_imp)
    scaler = MinMaxScaler()
    Xtr_s = scaler.fit_transform(Xtr_imp)
    Xte_s = scaler.transform(Xte_imp)
    n_comp = min(n_components, max(1, Xtr_s.shape[0] - 1), Xtr_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=PCA_RANDOM_STATE)
    cols = [f"PCA_{i}" for i in range(1, n_comp + 1)]
    Xtr_p = pd.DataFrame(pca.fit_transform(Xtr_s), columns=cols, index=X_train.index)
    Xte_p = pd.DataFrame(pca.transform(Xte_s), columns=cols, index=X_test.index)
    return Xtr_p, Xte_p, y_train, y_test, float(pca.explained_variance_ratio_.sum())


def late_full_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
) -> Tuple[pd.DataFrame, pd.Series, float]:
    """Full-table PCA (historical late-split arms). Snapshots are then split."""
    miss = X.isna().astype(float)
    miss.columns = [f"miss_{c}" for c in X.columns]
    keep = [c for c in miss.columns if miss[c].sum() > 0]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    if keep:
        X_imp = pd.concat([X_imp, miss[keep]], axis=1)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X_imp)
    n_comp = min(n_components, max(1, Xs.shape[0] - 1), Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=PCA_RANDOM_STATE)
    cols = [f"PCA_{i}" for i in range(1, n_comp + 1)]
    Xp = pd.DataFrame(pca.fit_transform(Xs), columns=cols, index=X.index)
    return Xp, y, float(pca.explained_variance_ratio_.sum())


def prepare_protocol_pools(
    dataset_key: str,
    protocol_bucket: str,
) -> Dict[str, Any]:
    """
    Build class pools that are honest to the arm.

    Early-split arms: split customers first (random_state=0), PCA on train only,
    optional undersample inside each split. Train snapshots come from train
    pools; test snapshots come from test pools.

    Late-split arms: optional undersample on the full table, full-table PCA,
    then snapshots are drawn from the full class pools and split at snapshot
    level (historical barcode-row split). Nested prefixes change only the
    training snapshot count; 15 test snapshots are held out.
    """
    if protocol_bucket not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol_bucket!r}")
    spec = PROTOCOLS[protocol_bucket]
    X, y, cfg = load_processed_features(dataset_key)
    pca_n = dataset_pca_rank(dataset_key)
    split_timing = spec["split_timing"]
    undersample = bool(spec["undersample"])
    positive = int(cfg.positive_label)

    if split_timing == "early":
        X_train, X_test, y_train, y_test, var = early_split_pca(
            X, y, n_components=pca_n, random_state=CUSTOMER_SPLIT_SEED
        )
        if undersample:
            X_train, y_train = undersample_xy(
                X_train, y_train, positive_label=positive, random_state=CUSTOMER_SPLIT_SEED
            )
            X_test, y_test = undersample_xy(
                X_test, y_test, positive_label=positive, random_state=CUSTOMER_SPLIT_SEED
            )
        train_classes = split_classes(X_train, y_train, positive_label=positive)
        test_classes = split_classes(X_test, y_test, positive_label=positive)
        pca_fit = "train_only"
        snapshot_split = "customer"
    else:
        if undersample:
            X, y = undersample_xy(
                X, y, positive_label=positive, random_state=CUSTOMER_SPLIT_SEED
            )
        Xp, y_full, var = late_full_pca(X, y, n_components=pca_n)
        full_classes = split_classes(Xp, y_full, positive_label=positive)
        train_classes = full_classes
        test_classes = full_classes
        pca_fit = "full_table"
        snapshot_split = "snapshot"
        y_train = y_full
        y_test = y_full

    train_pos = int(len(train_classes["default"]))
    train_neg = int(len(train_classes["non-default"]))
    test_pos = int(len(test_classes["default"]))
    test_neg = int(len(test_classes["non-default"]))
    train_min = int(min(train_pos, train_neg))
    train_maj = int(max(train_pos, train_neg))
    test_min = int(min(test_pos, test_neg))
    test_maj = int(max(test_pos, test_neg))
    if snapshot_split == "customer":
        binding = int(min(train_min, test_min))
        binding_note = (
            "Early split: a candidate is dropped unless it is strictly smaller "
            "than both the train and test class pools (without-replacement draw)."
        )
    else:
        binding = int(min(train_min, train_maj))
        binding_note = (
            "Late split: snapshots are drawn from the full class pools after "
            "full-table PCA (and optional undersample). The binding count is "
            "the smaller class on that pool."
        )

    return {
        "dataset_key": dataset_key,
        "display_name": cfg.display_name,
        "folder_name": cfg.folder_name,
        "protocol_bucket": protocol_bucket,
        "split_timing": split_timing,
        "undersample": undersample,
        "pca_rank": pca_n,
        "pca_fit": pca_fit,
        "pca_variance_retained": float(var),
        "snapshot_split": snapshot_split,
        "train_classes": train_classes,
        "test_classes": test_classes,
        "train_minority_count": train_min,
        "train_majority_count": train_maj,
        "test_minority_count": test_min,
        "test_majority_count": test_maj,
        "binding_class_count": binding,
        "binding_note": binding_note,
        "customer_split_random_state": CUSTOMER_SPLIT_SEED,
    }


def surviving_points_per_snapshot(
    binding_class_count: int,
    candidates: Sequence[int] = CANDIDATE_POINTS_PER_SNAPSHOT,
) -> Dict[str, Any]:
    """
    Drop any candidate with points_per_snapshot >= class count.

    No silent clipping. If every candidate is dropped, a single clipped value
    of (binding_class_count - 1) is added and flagged so captions can say so.
    """
    dropped = []
    kept: List[int] = []
    for value in candidates:
        if value >= binding_class_count:
            dropped.append(
                {
                    "points_per_snapshot": int(value),
                    "reason": (
                        f"cannot draw {value} points without replacement from a "
                        f"class pool of {binding_class_count}"
                    ),
                }
            )
        else:
            kept.append(int(value))
    clipped = False
    clip_note = None
    if not kept:
        clipped_value = max(5, int(binding_class_count) - 1)
        kept = [clipped_value]
        clipped = True
        clip_note = (
            f"Every candidate in {list(candidates)} was at least the class count "
            f"{binding_class_count}, so the grid uses a documented clipped value "
            f"{clipped_value} = class count minus one. This is not silent clipping."
        )
    return {
        "candidates": [int(v) for v in candidates],
        "surviving": kept,
        "dropped": dropped,
        "clipped": clipped,
        "clip_note": clip_note,
        "default_points_per_snapshot": max(kept),
        "default_rule": (
            "Largest surviving candidate in {15, 30, 45, 60} that is strictly "
            "smaller than the protocol's binding class pool. Historical Exp 3 "
            "cloud sizes (for example 331 on DCCCD) sit above this grid, so they "
            "are not used as the item-1 default. Item 4 already varies cloud size "
            "inside the surviving grid."
        ),
    }


def resolve_grid(dataset_key: str, protocol_bucket: str) -> Dict[str, Any]:
    pools = prepare_protocol_pools(dataset_key, protocol_bucket)
    grid = surviving_points_per_snapshot(pools["binding_class_count"])
    design = {
        "dataset_key": dataset_key,
        "display_name": pools["display_name"],
        "folder_name": pools["folder_name"],
        "protocol_bucket": protocol_bucket,
        "split_timing": pools["split_timing"],
        "undersample": pools["undersample"],
        "pca_rank": pools["pca_rank"],
        "pca_fit": pools["pca_fit"],
        "pca_variance_retained": pools["pca_variance_retained"],
        "snapshot_split": pools["snapshot_split"],
        "train_minority_count": pools["train_minority_count"],
        "train_majority_count": pools["train_majority_count"],
        "test_minority_count": pools["test_minority_count"],
        "test_majority_count": pools["test_majority_count"],
        "binding_class_count": pools["binding_class_count"],
        "binding_note": pools["binding_note"],
        "n_snapshots_grid": list(N_SNAPSHOTS_GRID),
        "n_train_pool": N_TRAIN_POOL,
        "n_test_snapshots": N_TEST_SNAPSHOTS,
        "n_repeats": N_REPEATS,
        "customer_split_random_state": CUSTOMER_SPLIT_SEED,
        **grid,
        "reuse_at_default": reuse_ratio(
            grid["default_points_per_snapshot"],
            N_TRAIN_POOL,
            pools["train_minority_count"],
        ),
        "snapshot_size_percent_of_minority_at_default": snapshot_size_percent_of_class(
            grid["default_points_per_snapshot"],
            pools["train_minority_count"],
        ),
    }
    return design, pools


# =============================================================================
# Snapshot draws + Ripser (generate 60 once; nested prefixes reuse barcodes)
# =============================================================================
def _draw_index_sets(
    n_pool: int,
    points_per_snapshot: int,
    n_snapshots: int,
    random_state: int,
) -> List[List[int]]:
    rng = check_random_state(random_state)
    if points_per_snapshot >= n_pool:
        raise ValueError(
            f"Cannot draw {points_per_snapshot} points without replacement "
            f"from a pool of {n_pool}"
        )
    sets: List[List[int]] = []
    for _ in range(n_snapshots):
        idx = rng.choice(n_pool, size=points_per_snapshot, replace=False)
        sets.append(sorted(int(i) for i in idx.tolist()))
    return sets


def _nested_prefix_order(n_pool: int, random_state: int) -> List[int]:
    rng = check_random_state(random_state)
    order = np.arange(n_pool)
    rng.shuffle(order)
    return [int(i) for i in order.tolist()]


def artefact_root(kind: str, protocol_bucket: str, dataset_folder: str) -> Path:
    return (
        REPO_ROOT
        / "1_Data"
        / kind
        / BUCKET
        / protocol_bucket
        / SHARED_EXPERIMENT
        / dataset_folder
    )


def results_shared_dir(protocol_bucket: str, dataset_folder: str) -> Path:
    return REPO_ROOT / "6_Results" / BUCKET / "shared" / protocol_bucket / dataset_folder


def experiment_results_dir(
    item_folder: str, protocol_bucket: str, dataset_folder: str
) -> Path:
    return REPO_ROOT / "6_Results" / BUCKET / item_folder / protocol_bucket / dataset_folder


def visualizations_dir(item_folder: str) -> Path:
    return REPO_ROOT / "6_Results" / BUCKET / item_folder / "Visualizations"


def _save_figure(fig, path: Path, dpi: int = 160) -> Path:
    """Write a PNG through the Windows long-path prefix when needed."""
    target = win_long_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(os.fspath(target), dpi=dpi, bbox_inches="tight")
    return Path(os.path.abspath(os.fspath(path)))


def pool_dir(
    protocol_bucket: str,
    dataset_folder: str,
    points_per_snapshot: int,
    repeat: int,
) -> Path:
    return (
        artefact_root("Barcode_Statistics", protocol_bucket, dataset_folder)
        / f"pps_{points_per_snapshot}"
        / f"repeat_{repeat:02d}"
    )


def landmark_dir(
    protocol_bucket: str,
    dataset_folder: str,
    points_per_snapshot: int,
    repeat: int,
) -> Path:
    return (
        artefact_root("Landmark_Sets", protocol_bucket, dataset_folder)
        / f"pps_{points_per_snapshot}"
        / f"repeat_{repeat:02d}"
    )


def tda_dir(
    protocol_bucket: str,
    dataset_folder: str,
    points_per_snapshot: int,
    repeat: int,
) -> Path:
    return (
        artefact_root("TDA_Datasets", protocol_bucket, dataset_folder)
        / f"pps_{points_per_snapshot}"
        / f"repeat_{repeat:02d}"
    )


def _barcode_cache_path(cache_root: Path, split: str, class_name: str, index: int) -> Path:
    return win_long_path(cache_root / split / class_name / f"snapshot_{index:03d}.csv")


def _points_to_barcode_row(points: np.ndarray, label: int) -> List[float]:
    from ripser import ripser

    dgms = ripser(np.asarray(points, dtype=float), maxdim=HOMOLOGY_DIM - 1)["dgms"]
    row: List[float] = []
    for dim in range(HOMOLOGY_DIM):
        row.extend(compute_barcode_statistics(dgms[dim]))
    row.append(int(label))
    return row


def _write_barcode_row(path: Path, row: Sequence[float]) -> None:
    path = win_long_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([list(row)], columns=BARCODE_COLUMNS + ["label"])
    frame.to_csv(path, index=False)


def _read_barcode_row(path: Path) -> pd.Series:
    frame = pd.read_csv(win_long_path(path))
    return frame.iloc[0]


def _assemble_split_matrix(
    cache_root: Path,
    split: str,
    index_order: Sequence[int],
    n_keep: Optional[int] = None,
) -> pd.DataFrame:
    keep = list(index_order) if n_keep is None else list(index_order[:n_keep])
    frames = []
    for class_name, label in (("default", 1), ("non-default", 0)):
        rows = []
        for idx in keep:
            path = _barcode_cache_path(cache_root, split, class_name, idx)
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(_read_barcode_row(path))
        part = pd.DataFrame(rows).reset_index(drop=True)
        part["label"] = label
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def ensure_barcode_pool(
    dataset_key: str,
    protocol_bucket: str,
    points_per_snapshot: int,
    repeat: int,
    pools: Optional[Dict[str, Any]] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Draw 60 train + 15 test snapshots and Ripser each cloud once.

    Nested snapshot-count values reuse these barcodes. skip_existing is
    per-snapshot so a killed run resumes.
    """
    if pools is None:
        _, pools = resolve_grid(dataset_key, protocol_bucket)
    folder = pools["folder_name"]
    cache = pool_dir(protocol_bucket, folder, points_per_snapshot, repeat)
    lm_root = landmark_dir(protocol_bucket, folder, points_per_snapshot, repeat)
    tda_root = tda_dir(protocol_bucket, folder, points_per_snapshot, repeat)
    meta_path = win_long_path(cache / "pool_meta.json")
    train_csv = win_long_path(tda_root / "train_pool.csv")
    test_csv = win_long_path(tda_root / "test_pool.csv")

    train_seed = 10_000 + 1_000 * repeat + int(points_per_snapshot)
    test_seed = 80_000 + 1_000 * repeat + int(points_per_snapshot)
    shuffle_seed = 50_000 + 1_000 * repeat + int(points_per_snapshot)
    index_path = win_long_path(lm_root / "index_sets.json")
    order_path = win_long_path(lm_root / "nested_prefix_order.json")

    if skip_existing and meta_path.exists() and train_csv.exists() and test_csv.exists():
        meta = load_json(meta_path)
        if int(meta.get("n_train_complete", 0)) >= N_TRAIN_POOL and int(
            meta.get("n_test_complete", 0)
        ) >= N_TEST_SNAPSHOTS:
            print(
                f"[skip] barcodes {protocol_bucket}/{folder} "
                f"pps={points_per_snapshot} repeat={repeat}"
            )
            return meta

    if skip_existing and index_path.exists() and order_path.exists():
        index_sets = load_json(index_path)
        prefix_order = load_json(order_path)
    else:
        index_sets = {"train": {}, "test": {}}
        for class_name, frame in pools["train_classes"].items():
            index_sets["train"][class_name] = _draw_index_sets(
                n_pool=len(frame),
                points_per_snapshot=points_per_snapshot,
                n_snapshots=N_TRAIN_POOL,
                random_state=train_seed + (0 if class_name == "default" else 1),
            )
        for class_name, frame in pools["test_classes"].items():
            index_sets["test"][class_name] = _draw_index_sets(
                n_pool=len(frame),
                points_per_snapshot=points_per_snapshot,
                n_snapshots=N_TEST_SNAPSHOTS,
                random_state=test_seed + (0 if class_name == "default" else 1),
            )
        prefix_order = _nested_prefix_order(N_TRAIN_POOL, random_state=shuffle_seed)
        save_json(index_path, index_sets)
        save_json(order_path, prefix_order)

    label_map = {"default": 1, "non-default": 0}
    n_train_done = 0
    n_test_done = 0
    t0 = time.time()
    for split, n_need, class_frames in (
        ("train", N_TRAIN_POOL, pools["train_classes"]),
        ("test", N_TEST_SNAPSHOTS, pools["test_classes"]),
    ):
        for class_name, frame in class_frames.items():
            values = frame.to_numpy(dtype=float)
            for i in range(n_need):
                out_path = _barcode_cache_path(cache, split, class_name, i)
                if skip_existing and out_path.exists():
                    if split == "train":
                        n_train_done += 1
                    else:
                        n_test_done += 1
                    continue
                idx = index_sets[split][class_name][i]
                points = values[idx]
                row = _points_to_barcode_row(points, label=label_map[class_name])
                _write_barcode_row(out_path, row)
                lm_path = win_long_path(
                    lm_root / split / class_name / f"landmarks_{i:03d}.csv"
                )
                lm_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(points, columns=list(frame.columns)).to_csv(
                    lm_path, index=False
                )
                if split == "train":
                    n_train_done += 1
                else:
                    n_test_done += 1
                if (n_train_done + n_test_done) % 20 == 0:
                    print(
                        f"  Ripser {protocol_bucket}/{folder} pps={points_per_snapshot} "
                        f"repeat={repeat}: train={n_train_done}/{N_TRAIN_POOL * 2} "
                        f"test={n_test_done}/{N_TEST_SNAPSHOTS * 2}"
                    )

    train_pool = _assemble_split_matrix(cache, "train", list(range(N_TRAIN_POOL)))
    test_pool = _assemble_split_matrix(cache, "test", list(range(N_TEST_SNAPSHOTS)))
    tda_root = win_long_path(tda_root)
    tda_root.mkdir(parents=True, exist_ok=True)
    train_pool.to_csv(train_csv, index=False)
    test_pool.to_csv(test_csv, index=False)
    save_json(win_long_path(lm_root / "index_sets.json"), index_sets)
    save_json(win_long_path(lm_root / "nested_prefix_order.json"), prefix_order)

    meta = {
        "dataset_key": dataset_key,
        "folder_name": folder,
        "protocol_bucket": protocol_bucket,
        "points_per_snapshot": int(points_per_snapshot),
        "repeat": int(repeat),
        "n_train_pool": N_TRAIN_POOL,
        "n_test_snapshots": N_TEST_SNAPSHOTS,
        "nested_prefix_order": prefix_order,
        "n_train_complete": N_TRAIN_POOL * 2,
        "n_test_complete": N_TEST_SNAPSHOTS * 2,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "shuffle_seed": shuffle_seed,
        "elapsed_seconds": round(time.time() - t0, 3),
        "train_pool_csv": str(train_csv),
        "test_pool_csv": str(test_csv),
        "reuse_ratio_at_60": reuse_ratio(
            points_per_snapshot, N_TRAIN_POOL, pools["train_minority_count"]
        ),
        "minority_count": pools["train_minority_count"],
        "majority_count": pools["train_majority_count"],
        "snapshot_size_percent_of_class": snapshot_size_percent_of_class(
            points_per_snapshot, pools["train_minority_count"]
        ),
    }
    save_json(meta_path, meta)
    print(
        f"[ok] barcodes {protocol_bucket}/{folder} pps={points_per_snapshot} "
        f"repeat={repeat} in {meta['elapsed_seconds']}s"
    )
    return meta


def _default_models() -> Dict[str, Any]:
    """Exp 1 TDA default hyperparameters (empty kwargs), seeds fixed for CI."""
    return {
        "svm": SVC(random_state=MODEL_RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "xgb": XGBClassifier(
            eval_metric="logloss",
            random_state=MODEL_RANDOM_STATE,
            verbosity=0,
        ),
        "logistic": LogisticRegression(max_iter=1000, random_state=MODEL_RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=MODEL_RANDOM_STATE),
    }


def fit_default_classifiers(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    feature_cols = [c for c in train_df.columns if c != "label"]
    X_tr = train_df[feature_cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy()
    X_te = test_df[feature_cols].to_numpy(dtype=float)
    y_te = test_df["label"].to_numpy()
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    rows = []
    for name, model in _default_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append(
            {
                "model": name,
                "model_display": CLASSIFIER_DISPLAY[name],
                "accuracy": float(accuracy_score(y_te, pred)),
                "precision": float(precision_score(y_te, pred, zero_division=0)),
                "recall": float(recall_score(y_te, pred, zero_division=0)),
                "f1": float(f1_score(y_te, pred, zero_division=0)),
            }
        )
    return rows


def evaluate_nested_prefixes(
    dataset_key: str,
    protocol_bucket: str,
    points_per_snapshot: int,
    repeat: int,
    pools: Optional[Dict[str, Any]] = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    if pools is None:
        design, pools = resolve_grid(dataset_key, protocol_bucket)
    else:
        design, _ = resolve_grid(dataset_key, protocol_bucket)
    folder = pools["folder_name"]
    out_dir = results_shared_dir(protocol_bucket, folder)
    out_path = win_long_path(
        out_dir / f"repeat_{repeat:02d}_pps_{points_per_snapshot}_metrics.csv"
    )
    if skip_existing and out_path.exists():
        print(f"[skip] metrics {out_path.name}")
        return pd.read_csv(out_path)

    meta = ensure_barcode_pool(
        dataset_key,
        protocol_bucket,
        points_per_snapshot,
        repeat,
        pools=pools,
        skip_existing=skip_existing,
    )
    cache = pool_dir(protocol_bucket, folder, points_per_snapshot, repeat)
    prefix_order = meta["nested_prefix_order"]
    test_df = _assemble_split_matrix(cache, "test", list(range(N_TEST_SNAPSHOTS)))
    rows = []
    for n_snapshots in N_SNAPSHOTS_GRID:
        train_df = _assemble_split_matrix(
            cache, "train", prefix_order, n_keep=n_snapshots
        )
        for metrics in fit_default_classifiers(train_df, test_df):
            metrics.update(
                {
                    "dataset_key": dataset_key,
                    "dataset_display": pools["display_name"],
                    "folder_name": folder,
                    "protocol": protocol_bucket,
                    "protocol_display": PROTOCOLS[protocol_bucket]["display"],
                    "points_per_snapshot": int(points_per_snapshot),
                    "n_snapshots": int(n_snapshots),
                    "repeat": int(repeat),
                    "minority_class_count": int(pools["train_minority_count"]),
                    "majority_class_count": int(pools["train_majority_count"]),
                    "binding_class_count": int(pools["binding_class_count"]),
                    "reuse_ratio": reuse_ratio(
                        points_per_snapshot,
                        n_snapshots,
                        pools["train_minority_count"],
                    ),
                    "snapshot_size_percent_of_class": snapshot_size_percent_of_class(
                        points_per_snapshot, pools["train_minority_count"]
                    ),
                    "is_default_points_per_snapshot": int(
                        points_per_snapshot == design["default_points_per_snapshot"]
                    ),
                    "n_train_barcode_rows": int(len(train_df)),
                    "n_test_barcode_rows": int(len(test_df)),
                    "customer_split_random_state": CUSTOMER_SPLIT_SEED,
                    "ci_source": "snapshot_sampling_not_customer_split",
                }
            )
            rows.append(metrics)
    frame = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def run_shared_grid(
    datasets: Optional[Sequence[str]] = None,
    protocols: Optional[Sequence[str]] = None,
    skip_existing: bool = True,
    write_exports: bool = True,
) -> pd.DataFrame:
    datasets = list(datasets or DATASET_RUN_ORDER)
    protocols = list(protocols or PROTOCOLS.keys())
    frames: List[pd.DataFrame] = []
    for dataset_key in datasets:
        for protocol_bucket in protocols:
            print(f"\n==== {protocol_bucket} / {dataset_key} ====")
            try:
                design, pools = resolve_grid(dataset_key, protocol_bucket)
            except Exception as exc:
                print(f"DESIGN FAILED {protocol_bucket}/{dataset_key}: {exc}")
                traceback.print_exc()
                continue
            folder = pools["folder_name"]
            design_dir = results_shared_dir(protocol_bucket, folder)
            save_json(design_dir / "design.json", {k: v for k, v in design.items()})
            pd.DataFrame(
                [
                    {
                        "points_per_snapshot": v,
                        "status": "surviving",
                        "reason": "",
                    }
                    for v in design["surviving"]
                ]
                + [
                    {
                        "points_per_snapshot": d["points_per_snapshot"],
                        "status": "dropped",
                        "reason": d["reason"],
                    }
                    for d in design["dropped"]
                ]
            ).to_csv(design_dir / "points_per_snapshot_grid.csv", index=False)
            print(
                f"  binding class count={design['binding_class_count']} "
                f"surviving={design['surviving']} "
                f"default={design['default_points_per_snapshot']}"
            )
            for pps in design["surviving"]:
                for repeat in range(N_REPEATS):
                    try:
                        frames.append(
                            evaluate_nested_prefixes(
                                dataset_key,
                                protocol_bucket,
                                pps,
                                repeat,
                                pools=pools,
                                skip_existing=skip_existing,
                            )
                        )
                    except Exception as exc:
                        print(
                            f"FAILED {protocol_bucket}/{dataset_key} "
                            f"pps={pps} repeat={repeat}: {exc}"
                        )
                        traceback.print_exc()
    if not frames:
        combined = load_all_repeat_metrics()
    else:
        combined = pd.concat(frames, ignore_index=True)
    agg_path = REPO_ROOT / "6_Results" / BUCKET / "shared" / "all_repeat_metrics.csv"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    # Always rebuild the aggregate from per-repeat files so a one-dataset
    # generate job cannot overwrite the full grid.
    rebuilt = load_all_repeat_metrics()
    if not rebuilt.empty:
        combined = rebuilt
    if not combined.empty:
        combined.to_csv(agg_path, index=False)
    if write_exports and not combined.empty:
        export_all_experiment_tables(combined)
    return combined


def _ci_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset_key",
        "dataset_display",
        "folder_name",
        "protocol",
        "protocol_display",
        "points_per_snapshot",
        "n_snapshots",
        "model",
        "model_display",
        "minority_class_count",
        "majority_class_count",
        "binding_class_count",
        "is_default_points_per_snapshot",
    ]
    rows = []
    grouped = frame.groupby(keys, dropna=False)
    for key, grp in grouped:
        record = dict(zip(keys, key))
        record["n_repeats"] = int(len(grp))
        record["reuse_ratio"] = float(grp["reuse_ratio"].iloc[0]) if len(grp) else float("nan")
        record["snapshot_size_percent_of_class"] = float(
            grp["snapshot_size_percent_of_class"].iloc[0]
        ) if len(grp) else float("nan")
        for metric in ("f1", "accuracy", "precision", "recall"):
            values = grp[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            se = std / math.sqrt(len(values)) if len(values) else float("nan")
            lo, hi = np.percentile(values, [2.5, 97.5]) if len(values) else (mean, mean)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_se"] = float(se)
            record[f"{metric}_ci95_low"] = mean - Z_95 * se
            record[f"{metric}_ci95_high"] = mean + Z_95 * se
            record[f"{metric}_percentile_low"] = float(lo)
            record[f"{metric}_percentile_high"] = float(hi)
        rows.append(record)
    return pd.DataFrame(rows)


def load_all_repeat_metrics() -> pd.DataFrame:
    """Load every per-repeat metrics CSV. Do not trust a partial aggregate."""
    parts = list(
        (REPO_ROOT / "6_Results" / BUCKET / "shared").glob(
            "*/*/repeat_*_pps_*_metrics.csv"
        )
    )
    if not parts:
        path = REPO_ROOT / "6_Results" / BUCKET / "shared" / "all_repeat_metrics.csv"
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if "is_default_points_per_snapshot" in frame.columns:
        frame["is_default_points_per_snapshot"] = (
            frame["is_default_points_per_snapshot"].astype(int)
        )
    return frame


def export_experiment_tables(item: str, frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if frame is None:
        frame = load_all_repeat_metrics()
    if frame.empty:
        return frame
    item_folder = ITEM_FOLDERS[item]
    # Three views of one grid, different x-factors:
    #   item 1 — n_snapshots moves; points_per_snapshot locked at the default
    #   item 2 — points_per_snapshot moves; n_snapshots locked at 60
    #   item 4 — full (points_per_snapshot, n_snapshots) family
    if item == "1":
        sliced = frame[frame["is_default_points_per_snapshot"].astype(int) == 1].copy()
    elif item == "2":
        sliced = frame[frame["n_snapshots"] == N_TRAIN_POOL].copy()
    else:
        sliced = frame.copy()
    summary = _ci_summary(sliced)
    for (protocol, folder), grp in sliced.groupby(["protocol", "folder_name"]):
        dest = experiment_results_dir(item_folder, protocol, folder)
        dest.mkdir(parents=True, exist_ok=True)
        grp.to_csv(dest / "repeat_metrics.csv", index=False)
        _ci_summary(grp).to_csv(dest / "summary.csv", index=False)
    root = REPO_ROOT / "6_Results" / BUCKET / item_folder
    root.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(root / "all_repeat_metrics.csv", index=False)
    summary.to_csv(root / "all_summary.csv", index=False)
    return summary


def export_all_experiment_tables(frame: Optional[pd.DataFrame] = None) -> None:
    if frame is None:
        frame = load_all_repeat_metrics()
    if frame.empty:
        return
    agg_path = REPO_ROOT / "6_Results" / BUCKET / "shared" / "all_repeat_metrics.csv"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(agg_path, index=False)
    for item in ("1", "2", "4"):
        export_experiment_tables(item, frame)


def write_master_design_table(
    datasets: Optional[Sequence[str]] = None,
    protocols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    datasets = list(datasets or DATASET_RUN_ORDER)
    protocols = list(protocols or PROTOCOLS.keys())
    rows = []
    for dataset_key in datasets:
        for protocol_bucket in protocols:
            design, _pools = resolve_grid(dataset_key, protocol_bucket)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_display": design["display_name"],
                    "folder_name": design["folder_name"],
                    "protocol": protocol_bucket,
                    "split_timing": design["split_timing"],
                    "undersample": design["undersample"],
                    "train_minority_count": design["train_minority_count"],
                    "train_majority_count": design["train_majority_count"],
                    "test_minority_count": design["test_minority_count"],
                    "test_majority_count": design["test_majority_count"],
                    "binding_class_count": design["binding_class_count"],
                    "surviving_points_per_snapshot": ",".join(
                        str(v) for v in design["surviving"]
                    ),
                    "dropped_points_per_snapshot": ",".join(
                        str(d["points_per_snapshot"]) for d in design["dropped"]
                    ),
                    "default_points_per_snapshot": design["default_points_per_snapshot"],
                    "clipped": design["clipped"],
                    "clip_note": design["clip_note"] or "",
                    "n_snapshots_grid": "15,30,45,60",
                    "n_test_snapshots": N_TEST_SNAPSHOTS,
                    "n_repeats": N_REPEATS,
                    "reuse_ratio_at_default_and_60": design["reuse_at_default"],
                    "snapshot_size_percent_of_class_at_default": design[
                        "snapshot_size_percent_of_minority_at_default"
                    ],
                }
            )
    frame = pd.DataFrame(rows)
    dest = REPO_ROOT / "6_Results" / BUCKET / "shared" / "dataset_aware_grid.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    save_json(dest.with_suffix(".json"), frame.to_dict(orient="records"))
    return frame


# =============================================================================
# Figures — English labels, methodology notes under every graph
# =============================================================================
ITEM_NOTES = {
    "1": (
        "Item 1 — number of snapshots on the x-axis; each cloud has the dataset-aware "
        "default number of points (largest surviving value in 15/30/45/60 that still fits "
        "this protocol's class pool; 60 on DCCCD). Points per snapshot is held fixed. "
        "This is not item 2, which instead holds 60 snapshots and moves points per snapshot. "
        "Item 3 is this sample-size study (items 1, 2, and 4 together), not a third "
        "independent grid.\n"
        "Headline is F1 because several tables are imbalanced, especially with no "
        "undersampling. Accuracy is plotted as the secondary metric.\n"
        "Nested prefixes: 15 ⊂ 30 ⊂ 45 ⊂ 60 from a shuffled pool of 60 training snapshots. "
        "Ten repeats redraw that pool. The customer train/test split is fixed "
        "(random_state=0). The combined overlay is the mean trend across those 10 repeats "
        "(five models, no error bars). 95% intervals (mean ± 1.96×SE) live on the companion "
        "per-model CI panels as ribbons. A 2.5–97.5 percentile interval is stored in the "
        "summary CSV. This study does not also run five customer splits on the full grid."
    ),
    "2": (
        "Item 2 — points per snapshot on the x-axis; always 60 snapshots. Number of "
        "snapshots is held fixed. This is not item 1, which instead holds points per "
        "snapshot at the dataset-aware default and moves the number of snapshots. "
        "A universal 15/30/45/60 cloud-size grid is not used: PKDD's class pool cannot "
        "host the larger steps that DCCCD can. Candidates with points per snapshot "
        "≥ class count are dropped (no silent clipping). Item 3 is this sample-size "
        "study (items 1, 2, and 4 together), not a third independent grid.\n"
        "Headline is F1; accuracy is always shown. Nested prefixes and 10 snapshot-draw "
        "repeats match item 1. Customer split is fixed (random_state=0). The combined "
        "overlay is the mean trend across 10 repeats (five models, no error bars). "
        "95% intervals live on the companion per-model CI panels as ribbons."
    ),
    "4": (
        "Item 4 — number of snapshots on the x-axis; one curve per surviving "
        "points-per-snapshot value (families of cloud size). SVM and Logistic Regression "
        "are the focus overlay; KNN, XGBoost, and Random Forest are the same colour and "
        "marker language, not washed out. Item 3 is this sample-size study (items 1, 2, "
        "and 4 together), not a third independent grid.\n"
        "Nested prefixes: 15 ⊂ 30 ⊂ 45 ⊂ 60 from one pool of 60 training snapshots per "
        "repeat. Ten repeats. Customer split fixed (random_state=0). Combined overlays "
        "are mean trends only (no error bars). Companion CI panels use a single ribbon "
        "per (model, points-per-snapshot) cell (mean ± 1.96×SE across snapshot draws). "
        "This CI is snapshot-sampling uncertainty, not customer-split uncertainty."
    ),
}


def _wrap_note(text: str, width: int = 148) -> str:
    """Pre-wrap methodology notes. matplotlib wrap=True is extremely slow with tight bbox."""
    return "\n".join(textwrap.fill(part, width=width) for part in text.split("\n"))


def _footnote(ax, text: str) -> None:
    ax.figure.text(
        0.02,
        0.01,
        _wrap_note(text),
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#2d3748",
    )


def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.06)


def _model_legend_handle(model: str) -> Line2D:
    """Colour + marker only for a stable legend."""
    return Line2D(
        [0],
        [0],
        color=MODEL_COLORS[model],
        linewidth=MODEL_LINEWIDTH[model],
        linestyle="-",
        marker=MODEL_MARKERS[model],
        markersize=MODEL_MARKERSIZE[model],
        markerfacecolor=MODEL_COLORS[model],
        markeredgecolor=MODEL_COLORS[model],
        markeredgewidth=0.6,
        alpha=MODEL_LINE_ALPHA,
        label=CLASSIFIER_DISPLAY[model],
    )


def _legend_below(ax, ncol: int = 5, bbox_y: float = -0.18) -> None:
    handles = [_model_legend_handle(model) for model in CLASSIFIER_ORDER]
    ax.legend(
        handles=handles,
        labels=[CLASSIFIER_DISPLAY[m] for m in CLASSIFIER_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, bbox_y),
        ncol=ncol,
        frameon=False,
        fontsize=9,
        handlelength=2.6,
        columnspacing=1.6,
        borderaxespad=0.0,
    )


def _cloud_size_style(pps_values: Sequence[float]) -> Dict[int, Dict[str, Any]]:
    ordered = [int(v) for v in pps_values]
    styles: Dict[int, Dict[str, Any]] = {}
    for i, pps in enumerate(ordered):
        styles[pps] = {
            "color": CLOUD_SIZE_COLOR_CYCLE[i % len(CLOUD_SIZE_COLOR_CYCLE)],
            "marker": CLOUD_SIZE_MARKER_CYCLE[i % len(CLOUD_SIZE_MARKER_CYCLE)],
        }
    return styles


def _cloud_size_legend_handles(pps_values: Sequence[float]) -> List[Line2D]:
    styles = _cloud_size_style(pps_values)
    handles = []
    for pps in pps_values:
        spec = styles[int(pps)]
        handles.append(
            Line2D(
                [0],
                [0],
                color=spec["color"],
                linewidth=2.0,
                linestyle="-",
                marker=spec["marker"],
                markersize=7.0,
                markerfacecolor=spec["color"],
                markeredgecolor=spec["color"],
                label=f"{int(pps)} points per snapshot",
            )
        )
    return handles


def _plot_mean_trend(
    ax,
    summary: pd.DataFrame,
    x_col: str,
    metric: str,
    model: str,
    *,
    label: Optional[str] = None,
) -> None:
    """Mean trend line only. Combined overlays never draw error bars."""
    sub = summary[summary["model"] == model].sort_values(x_col)
    if sub.empty:
        return
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[f"{metric}_mean"].to_numpy(dtype=float)
    highlight = model in HIGHLIGHT_MODELS
    ax.plot(
        x,
        y,
        color=MODEL_COLORS[model],
        linewidth=MODEL_LINEWIDTH[model],
        linestyle="-",
        marker=MODEL_MARKERS[model],
        markersize=MODEL_MARKERSIZE[model],
        markerfacecolor=MODEL_COLORS[model],
        markeredgecolor=MODEL_COLORS[model],
        markeredgewidth=0.6,
        alpha=MODEL_LINE_ALPHA,
        label=label or CLASSIFIER_DISPLAY[model],
        zorder=3 if highlight else 2,
    )


def _plot_mean_ribbon(
    ax,
    summary: pd.DataFrame,
    x_col: str,
    metric: str,
    model: str,
    *,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    lw: Optional[float] = None,
    label: Optional[str] = None,
) -> None:
    """Single-series ribbon. Use only when one CI is drawn on the axes."""
    sub = summary[summary["model"] == model].sort_values(x_col)
    if sub.empty:
        return
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[f"{metric}_mean"].to_numpy(dtype=float)
    lo = sub[f"{metric}_ci95_low"].to_numpy(dtype=float)
    hi = sub[f"{metric}_ci95_high"].to_numpy(dtype=float)
    color = color or MODEL_COLORS[model]
    marker = marker or MODEL_MARKERS[model]
    lw = MODEL_LINEWIDTH[model] if lw is None else float(lw)
    ax.fill_between(x, lo, hi, color=color, alpha=CI_RIBBON_ALPHA, linewidth=0, zorder=1)
    ax.plot(
        x,
        y,
        color=color,
        lw=lw,
        alpha=MODEL_LINE_ALPHA,
        marker=marker,
        markersize=MODEL_MARKERSIZE[model],
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.6,
        label=label or CLASSIFIER_DISPLAY[model],
        zorder=3,
    )


def _plot_pps_trend(
    ax,
    frame: pd.DataFrame,
    metric: str,
    model: str,
    pps: int,
    *,
    color: str,
    marker: str,
    label: Optional[str] = None,
) -> None:
    """Mean trend for one cloud size. Combined overlays never draw error bars."""
    sub = frame[
        (frame["model"] == model) & (frame["points_per_snapshot"] == pps)
    ].sort_values("n_snapshots")
    if sub.empty:
        return
    x = sub["n_snapshots"].to_numpy(dtype=float)
    y = sub[f"{metric}_mean"].to_numpy(dtype=float)
    highlight = model in HIGHLIGHT_MODELS
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.4 if highlight else 1.8,
        linestyle="-",
        marker=marker,
        markersize=8.0 if highlight else 6.6,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.6,
        alpha=MODEL_LINE_ALPHA,
        label=label,
        zorder=3 if highlight else 2,
    )


def _as_axes_grid(axes, n_rows: int, n_cols: int) -> np.ndarray:
    grid = np.array(axes, dtype=object)
    if n_rows == 1 and n_cols == 1:
        return np.array([[grid]], dtype=object)
    if n_rows == 1:
        return grid.reshape(1, n_cols)
    if n_cols == 1:
        return grid.reshape(n_rows, 1)
    return grid.reshape(n_rows, n_cols)


def _plot_ci_panels(
    summary: pd.DataFrame,
    viz: Path,
    *,
    x_col: str,
    metric: str,
    path: Path,
    xlabel: str,
    title: str,
    note: str,
    xticks: Optional[Sequence[float]] = None,
) -> Path:
    fig, axes = plt.subplots(1, len(CLASSIFIER_ORDER), figsize=(15.4, 4.9), sharey=True)
    fig.subplots_adjust(bottom=0.36, wspace=0.16, top=0.80, left=0.06, right=0.99)
    for i, model in enumerate(CLASSIFIER_ORDER):
        ax = axes[i]
        _plot_mean_ribbon(ax, summary, x_col, metric, model)
        _style_axes(
            ax,
            xlabel=xlabel,
            ylabel=METRIC_DISPLAY[metric] if i == 0 else "",
            title=CLASSIFIER_DISPLAY[model],
        )
        if xticks is not None:
            ax.set_xticks(list(xticks))
        else:
            vals = sorted(summary[x_col].unique())
            ax.set_xticks(vals)
    fig.suptitle(title)
    fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.2, color="#2d3748")
    written = _save_figure(fig, path, dpi=160)
    plt.close(fig)
    return written


def _require_summary(item_folder: str) -> pd.DataFrame:
    path = REPO_ROOT / "6_Results" / BUCKET / item_folder / "all_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"results not generated yet: expected {path}. "
            "Run 5_Experiments/Snapshot_Sample_Size/run_shared.py first."
        )
    return pd.read_csv(path)


def visualize_item(item: str) -> List[Path]:
    item_folder = ITEM_FOLDERS[item]
    summary = _require_summary(item_folder)
    viz = visualizations_dir(item_folder)
    win_long_path(viz).mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    note = ITEM_NOTES[item]
    if item == "1":
        written.extend(_plot_item1(summary, viz, note))
    elif item == "2":
        written.extend(_plot_item2(summary, viz, note))
    else:
        written.extend(_plot_item4(summary, viz, note))
    return written


def _plot_item1(summary: pd.DataFrame, viz: Path, note: str) -> List[Path]:
    written = []
    for (folder, protocol), grp in summary.groupby(["folder_name", "protocol"]):
        for metric in ("f1", "accuracy"):
            fig, ax = plt.subplots(figsize=(10.2, 6.8))
            fig.subplots_adjust(bottom=0.38, top=0.88)
            for model in CLASSIFIER_ORDER:
                _plot_mean_trend(ax, grp, "n_snapshots", metric, model)
            pps = int(grp["points_per_snapshot"].iloc[0])
            _style_axes(
                ax,
                xlabel="Number of snapshots (training)",
                ylabel=METRIC_DISPLAY[metric],
                title=(
                    f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
                    f"{METRIC_DISPLAY[metric]} vs number of snapshots\n"
                    f"Number of snapshots on the x-axis; each cloud has {pps} points"
                ),
            )
            _legend_below(ax, ncol=5, bbox_y=-0.20)
            ax.set_xticks(list(N_SNAPSHOTS_GRID))
            _footnote(ax, note)
            path = viz / f"{folder}_{protocol}_{metric}_by_n_snapshots.png"
            written.append(_save_figure(fig, path, dpi=160))
            plt.close(fig)
            written.append(
                _plot_ci_panels(
                    grp,
                    viz,
                    x_col="n_snapshots",
                    metric=metric,
                    path=viz / f"{folder}_{protocol}_{metric}_by_n_snapshots_ci_panels.png",
                    xlabel="Number of snapshots (training)",
                    title=(
                        f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
                        f"{METRIC_DISPLAY[metric]} vs number of snapshots, one model per panel "
                        f"(95% ribbon). Number of snapshots on the x-axis; each cloud has {pps} points"
                    ),
                    note=note,
                    xticks=list(N_SNAPSHOTS_GRID),
                )
            )
    written.extend(_plot_cross_dataset_facet(summary, viz, "1", "n_snapshots", note))
    return written


def _plot_item2(summary: pd.DataFrame, viz: Path, note: str) -> List[Path]:
    written = []
    for (folder, protocol), grp in summary.groupby(["folder_name", "protocol"]):
        xticks = sorted(int(v) for v in grp["points_per_snapshot"].unique())
        dropped = ""
        if int(grp["binding_class_count"].iloc[0]) <= 60:
            dropped = (
                f" Binding class count on this arm is "
                f"{int(grp['binding_class_count'].iloc[0])}, so any candidate "
                f"≥ that count was dropped."
            )
        panel_note = note + dropped
        for metric in ("f1", "accuracy"):
            fig, ax = plt.subplots(figsize=(10.2, 7.0))
            fig.subplots_adjust(bottom=0.40, top=0.88)
            for model in CLASSIFIER_ORDER:
                _plot_mean_trend(ax, grp, "points_per_snapshot", metric, model)
            _style_axes(
                ax,
                xlabel="Points per snapshot",
                ylabel=METRIC_DISPLAY[metric],
                title=(
                    f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
                    f"{METRIC_DISPLAY[metric]} vs points per snapshot\n"
                    "Points per snapshot on the x-axis; always 60 snapshots"
                ),
            )
            _legend_below(ax, ncol=5, bbox_y=-0.22)
            ax.set_xticks(xticks)
            _footnote(ax, panel_note)
            path = viz / f"{folder}_{protocol}_{metric}_by_points_per_snapshot.png"
            written.append(_save_figure(fig, path, dpi=160))
            plt.close(fig)
            written.append(
                _plot_ci_panels(
                    grp,
                    viz,
                    x_col="points_per_snapshot",
                    metric=metric,
                    path=viz
                    / f"{folder}_{protocol}_{metric}_by_points_per_snapshot_ci_panels.png",
                    xlabel="Points per snapshot",
                    title=(
                        f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
                        f"{METRIC_DISPLAY[metric]} vs points per snapshot, one model per panel "
                        "(95% ribbon). Points per snapshot on the x-axis; always 60 snapshots"
                    ),
                    note=panel_note,
                    xticks=xticks,
                )
            )
    written.extend(
        _plot_cross_dataset_facet(summary, viz, "2", "points_per_snapshot", note)
    )
    return written


def _plot_item4_cloud_size_ci_panels(
    grp: pd.DataFrame,
    viz: Path,
    folder: str,
    protocol: str,
    metric: str,
    note: str,
) -> Path:
    pps_values = [int(v) for v in sorted(grp["points_per_snapshot"].unique())]
    n_pps = max(1, len(pps_values))
    n_models = len(CLASSIFIER_ORDER)
    fig, axes = plt.subplots(
        n_models,
        n_pps,
        figsize=(3.35 * n_pps, 2.45 * n_models),
        sharex=True,
        sharey="row",
    )
    fig.subplots_adjust(bottom=0.14, hspace=0.38, wspace=0.16, top=0.90, left=0.08, right=0.99)
    grid = _as_axes_grid(axes, n_models, n_pps)
    styles = _cloud_size_style(pps_values)
    for row_i, model in enumerate(CLASSIFIER_ORDER):
        for col_i, pps in enumerate(pps_values):
            ax = grid[row_i, col_i]
            sub = grp[grp["points_per_snapshot"] == pps]
            spec = styles[int(pps)]
            _plot_mean_ribbon(
                ax,
                sub,
                "n_snapshots",
                metric,
                model,
                color=spec["color"],
                marker=spec["marker"],
            )
            title = ""
            if row_i == 0:
                title = f"{int(pps)} points per snapshot"
            ylabel = METRIC_DISPLAY[metric] if col_i == 0 else ""
            _style_axes(
                ax,
                xlabel="Number of snapshots" if row_i == n_models - 1 else "",
                ylabel=ylabel,
                title=title,
            )
            if col_i == 0:
                ax.text(
                    -0.28,
                    0.5,
                    CLASSIFIER_DISPLAY[model],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9,
                    color=MODEL_COLORS[model],
                    fontweight="bold" if model in HIGHLIGHT_MODELS else "normal",
                )
            ax.set_xticks(list(N_SNAPSHOTS_GRID))
    fig.suptitle(
        f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
        f"{METRIC_DISPLAY[metric]} families of cloud size, one 95% ribbon per panel"
    )
    fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.0, color="#2d3748")
    path = viz / f"{folder}_{protocol}_{metric}_cloud_size_ci_panels.png"
    written = _save_figure(fig, path, dpi=150)
    plt.close(fig)
    return written


def _plot_item4(summary: pd.DataFrame, viz: Path, note: str) -> List[Path]:
    written = []
    for (folder, protocol), grp in summary.groupby(["folder_name", "protocol"]):
        pps_values = [int(v) for v in sorted(grp["points_per_snapshot"].unique())]
        styles = _cloud_size_style(pps_values)
        # Focus overlay: SVM + Logistic × F1 + accuracy, mean trends only.
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6), sharex=True)
        fig.subplots_adjust(bottom=0.30, hspace=0.30, wspace=0.22, top=0.86)
        for row_i, model in enumerate(("svm", "logistic")):
            for col_i, metric in enumerate(("f1", "accuracy")):
                ax = axes[row_i][col_i]
                for pps in pps_values:
                    spec = styles[int(pps)]
                    _plot_pps_trend(
                        ax,
                        grp,
                        metric,
                        model,
                        pps,
                        color=spec["color"],
                        marker=spec["marker"],
                        label=f"{int(pps)} points per snapshot",
                    )
                _style_axes(
                    ax,
                    xlabel="Number of snapshots (training)",
                    ylabel=METRIC_DISPLAY[metric],
                    title=f"{CLASSIFIER_DISPLAY[model]} — {METRIC_DISPLAY[metric]}",
                )
                ax.set_xticks(list(N_SNAPSHOTS_GRID))
        fig.legend(
            handles=_cloud_size_legend_handles(pps_values),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(4, len(pps_values)),
            frameon=False,
        )
        fig.suptitle(
            f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
            "Families of cloud size (item 4). Number of snapshots on the x-axis; "
            "one curve per cloud size. SVM and Logistic Regression are the focus. "
            "Mean trend across 10 repeats; 95% intervals are on the CI panels.",
            y=1.02,
        )
        fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.2, color="#2d3748")
        path = viz / f"{folder}_{protocol}_svm_logistic_cloud_size_families.png"
        written.append(_save_figure(fig, path, dpi=160))
        plt.close(fig)

        # Overlay small multiples: 5 classifiers × 2 metrics, mean trends only.
        fig, axes = plt.subplots(5, 2, figsize=(10.8, 13.8), sharex=True)
        fig.subplots_adjust(bottom=0.16, hspace=0.40, wspace=0.22, top=0.91)
        for row_i, model in enumerate(CLASSIFIER_ORDER):
            for col_i, metric in enumerate(("f1", "accuracy")):
                ax = axes[row_i][col_i]
                for pps in pps_values:
                    spec = styles[int(pps)]
                    _plot_pps_trend(
                        ax,
                        grp,
                        metric,
                        model,
                        pps,
                        color=spec["color"],
                        marker=spec["marker"],
                        label=f"{int(pps)} points per snapshot"
                        if row_i == 0 and col_i == 0
                        else None,
                    )
                _style_axes(
                    ax,
                    xlabel="Number of snapshots",
                    ylabel=METRIC_DISPLAY[metric],
                    title=f"{CLASSIFIER_DISPLAY[model]} — {METRIC_DISPLAY[metric]}",
                )
                ax.set_xticks(list(N_SNAPSHOTS_GRID))
        fig.legend(
            handles=_cloud_size_legend_handles(pps_values),
            loc="upper center",
            ncol=min(4, len(pps_values)),
            frameon=False,
        )
        fig.suptitle(
            f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
            "All five classifiers (SVM / Logistic thicker; KNN, XGBoost, and "
            "Random Forest at full saturation). Number of snapshots on the x-axis; "
            "one curve per cloud size. Mean trend across 10 repeats; 95% intervals "
            "are on the CI panels."
        )
        fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.0, color="#2d3748")
        path = viz / f"{folder}_{protocol}_all_classifiers_small_multiples.png"
        written.append(_save_figure(fig, path, dpi=150))
        plt.close(fig)

        for metric in ("f1", "accuracy"):
            written.append(
                _plot_item4_cloud_size_ci_panels(grp, viz, folder, protocol, metric, note)
            )
    return written


def _plot_cross_dataset_facet(
    summary: pd.DataFrame,
    viz: Path,
    item: str,
    x_col: str,
    note: str,
) -> List[Path]:
    written = []
    if x_col == "n_snapshots":
        xlabel = "Number of snapshots (training)"
        axis_note = (
            "Number of snapshots on the x-axis; each cloud uses the dataset-aware "
            "default point count (held fixed)"
        )
    else:
        xlabel = "Points per snapshot"
        axis_note = "Points per snapshot on the x-axis; always 60 snapshots"
    for protocol, proto_grp in summary.groupby("protocol"):
        datasets = list(proto_grp["folder_name"].unique())
        n = len(datasets)
        if n == 0:
            continue
        fig, axes = plt.subplots(n, 2, figsize=(11.2, 3.2 * n), sharex=True)
        if n == 1:
            axes = np.array([axes])
        fig.subplots_adjust(bottom=0.20, hspace=0.48, wspace=0.22, top=0.90)
        for i, folder in enumerate(datasets):
            sub = proto_grp[proto_grp["folder_name"] == folder]
            for j, metric in enumerate(("f1", "accuracy")):
                ax = axes[i][j]
                for model in CLASSIFIER_ORDER:
                    _plot_mean_trend(ax, sub, x_col, metric, model)
                _style_axes(
                    ax,
                    xlabel=xlabel,
                    ylabel=METRIC_DISPLAY[metric],
                    title=f"{sub['dataset_display'].iloc[0]} — {METRIC_DISPLAY[metric]}",
                )
                if x_col == "n_snapshots":
                    ax.set_xticks(list(N_SNAPSHOTS_GRID))
                else:
                    ax.set_xticks(sorted(sub[x_col].unique()))
        fig.legend(
            handles=[_model_legend_handle(m) for m in CLASSIFIER_ORDER],
            loc="upper center",
            ncol=5,
            frameon=False,
        )
        fig.suptitle(
            f"{PROTOCOLS[protocol]['display']} — item {item} across datasets\n{axis_note}"
        )
        fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.0, color="#2d3748")
        path = viz / f"cross_dataset_{protocol}_item{item}.png"
        written.append(_save_figure(fig, path, dpi=150))
        plt.close(fig)
    return written


def parse_cli_list(raw: Optional[Sequence[str]], allowed: Iterable[str]) -> List[str]:
    allowed_list = list(allowed)
    if not raw:
        return allowed_list
    out = []
    for item in raw:
        if item not in allowed_list:
            raise ValueError(f"Unknown value {item!r}. Allowed: {allowed_list}")
        out.append(item)
    return out
