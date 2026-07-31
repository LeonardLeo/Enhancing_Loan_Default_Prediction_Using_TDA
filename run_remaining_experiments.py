"""Run the remaining bounded experiments for the four-dataset extension.

This runner consumes artefacts from ``run_new_datasets.py`` and is independently
resumable. Heavy historical500-dependent units are marked waiting until their
barcode CSVs exist; rerunning the same command safely completes them later.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import kmapper as km
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import plot_diagrams
from ripser import ripser
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from run_new_datasets import (
    DATASET_KEYS,
    OUT,
    PROCESSED,
    ROOT,
    SEED,
    TDA,
    get_dataset_config,
    prepare_protocol,
    read_processed,
    snapshots,
)
from utils import (
    compute_barcode_statistics,
    estimate_intrinsic_dimension_levina_bickel,
    estimate_intrinsic_dimension_two_nn,
    permutation_test_algorithm2,
)

matplotlib.use("Agg")

EXT = OUT / "extended"
MANIFEST_PATH = OUT / "extended_manifest.json"
RESULTS_PATH = OUT / "extended_results.csv"


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial(value), indent=2, sort_keys=True), encoding="utf-8")


def manifest_load() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"seed": SEED, "runs": {}}


def set_status(manifest: dict, unit: str, status: str, started: float, error=None, **meta):
    manifest["runs"][unit] = {
        "status": status,
        "runtime_seconds": round(time.time() - started, 3),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "error": error,
        **serial(meta),
    }
    write_json(MANIFEST_PATH, manifest)


def append_results(rows: list[dict]):
    if not rows:
        return
    old = pd.read_csv(RESULTS_PATH).to_dict("records") if RESULTS_PATH.exists() else []
    combined = pd.DataFrame(old + rows)
    keys = [c for c in ["experiment", "dataset", "protocol", "variant", "landmark_percent", "model", "setting"] if c in combined]
    combined.drop_duplicates(keys, keep="last").to_csv(RESULTS_PATH, index=False)


def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        prob = 1 / (1 + np.exp(-decision))
    else:
        prob = np.asarray(pred, dtype=float)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "average_precision": average_precision_score(y_test, prob),
    }


def tuning_configs(y: np.ndarray) -> dict:
    neg, pos = np.bincount(y, minlength=2)
    weight = neg / max(pos, 1)
    max_fold_neighbors = max(1, int(len(y) * 2 / 3) - 1)
    neighbor_grid = sorted({min(3, max_fold_neighbors), min(5, max_fold_neighbors)})
    return {
        "logistic": (
            LogisticRegression(max_iter=2500, class_weight="balanced", random_state=SEED),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=1),
            {"max_depth": [None, 8], "min_samples_leaf": [1, 5]},
        ),
        "svm": (
            SVC(class_weight="balanced", probability=True, random_state=SEED, cache_size=1500),
            {"C": [0.5, 2.0], "kernel": ["linear", "rbf"]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"n_neighbors": neighbor_grid, "weights": ["uniform", "distance"]},
        ),
        "xgb": (
            XGBClassifier(
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=weight, eval_metric="logloss", random_state=SEED, n_jobs=1,
            ),
            {"n_estimators": [100, 200], "max_depth": [3, 5]},
        ),
    }


def tune_all(X_train, y_train, X_test, y_test) -> list[dict]:
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    rows = []
    for name, (model, grid) in tuning_configs(y_train).items():
        search = GridSearchCV(model, grid, scoring="f1", cv=cv, n_jobs=1, error_score="raise")
        search.fit(X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_train, y_train, X_test, y_test)
        rows.append({"model": name, "best_params": json.dumps(search.best_params_, sort_keys=True),
                     "cv_best_f1": search.best_score_, **metrics})
    return rows


def barcode_dirs():
    for key in DATASET_KEYS:
        for protocol in ("historical", "clean"):
            for pct in (10.0, 20.0):
                for variant in ("revised", "historical500"):
                    folder = TDA / key / protocol / f"L{pct:g}" / variant
                    yield key, protocol, pct, variant, folder


def read_barcodes(folder: Path):
    train_path, test_path = folder / "train_barcodes.csv", folder / "test_barcodes.csv"
    if not train_path.exists() or not test_path.exists():
        return None
    return pd.read_csv(train_path), pd.read_csv(test_path)


def run_exp2(manifest: dict, force: bool):
    rows = []
    for key in DATASET_KEYS:
        for protocol in ("historical", "clean"):
            unit, started = f"exp2/{key}/{protocol}", time.time()
            if manifest["runs"].get(unit, {}).get("status") == "completed" and not force:
                continue
            try:
                df = read_processed(key)
                Xtr, Xte, ytr, yte, *_ = prepare_protocol(key, df, protocol)
                tuned = tune_all(Xtr, ytr, Xte, yte)
                for row in tuned:
                    row.update(experiment=2, dataset=key, protocol=protocol,
                               variant="tuned_baseline", setting="bounded_grid_3fold")
                rows.extend(tuned)
                append_results(rows)
                set_status(manifest, unit, "completed", started, models=5)
            except Exception:
                set_status(manifest, unit, "failed", started, traceback.format_exc())
    return rows


def run_barcode_model_experiments(manifest: dict, force: bool):
    """Exp 4 tuned, 6 H0, 11 correlation ablation, 15 KNN, 19 linear."""
    for key, protocol, pct, variant, folder in barcode_dirs():
        pair = read_barcodes(folder)
        if pair is None:
            for exp in (4, 6, 11, 15, 19):
                unit = f"exp{exp}/{key}/{protocol}/L{pct:g}/{variant}"
                if manifest["runs"].get(unit, {}).get("status") != "completed":
                    set_status(manifest, unit, "waiting", time.time(), reason="barcode CSVs not yet available")
            continue
        train, test = pair
        all_features = [c for c in train if c.startswith("g")]
        for exp in (4, 6, 11, 15, 19):
            unit, started = f"exp{exp}/{key}/{protocol}/L{pct:g}/{variant}", time.time()
            if manifest["runs"].get(unit, {}).get("status") == "completed" and not force:
                continue
            try:
                Xtr, Xte = train[all_features], test[all_features]
                ytr, yte = train["label"].astype(int), test["label"].astype(int)
                output = []
                if exp == 4:
                    output = tune_all(Xtr, ytr, Xte, yte)
                elif exp == 6:
                    h0 = [c for c in all_features if c.endswith("_0")]
                    output = [{"model": "logistic", **evaluate(
                        LogisticRegression(max_iter=2000, random_state=SEED),
                        Xtr[h0], ytr, Xte[h0], yte,
                    )}]
                elif exp == 11:
                    corr = Xtr.corr().abs()
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    keep = [c for c in all_features if not any(upper[c] > 0.80)]
                    output = [{"model": "logistic", "features_before": len(all_features), "features_after": len(keep),
                               **evaluate(LogisticRegression(max_iter=2000, random_state=SEED),
                                          Xtr[keep], ytr, Xte[keep], yte)}]
                elif exp == 15:
                    max_k = min(20, len(Xtr) - 1)
                    for k in range(1, max_k + 1):
                        output.append({"model": "knn", "setting": f"k={k}", **evaluate(
                            KNeighborsClassifier(n_neighbors=k), Xtr, ytr, Xte, yte
                        )})
                else:
                    reg = LinearRegression().fit(Xtr, ytr)
                    pred = (reg.predict(Xte) >= 0.5).astype(int)
                    output = [{"model": "linear_regression", "accuracy": accuracy_score(yte, pred),
                               "balanced_accuracy": balanced_accuracy_score(yte, pred),
                               "precision": precision_score(yte, pred, zero_division=0),
                               "recall": recall_score(yte, pred, zero_division=0),
                               "f1": f1_score(yte, pred, zero_division=0)}]
                for row in output:
                    row.update(experiment=exp, dataset=key, protocol=protocol, variant=variant,
                               landmark_percent=pct, setting=row.get("setting", "default"))
                append_results(output)
                set_status(manifest, unit, "completed", started, rows=len(output))
            except Exception:
                set_status(manifest, unit, "failed", started, traceback.format_exc())


def run_eda_dr_covariance(manifest: dict, force: bool):
    """Experiments 7–10 and 17 with bounded samples."""
    for key in DATASET_KEYS:
        unit, started = f"exp9_17/{key}/clean", time.time()
        if manifest["runs"].get(unit, {}).get("status") != "completed" or force:
            try:
                df = read_processed(key)
                _, _, ytr, yte, Xtr, Xte, _, pca, meta = prepare_protocol(key, df, "clean")
                X = np.vstack([Xtr, Xte])
                y = np.hstack([ytr, yte])
                folder = EXT / "9_17_original_dr" / key
                folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "pca1": X[:, 0], "pca2": X[:, 1] if X.shape[1] > 1 else 0,
                    "target": y, "split": ["train"] * len(Xtr) + ["test"] * len(Xte),
                }).to_csv(folder / "pca_coordinates.csv", index=False)
                rng = np.random.default_rng(SEED)
                idx = rng.choice(len(X), min(2000, len(X)), replace=False)
                perplexity = min(30, max(5, len(idx) // 20))
                ts = TSNE(n_components=2, perplexity=perplexity, random_state=SEED,
                          init="pca", learning_rate="auto").fit_transform(X[idx])
                pd.DataFrame({"tsne1": ts[:, 0], "tsne2": ts[:, 1], "target": y[idx]}).to_csv(
                    folder / "tsne_coordinates.csv", index=False
                )
                write_json(folder / "pca_metadata.json", meta)
                set_status(manifest, unit, "completed", started, pca_components=pca.n_components_)
            except Exception:
                set_status(manifest, unit, "failed", started, traceback.format_exc())

    for key, protocol, pct, variant, folder in barcode_dirs():
        pair = read_barcodes(folder)
        if pair is None:
            continue
        unit, started = f"exp7_8_10/{key}/{protocol}/L{pct:g}/{variant}", time.time()
        if manifest["runs"].get(unit, {}).get("status") == "completed" and not force:
            continue
        try:
            train, test = pair
            data = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
            features = [c for c in data if c.startswith("g")]
            target = EXT / "7_8_10_barcode_analysis" / key / protocol / f"L{pct:g}" / variant
            target.mkdir(parents=True, exist_ok=True)
            desc = data.groupby(["split", "label"])[features].agg(["mean", "std", "median"])
            desc.to_csv(target / "barcode_eda.csv")
            pca = PCA(n_components=2, random_state=SEED)
            coords = pca.fit_transform(MinMaxScaler().fit_transform(data[features]))
            pd.DataFrame({"pca1": coords[:, 0], "pca2": coords[:, 1],
                          "label": data.label, "split": data.split}).to_csv(target / "pca_coordinates.csv", index=False)
            cov_rows = []
            scaled = MinMaxScaler().fit_transform(data[features])
            for label in (0, 1):
                points = scaled[data.label.to_numpy() == label]
                centroid = points.mean(axis=0)
                distances = np.linalg.norm(points - centroid, axis=1)
                cov_rows.append({"label": label, "mean_centroid_distance": distances.mean(),
                                 "max_centroid_distance": distances.max(),
                                 "covariance_trace": np.trace(np.cov(points, rowvar=False))})
            pd.DataFrame(cov_rows).to_csv(target / "covariance_centroid_summary.csv", index=False)
            set_status(manifest, unit, "completed", started, rows=len(data))
        except Exception:
            set_status(manifest, unit, "failed", started, traceback.format_exc())


def snapshots_fixed_t(X, y, t: int, count: int):
    rows = []
    actual_t = min(t, int(min(np.bincount(y))))
    for label in (0, 1):
        pool = X[y == label]
        for i in range(count):
            rng = np.random.default_rng(SEED + i + label * 1_000_003)
            points = pool[rng.choice(len(pool), size=actual_t, replace=False)]
            diagrams = ripser(points, maxdim=1)["dgms"]
            row = {"label": label, "snapshot": i}
            for dim, diagram in enumerate(diagrams[:2]):
                for j, value in enumerate(compute_barcode_statistics(diagram), 1):
                    row[f"g{j}_{dim}"] = value
            rows.append(row)
    return pd.DataFrame(rows), actual_t, {}


def run_controls(manifest: dict, force: bool):
    """Experiments 12/13/16/18: matched t, common variance, PCA sweeps."""
    reference_t = [6, 12]  # 10% and 20% of PKDD clean train minority n=61
    calibration = {
        "sample_size_reference": "PKDD clean training minority class (n=61)",
        "matched_landmark_sizes": reference_t,
        "pca_variance_reference": 0.90,
        "pca_sweep_components": [2, 5, 10, 20],
    }
    write_json(EXT / "12_13_16_18_controls" / "calibration.json", calibration)
    for key in DATASET_KEYS:
        unit, started = f"exp12_13_16_18/{key}/clean", time.time()
        if manifest["runs"].get(unit, {}).get("status") == "completed" and not force:
            continue
        try:
            df = read_processed(key)
            Xraw_tr, Xraw_te, ytr, yte, Xtr, Xte, *_rest, meta = prepare_protocol(key, df, "clean")
            rows = []
            for n_components in [c for c in calibration["pca_sweep_components"] if c <= min(Xraw_tr.shape)]:
                pca = PCA(n_components=n_components, random_state=SEED)
                tr, te = pca.fit_transform(Xraw_tr), pca.transform(Xraw_te)
                metrics = evaluate(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
                                   tr, ytr, te, yte)
                rows.append({"experiment": 16 if key in ("pkdd_czech", "polish_bankruptcy") else 18,
                             "dataset": key, "protocol": "clean", "variant": "pca_component_sweep",
                             "setting": f"components={n_components}", "pca_variance": pca.explained_variance_ratio_.sum(),
                             "model": "logistic", **metrics})
            for t in reference_t:
                l_train = max(2, math.ceil(min(np.bincount(ytr)) / t))
                l_test = max(2, math.ceil(min(np.bincount(yte)) / min(t, min(np.bincount(yte)))))
                train_bc, actual_t, _ = snapshots_fixed_t(Xtr, ytr, t, l_train)
                test_bc, _, _ = snapshots_fixed_t(Xte, yte, min(t, min(np.bincount(yte))), l_test)
                feats = [c for c in train_bc if c.startswith("g")]
                metrics = evaluate(LogisticRegression(max_iter=2000, random_state=SEED),
                                   train_bc[feats], train_bc.label, test_bc[feats], test_bc.label)
                rows.append({"experiment": 12, "dataset": key, "protocol": "clean",
                             "variant": "matched_sample_size", "setting": f"t={actual_t}",
                             "model": "logistic", **metrics})
            variance_metrics = evaluate(
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
                Xtr, ytr, Xte, yte,
            )
            rows.append({"experiment": 13, "dataset": key, "protocol": "clean",
                         "variant": "matched_pca_variance", "setting": "reference_variance=0.90",
                         "model": "logistic", "pca_variance": meta["pca_variance"],
                         "pca_components": meta["pca_components"], **variance_metrics})
            append_results(rows)
            set_status(manifest, unit, "completed", started, rows=len(rows))
        except Exception:
            set_status(manifest, unit, "failed", started, traceback.format_exc())


def run_imbalanced(manifest: dict, force: bool):
    """Experiment 14: bounded 1:4 default/non-default snapshot training ratio."""
    for key in DATASET_KEYS:
        unit, started = f"exp14/{key}/clean/L10", time.time()
        if manifest["runs"].get(unit, {}).get("status") == "completed" and not force:
            continue
        try:
            df = read_processed(key)
            _, _, ytr, yte, Xtr, Xte, *_ = prepare_protocol(key, df, "clean")
            base, _, _ = snapshots(Xtr, ytr, 10, 80)
            train = pd.concat([
                base[base.label == 1].iloc[:20],
                base[base.label == 0].iloc[:80],
            ]).sample(frac=1, random_state=SEED)
            test, _, _ = snapshots(Xte, yte, 10, 40)
            feats = [c for c in train if c.startswith("g")]
            metrics = evaluate(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
                               train[feats], train.label, test[feats], test.label)
            append_results([{**metrics, "experiment": 14, "dataset": key, "protocol": "clean",
                             "variant": "1_default_to_4_nondefault", "landmark_percent": 10,
                             "setting": "20 default / 80 non-default snapshots", "model": "logistic"}])
            set_status(manifest, unit, "completed", started)
        except Exception:
            set_status(manifest, unit, "failed", started, traceback.format_exc())


def run_mapper_and_diagrams(manifest: dict, force: bool):
    """Experiments 5, 21, 22 with one documented bounded Mapper configuration."""
    mapper_config = {"n_cubes": 10, "perc_overlap": 0.30, "clusters": 3, "lens": "PCA-2"}
    write_json(EXT / "5_21_22_visuals" / "bounded_mapper_config.json", mapper_config)
    for key in DATASET_KEYS:
        for exp in (5, 21, 22):
            unit, started = f"exp{exp}/{key}/clean", time.time()
            if manifest["runs"].get(unit, {}).get("status") == "completed" and not force:
                continue
            try:
                target = EXT / "5_21_22_visuals" / key
                target.mkdir(parents=True, exist_ok=True)
                if exp == 5:
                    df = read_processed(key)
                    _, _, ytr, _, Xtr, _, *_ = prepare_protocol(key, df, "clean")
                    rng = np.random.default_rng(SEED)
                    idx = rng.choice(len(Xtr), min(2500, len(Xtr)), replace=False)
                    points, labels = Xtr[idx], ytr[idx]
                    lens = PCA(2, random_state=SEED).fit_transform(points)
                    graph = km.KeplerMapper(verbose=0).map(
                        lens, points,
                        cover=km.Cover(n_cubes=10, perc_overlap=0.30),
                        clusterer=KMeans(n_clusters=3, random_state=SEED, n_init=10),
                    )
                    km.KeplerMapper(verbose=0).visualize(
                        graph, path_html=str(target / "exp5_original_mapper.html"),
                        color_values=labels, color_function_name=["Default status"],
                        title=f"{key} original-feature Mapper",
                    )
                elif exp == 21:
                    folder = TDA / key / "clean" / "L10" / "revised"
                    pair = read_barcodes(folder)
                    if pair is None:
                        raise FileNotFoundError(folder)
                    train, _ = pair
                    feats = [c for c in train if c.startswith("g")]
                    points = MinMaxScaler().fit_transform(train[feats])
                    lens = PCA(2, random_state=SEED).fit_transform(points)
                    graph = km.KeplerMapper(verbose=0).map(
                        lens, points, cover=km.Cover(n_cubes=10, perc_overlap=0.30),
                        clusterer=KMeans(n_clusters=1, random_state=SEED, n_init=10),
                    )
                    km.KeplerMapper(verbose=0).visualize(
                        graph, path_html=str(target / "exp21_barcode_mapper.html"),
                        color_values=train.label.to_numpy(), color_function_name=["Default status"],
                        title=f"{key} barcode Mapper",
                    )
                else:
                    source = TDA / key / "clean" / "L10" / "revised" / "representative_diagrams.npz"
                    diagrams = np.load(source)
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    for i, label in enumerate(("0", "1")):
                        diagram = diagrams[f"class_{label}"]
                        if len(diagram):
                            plot_diagrams(diagram, ax=axes[i], show=False)
                        else:
                            axes[i].text(0.5, 0.5, "No finite H1 pairs", ha="center", va="center")
                            axes[i].set_xlabel("Birth")
                            axes[i].set_ylabel("Death")
                        axes[i].set_title(f"Class {label}")
                    fig.suptitle(f"{key}: representative persistence diagrams")
                    fig.tight_layout()
                    fig.savefig(target / "exp22_persistence_diagrams.png", dpi=160)
                    plt.close(fig)
                set_status(manifest, unit, "completed", started, **mapper_config if exp in (5, 21) else {})
            except Exception:
                set_status(manifest, unit, "failed", started, traceback.format_exc())


def run_historical_stats(manifest: dict, force: bool):
    """Experiments 25/27 for historical500; Exp26 is variant-independent."""
    rows = []
    out_path = OUT / "statistical_results_historical500.csv"
    for key, protocol, pct, variant, folder in barcode_dirs():
        if variant != "historical500":
            continue
        unit, started = f"exp25_27/{key}/{protocol}/L{pct:g}/historical500", time.time()
        pair = read_barcodes(folder)
        if pair is None:
            set_status(manifest, unit, "waiting", started, reason="historical500 barcodes not available")
            continue
        if manifest["runs"].get(unit, {}).get("status") == "completed" and not force:
            continue
        try:
            train, _ = pair
            feats = [c for c in train if c.startswith("g")]
            a = train[train.label == 0][feats].to_numpy()
            b = train[train.label == 1][feats].to_numpy()
            result = {
                "dataset": key, "protocol": protocol, "landmark_percent": pct,
                "snapshot_variant": variant, "global_mean": train[feats].to_numpy().mean(),
                "global_variance": train[feats].to_numpy().var(ddof=1),
                **permutation_test_algorithm2(a, b, n_permutations=199, random_state=SEED),
                "fpq_method": "barcode-vector proxy; not true persistence-diagram distance",
            }
            rows.append(result)
            old = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
            pd.DataFrame(old + rows).drop_duplicates(
                ["dataset", "protocol", "landmark_percent"], keep="last"
            ).to_csv(out_path, index=False)
            set_status(manifest, unit, "completed", started)
        except Exception:
            set_status(manifest, unit, "failed", started, traceback.format_exc())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", type=int,
                        default=[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 27])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    EXT.mkdir(parents=True, exist_ok=True)
    manifest = manifest_load()
    requested = set(args.experiments)
    if 2 in requested:
        run_exp2(manifest, args.force)
    if requested & {4, 6, 11, 15, 19}:
        run_barcode_model_experiments(manifest, args.force)
    if requested & {7, 8, 9, 10, 17}:
        run_eda_dr_covariance(manifest, args.force)
    if requested & {12, 13, 16, 18}:
        run_controls(manifest, args.force)
    if 14 in requested:
        run_imbalanced(manifest, args.force)
    if requested & {5, 21, 22}:
        run_mapper_and_diagrams(manifest, args.force)
    if requested & {25, 27}:
        run_historical_stats(manifest, args.force)
    manifest["finished_at"] = pd.Timestamp.utcnow().isoformat()
    write_json(MANIFEST_PATH, manifest)


if __name__ == "__main__":
    main()
