# -*- coding: utf-8 -*-
"""
Experiment 9 — Revised Snapshot Protocol (historically Exp 28)

Arm knobs (this copy is Historical_Late_Split_Balanced_TDA):
  PROTOCOL_BUCKET = Historical_Late_Split_Balanced_TDA
  SPLIT_TIMING    = late
  UNDERSAMPLE     = True

Snapshot rules (this experiment only): fixed absolute t, default l_train/l_test = 60/15.
PCA ranks come from utils.DatasetConfig (same as Exp 3 / Design_Decisions.md).

Run from repo root:
  .\\tda_env\\Scripts\\python.exe 5_Experiments/{PROTOCOL_BUCKET}/9_Revised_Snapshot_Protocol/run_protocol.py
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol_lib import (  # noqa: E402
    DCCCD_FULL_L,
    DEFAULT_TEST_L,
    DEFAULT_TRAIN_L,
    ZANIAR_TEST_L,
    ZANIAR_TRAIN_L,
    analyze_snapshot_overlap,
    audit_reuse_constraints,
    build_barcode_matrix_for_tag,
    estimate_intrinsic_dimensions,
    fit_simple_models,
    formula_l_from_t_b,
    generate_fixed_t_snapshots,
    overlap_significance_tests,
    prepare_protocol_clouds,
    recommend_t_l_separated,
    save_json,
    split_classes_no_balance,
    undersample_xy,
    win_long_path,
)
from utils import get_dataset_config  # noqa: E402

# First-class arm knobs — change these three when cloning this folder.
PROTOCOL_BUCKET = "Historical_Late_Split_Balanced_TDA"
SPLIT_TIMING = "late"
UNDERSAMPLE = True
EXP_NAME = "9_Revised_Snapshot_Protocol"

RESULTS = REPO_ROOT / "6_Results" / PROTOCOL_BUCKET / EXP_NAME
DATA_LANDMARKS = REPO_ROOT / "1_Data" / "Landmark_Sets" / PROTOCOL_BUCKET / EXP_NAME
DATA_TDA = REPO_ROOT / "1_Data" / "TDA_Datasets" / PROTOCOL_BUCKET / EXP_NAME
DATA_BARCODES = REPO_ROOT / "1_Data" / "Barcode_Statistics" / PROTOCOL_BUCKET / EXP_NAME

DATASET_SPECS = {
    "credit_card_default": {
        "path": REPO_ROOT
        / "1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/processed_data.xlsx",
        "run_full_nonsplit": True,
    },
    "statlog_german": {
        "path": REPO_ROOT
        / "1_Data/Processed_Datasets/Statlog_German_Credit_Data/processed_data.xlsx",
        "run_full_nonsplit": False,
    },
    "south_german_credit": {
        "path": REPO_ROOT / "1_Data/Processed_Datasets/South_German_Credit/processed_data.csv",
        "run_full_nonsplit": False,
    },
    "pkdd_czech": {
        "path": REPO_ROOT / "1_Data/Processed_Datasets/PKDD_Czech_Financial/processed_data.csv",
        "run_full_nonsplit": False,
    },
    "polish_bankruptcy": {
        "path": REPO_ROOT
        / "1_Data/Processed_Datasets/Polish_Bankruptcy_3Year/processed_data.csv",
        "run_full_nonsplit": False,
    },
    "taiwan_bankruptcy": {
        "path": REPO_ROOT / "1_Data/Processed_Datasets/Taiwan_Bankruptcy/processed_data.csv",
        "run_full_nonsplit": False,
    },
}


def _pca_rank(dataset_key: str) -> int:
    return int(get_dataset_config(dataset_key).notes["pca_n_components_exp3"])


def load_xy(dataset_key: str):
    cfg = get_dataset_config(dataset_key)
    spec = DATASET_SPECS[dataset_key]
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    target = cfg.target_column
    if target not in df.columns:
        # common alternates
        for alt in ("Class", "class", "target", "Target", "default payment next month", "y"):
            if alt in df.columns:
                target = alt
                break
    drop = [c for c in ("Unnamed: 0", "id", "ID", target) if c in df.columns]
    X = df.drop(columns=drop)
    # keep numeric only for PCA/PH
    X = X.select_dtypes(include=[np.number]).copy()
    y = df[target].astype(int)
    # map south german if needed (already 0/1 in processed)
    return X, y, cfg, spec


def design_for_dataset(dataset_key: str) -> dict:
    from protocol_lib import choose_joint_t_train_test_l, max_t_for_reuse
    from sklearn.impute import SimpleImputer

    X, y, cfg, spec = load_xy(dataset_key)
    pos = int((y == cfg.positive_label).sum())
    neg = int((y != cfg.positive_label).sum())

    # ID on median-imputed raw numeric (no split leakage for exploratory ID)
    X_id = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X), columns=X.columns
    )
    id_raw = estimate_intrinsic_dimensions(X_id.values, n_samples=min(2000, len(X_id)))

    pca_n = _pca_rank(dataset_key)
    Xtr, Xte, ytr, yte, cloud = prepare_protocol_clouds(
        X,
        y,
        n_components=pca_n,
        split_timing=SPLIT_TIMING,
        undersample=UNDERSAMPLE,
        positive_label=cfg.positive_label,
        random_state=42,
    )
    var = cloud["variance_retained"]
    id_pca = estimate_intrinsic_dimensions(Xtr.values, n_samples=min(2000, len(Xtr)))
    b = id_pca["b_primary_TwoNN"]
    if not np.isfinite(b):
        b = id_raw["b_primary_TwoNN"]

    train_pos = int(cloud["train_pos"])
    train_neg = int(cloud["train_neg"])
    test_pos = int(cloud["test_pos"])
    test_neg = int(cloud["test_neg"])

    # Concern A/B tables still reported on each pool at meeting targets
    rec_train = recommend_t_l_separated(
        train_pos, train_neg, b=b, train_l_target=DEFAULT_TRAIN_L, test_l_target=DEFAULT_TEST_L
    )
    rec_test_pool = recommend_t_l_separated(
        test_pos, test_neg, b=b, train_l_target=DEFAULT_TRAIN_L, test_l_target=DEFAULT_TEST_L
    )

    # Joint choice: one t, train_l≈60, test_l≈15, reuse-safe on both pools
    joint = choose_joint_t_train_test_l(
        train_pos, train_neg, test_pos, test_neg,
        target_train_l=DEFAULT_TRAIN_L,
        target_test_l=DEFAULT_TEST_L,
    )
    chosen_t = int(joint["t"])
    eff_train_l = int(joint["train_l"])
    eff_test_l = int(joint["test_l"])

    notes = []
    if eff_train_l < DEFAULT_TRAIN_L or eff_test_l < DEFAULT_TEST_L:
        notes.append(
            f"Concern B joint choice set train_l={eff_train_l}, test_l={eff_test_l} "
            f"(meeting asked 60/15) at t={chosen_t}; "
            f"train_min={joint['n_train_min']}, test_min={joint['n_test_min']}"
        )
    if joint.get("relaxed_test_reuse"):
        notes.append(
            f"Test reuse limit relaxed to {joint['test_reuse_limit']} because "
            "strict reuse<=1 could not support min_test_l with a usable t"
        )

    t_cap = chosen_t
    flo = max(5, t_cap // 3) if t_cap >= 15 else max(3, t_cap // 3)
    mid = max(flo, (2 * t_cap) // 3)
    t_sweep = sorted({flo, mid, t_cap})

    full_rec = recommend_t_l_separated(pos, neg, b=b, train_l_target=60, test_l_target=15)

    design = {
        "dataset_key": dataset_key,
        "display_name": cfg.display_name,
        "n_total": int(len(X)),
        "n_pos": pos,
        "n_neg": neg,
        "default_rate": pos / len(X),
        "pca_components": pca_n,
        "pca_variance_retained": var,
        "pca_fit": cloud["pca_fit"],
        "protocol_bucket": PROTOCOL_BUCKET,
        "split_timing": SPLIT_TIMING,
        "undersample": UNDERSAMPLE,
        "intrinsic_dim_raw": id_raw,
        "intrinsic_dim_pca_train": id_pca,
        "b_used": float(b),
        "split_counts": {
            "train_pos": train_pos,
            "train_neg": train_neg,
            "test_pos": test_pos,
            "test_neg": test_neg,
        },
        "meeting_defaults": {"train_l": DEFAULT_TRAIN_L, "test_l": DEFAULT_TEST_L},
        "effective_defaults": {
            "train_l": eff_train_l,
            "test_l": eff_test_l,
            "joint_choice": joint,
            "notes": notes,
        },
        "zaniar_sweep": {"train_l": list(ZANIAR_TRAIN_L), "test_l": list(ZANIAR_TEST_L)},
        # Largest t that can host the full Zaniar corner (train_l=100, test_l=30) under R<=1
        "zaniar_t": int(
            max(
                0,
                min(
                    max_t_for_reuse(min(train_pos, train_neg), max(ZANIAR_TRAIN_L)),
                    max_t_for_reuse(min(test_pos, test_neg), max(ZANIAR_TEST_L)),
                ),
            )
        ),
        "dcccd_full_l": list(DCCCD_FULL_L),
        "concern_A_and_B_train_pool": rec_train,
        "concern_A_and_B_test_pool": rec_test_pool,
        "chosen_t": chosen_t,
        "t_sweep": t_sweep,
        "formula_at_chosen_t": (
            formula_l_from_t_b(chosen_t, b) if (chosen_t >= 3 and np.isfinite(b) and b > 0) else None
        ),
        "full_data_rec": full_rec,
        "worked_examples": _worked_examples(
            chosen_t, b, train_pos, train_neg, test_pos, test_neg, eff_train_l, eff_test_l
        ),
    }

    out_dir = RESULTS / cfg.folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "design.json", design)

    # Flat tables for the report
    formula_df = pd.DataFrame(rec_train["concern_A_formula"]["rows"])
    formula_df.to_csv(out_dir / "concern_A_formula_rows.csv", index=False)
    reuse_df = pd.DataFrame(rec_train["concern_B_reuse"]["rows"])
    reuse_df.to_csv(out_dir / "concern_B_reuse_rows.csv", index=False)

    # Worked calculation table
    pd.DataFrame(design["worked_examples"]).to_csv(out_dir / "worked_calculations.csv", index=False)
    return design


def _worked_examples(t, b, train_pos, train_neg, test_pos, test_neg, train_l=None, test_l=None):
    rows = []
    train_l = DEFAULT_TRAIN_L if train_l is None else int(train_l)
    test_l = DEFAULT_TEST_L if test_l is None else int(test_l)
    if t < 3 or not np.isfinite(b) or b <= 0:
        return rows
    l_f = formula_l_from_t_b(t, b)
    rows.append(
        {
            "step": "A1_formula",
            "expression": "l = (t / ln(t))^(2/b)",
            "t": t,
            "b": b,
            "ln_t": float(np.log(t)),
            "t_over_ln_t": float(t / np.log(t)),
            "exponent_2_over_b": float(2.0 / b),
            "result_l_formula": l_f,
            "notes": "Concern A only — theoretical snapshot count suggestion",
        }
    )
    for split, npos, nneg, l in (
        ("train", train_pos, train_neg, train_l),
        ("test", test_pos, test_neg, test_l),
    ):
        audit = audit_reuse_constraints(npos, nneg, t, l)
        rows.append(
            {
                "step": f"B_{split}_reuse",
                "expression": "R = (t*l)/n_class",
                "t": t,
                "l": l,
                "n_pos": npos,
                "n_neg": nneg,
                "reuse_pos": audit["reuse_pos"],
                "reuse_neg": audit["reuse_neg"],
                "ok_reuse": audit["ok_reuse"],
                "ok_t_fraction": audit["ok_t_fraction"],
                "notes": "Concern B only — sampling reuse feasibility at effective l",
            }
        )
    return rows


def _run_key_done(results_csv: Path, run_key: str) -> bool:
    if not results_csv.exists():
        return False
    df = pd.read_csv(results_csv)
    return run_key in set(df.get("run_key", []).astype(str))


def run_split_setting(
    dataset_key: str,
    design: dict,
    t: int,
    train_l: int,
    test_l: int,
    mode: str = "default",
) -> None:
    X, y, cfg, spec = load_xy(dataset_key)
    folder = cfg.folder_name
    run_key = f"{dataset_key}|split|t{t}|train{train_l}|test{test_l}|{mode}"
    results_csv = RESULTS / folder / "ml_results.csv"
    if _run_key_done(results_csv, run_key):
        print(f"[skip] {run_key}")
        return

    print(f"\n=== {run_key} ===")
    Xtr, Xte, ytr, yte, cloud = prepare_protocol_clouds(
        X,
        y,
        n_components=_pca_rank(dataset_key),
        split_timing=SPLIT_TIMING,
        undersample=UNDERSAMPLE,
        positive_label=cfg.positive_label,
        random_state=42,
    )
    var = cloud["variance_retained"]
    train_classes = split_classes_no_balance(Xtr, ytr, positive_label=cfg.positive_label)
    test_classes = split_classes_no_balance(Xte, yte, positive_label=cfg.positive_label)

    # Feasibility guard
    for name, frame in {**{f"train/{k}": v for k, v in train_classes.items()}, **{f"test/{k}": v for k, v in test_classes.items()}.items():
        if len(frame) < t:
            raise ValueError(f"{run_key}: pool {name} has {len(frame)} < t={t}")

    lm_train = win_long_path(DATA_LANDMARKS / folder / f"split_t{t}_tr{train_l}_te{test_l}" / "train")
    lm_test = win_long_path(DATA_LANDMARKS / folder / f"split_t{t}_tr{train_l}_te{test_l}" / "test")
    tda_dir = win_long_path(DATA_TDA / folder / f"split_t{t}_tr{train_l}_te{test_l}")
    bar_dir = win_long_path(DATA_BARCODES / folder / f"split_t{t}_tr{train_l}_te{test_l}")
    tda_dir.mkdir(parents=True, exist_ok=True)
    bar_dir.mkdir(parents=True, exist_ok=True)

    meta_tr = generate_fixed_t_snapshots(
        train_classes, t=t, l=train_l, output_root=lm_train, tag="train", random_state=42, undersample=UNDERSAMPLE
    )
    meta_te = generate_fixed_t_snapshots(
        test_classes, t=t, l=test_l, output_root=lm_test, tag="test", random_state=43, undersample=UNDERSAMPLE
    )

    # Overlap + significance on train default class (and non-default)
    overlap_report = {}
    for cname in ("default", "non-default"):
        idx = meta_tr["index_sets"][cname]
        n_pool = meta_tr["classes"][cname]["n_pool"]
        summary = analyze_snapshot_overlap(idx, t=t, n_pool=n_pool, random_state=42)
        # strip heavy arrays for JSON
        summary_light = {k: v for k, v in summary.items() if k not in ("jaccard_values", "overlap_frac_values")}
        sig = overlap_significance_tests(
            idx, t=t, n_pool=n_pool, n_permutations=150, random_state=42
        )
        overlap_report[f"train_{cname}"] = {"summary": summary_light, "significance": sig}

    save_json(RESULTS / folder / f"overlap_{run_key.replace('|', '_')}.json", overlap_report)

    train_bar = build_barcode_matrix_for_tag(lm_train, t=t)
    test_bar = build_barcode_matrix_for_tag(lm_test, t=t)
    train_bar.to_csv(tda_dir / "train_barcodes.csv", index=False)
    test_bar.to_csv(tda_dir / "test_barcodes.csv", index=False)
    train_bar.to_csv(bar_dir / "train_barcodes.csv", index=False)
    test_bar.to_csv(bar_dir / "test_barcodes.csv", index=False)

    ml_rows = fit_simple_models(train_bar, test_bar, random_state=42)
    for row in ml_rows:
        row.update(
            {
                "run_key": run_key,
                "dataset": dataset_key,
                "protocol": f"{SPLIT_TIMING}_split_{'undersample' if UNDERSAMPLE else 'no_undersample'}_fixed_t",
                "protocol_bucket": PROTOCOL_BUCKET,
                "mode": mode,
                "t": t,
                "train_l": train_l,
                "test_l": test_l,
                "b_used": design["b_used"],
                "formula_l": design.get("formula_at_chosen_t"),
                "pca_variance": var,
                "n_train_snapshots": len(train_bar),
                "n_test_snapshots": len(test_bar),
                "train_class_balance": train_bar["label"].value_counts().to_dict(),
                "reuse_train_default": audit_reuse_constraints(
                    design["split_counts"]["train_pos"],
                    design["split_counts"]["train_neg"],
                    t,
                    train_l,
                )["reuse_pos"],
                "reuse_train_nondefault": audit_reuse_constraints(
                    design["split_counts"]["train_pos"],
                    design["split_counts"]["train_neg"],
                    t,
                    train_l,
                )["reuse_neg"],
            }
        )

    out_df = pd.DataFrame(ml_rows)
    if results_csv.exists():
        out_df = pd.concat([pd.read_csv(results_csv), out_df], ignore_index=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(results_csv, index=False)
    print(f"Saved ML results -> {results_csv}")


def run_full_nonsplit(dataset_key: str, design: dict, t: int, l: int) -> None:
    """Non-split full-data snapshots (DCCCD 60–90). Evaluate via internal 80/20 on barcodes."""
    X, y, cfg, spec = load_xy(dataset_key)
    folder = cfg.folder_name
    run_key = f"{dataset_key}|full|t{t}|l{l}"
    results_csv = RESULTS / folder / "ml_results.csv"
    if _run_key_done(results_csv, run_key):
        print(f"[skip] {run_key}")
        return

    print(f"\n=== {run_key} ===")
    # Scale+PCA on FULL data intentionally for this non-split sensitivity arm
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA

    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X_imp)
    pca_n = _pca_rank(dataset_key)
    n_comp = min(pca_n, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    Xp = pd.DataFrame(
        pca.fit_transform(Xs), columns=[f"PCA_{i}" for i in range(1, n_comp + 1)]
    )
    if UNDERSAMPLE:
        Xp, y = undersample_xy(Xp, y, positive_label=cfg.positive_label, random_state=42)
    classes = split_classes_no_balance(Xp, y, positive_label=cfg.positive_label)

    lm = win_long_path(DATA_LANDMARKS / folder / f"full_t{t}_l{l}")
    meta = generate_fixed_t_snapshots(classes, t=t, l=l, output_root=lm, tag="full", random_state=42, undersample=UNDERSAMPLE)

    overlap_report = {}
    for cname in ("default", "non-default"):
        idx = meta["index_sets"][cname]
        n_pool = meta["classes"][cname]["n_pool"]
        summary = analyze_snapshot_overlap(idx, t=t, n_pool=n_pool)
        summary_light = {k: v for k, v in summary.items() if k not in ("jaccard_values", "overlap_frac_values")}
        sig = overlap_significance_tests(idx, t=t, n_pool=n_pool, n_permutations=150)
        overlap_report[cname] = {"summary": summary_light, "significance": sig}
    save_json(RESULTS / folder / f"overlap_{run_key.replace('|', '_')}.json", overlap_report)

    bar = build_barcode_matrix_for_tag(lm, t=t)
    tda_dir = win_long_path(DATA_TDA / folder / f"full_t{t}_l{l}")
    bar_dir = win_long_path(DATA_BARCODES / folder / f"full_t{t}_l{l}")
    tda_dir.mkdir(parents=True, exist_ok=True)
    bar_dir.mkdir(parents=True, exist_ok=True)
    bar.to_csv(tda_dir / "all_barcodes.csv", index=False)
    bar.to_csv(bar_dir / "all_barcodes.csv", index=False)

    # Stratified split on barcode rows (snapshot-level)
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(bar, test_size=0.2, random_state=42, stratify=bar["label"])
    ml_rows = fit_simple_models(tr, te, random_state=42)
    for row in ml_rows:
        row.update(
            {
                "run_key": run_key,
                "dataset": dataset_key,
                "protocol": "full_data_nonsplit_then_barcode_split",
                "mode": "full_dcccd_range",
                "t": t,
                "train_l": l,
                "test_l": None,
                "b_used": design["b_used"],
                "formula_l": formula_l_from_t_b(t, design["b_used"]),
                "pca_variance": float(pca.explained_variance_ratio_.sum()),
                "n_train_snapshots": len(tr),
                "n_test_snapshots": len(te),
                "reuse_binding": audit_reuse_constraints(
                    design["n_pos"], design["n_neg"], t, l
                )["reuse_binding"],
            }
        )
    out_df = pd.DataFrame(ml_rows)
    if results_csv.exists():
        out_df = pd.concat([pd.read_csv(results_csv), out_df], ignore_index=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(results_csv, index=False)
    print(f"Saved ML results -> {results_csv}")


def run_design_all(keys):
    designs = {}
    for k in keys:
        print(f"\n[design] {k}")
        try:
            designs[k] = design_for_dataset(k)
            print(f"  b={designs[k]['b_used']:.4f}  chosen_t={designs[k]['chosen_t']}  t_sweep={designs[k]['t_sweep']}")
        except Exception as exc:
            print(f"  DESIGN FAILED: {exc}")
            traceback.print_exc()
    save_json(RESULTS / "all_designs.json", {k: _strip_heavy(v) for k, v in designs.items()})
    return designs


def _strip_heavy(design: dict) -> dict:
    # keep JSON lighter
    d = dict(design)
    return d


def run_split_ml(designs, keys, sweep: bool = True):
    for k in keys:
        if k not in designs:
            continue
        d = designs[k]
        t0 = d["chosen_t"]
        eff_tr = int(d.get("effective_defaults", {}).get("train_l", DEFAULT_TRAIN_L))
        eff_te = int(d.get("effective_defaults", {}).get("test_l", DEFAULT_TEST_L))
        # Default (meeting 60/15, or Concern-B-adapted) at each t sweep point
        for t in d["t_sweep"]:
            try:
                run_split_setting(
                    k, d, t=t, train_l=eff_tr, test_l=eff_te, mode="default_60_15"
                )
            except Exception as exc:
                print(f"FAILED default {k} t={t}: {exc}")
                traceback.print_exc()
        if sweep:
            # Zaniar 3x3 at a t that can actually host the upper corner under R<=1.
            # (Using max chosen_t often makes the entire grid infeasible — e.g. DCCCD t=88.)
            t_z = int(d.get("zaniar_t") or 0)
            if t_z < 3:
                # fall back: largest t_sweep point that admits at least one non-default cell
                t_z = max(d["t_sweep"])
            print(f"[zaniar] {k}: using t={t_z} for sweep grid (chosen_t was {t0})")
            for tr_l in ZANIAR_TRAIN_L:
                for te_l in ZANIAR_TEST_L:
                    if tr_l == eff_tr and te_l == eff_te and t_z == t0:
                        continue
                    sc = d["split_counts"]
                    audit_tr = audit_reuse_constraints(sc["train_pos"], sc["train_neg"], t_z, tr_l)
                    audit_te = audit_reuse_constraints(sc["test_pos"], sc["test_neg"], t_z, te_l)
                    if not audit_tr["ok_reuse"] or not audit_te["ok_reuse"]:
                        print(
                            f"[reuse-skip] {k} train_l={tr_l} test_l={te_l} t={t_z} "
                            f"reuse_tr={audit_tr['reuse_binding']:.3f} "
                            f"reuse_te={audit_te['reuse_binding']:.3f}"
                        )
                        skip_path = RESULTS / get_dataset_config(k).folder_name / "reuse_skips.csv"
                        row = pd.DataFrame(
                            [
                                {
                                    "dataset": k,
                                    "t": t_z,
                                    "train_l": tr_l,
                                    "test_l": te_l,
                                    "reuse_train_binding": audit_tr["reuse_binding"],
                                    "reuse_test_binding": audit_te["reuse_binding"],
                                    "reason": "reuse_>1_on_train_or_test",
                                }
                            ]
                        )
                        if skip_path.exists():
                            row = pd.concat([pd.read_csv(skip_path), row], ignore_index=True)
                        skip_path.parent.mkdir(parents=True, exist_ok=True)
                        row.to_csv(skip_path, index=False)
                        continue
                    try:
                        run_split_setting(
                            k, d, t=t_z, train_l=tr_l, test_l=te_l, mode="zaniar_sweep"
                        )
                    except Exception as exc:
                        print(f"FAILED sweep {k} {tr_l}/{te_l}: {exc}")
                        traceback.print_exc()


def run_full_ml(designs, keys):
    for k in keys:
        if k not in designs:
            continue
        if not DATASET_SPECS[k].get("run_full_nonsplit"):
            continue
        d = designs[k]
        t = d["chosen_t"]
        for l in DCCCD_FULL_L:
            audit = audit_reuse_constraints(d["n_pos"], d["n_neg"], t, l)
            if not audit["ok_reuse"]:
                print(f"[reuse-skip full] {k} l={l} reuse={audit['reuse_binding']:.3f}")
                continue
            try:
                run_full_nonsplit(k, d, t=t, l=l)
            except Exception as exc:
                print(f"FAILED full {k} l={l}: {exc}")
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["design", "split_ml", "full_ml", "all"],
        default="all",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[
            "credit_card_default",
            "statlog_german",
            "south_german_credit",
            "pkdd_czech",
            "polish_bankruptcy",
            "taiwan_bankruptcy",
        ],
    )
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    keys = [k for k in args.datasets if k in DATASET_SPECS]
    t0 = time.time()

    designs = {}
    if args.stage in ("design", "all", "split_ml", "full_ml"):
        # always ensure designs exist
        designs = run_design_all(keys)

    if args.stage in ("split_ml", "all"):
        run_split_ml(designs, keys, sweep=not args.no_sweep)

    if args.stage in ("full_ml", "all"):
        run_full_ml(designs, keys)

    # Aggregate
    frames = []
    for k in keys:
        folder = get_dataset_config(k).folder_name
        p = RESULTS / folder / "ml_results.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if frames:
        agg = pd.concat(frames, ignore_index=True)
        agg.to_csv(RESULTS / "all_ml_results.csv", index=False)
        print(f"\nAggregated {len(agg)} ML rows -> {RESULTS / 'all_ml_results.csv'}")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
