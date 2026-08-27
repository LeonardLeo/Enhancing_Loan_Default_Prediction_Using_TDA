# Audit barcode-statistic columns against the statistic set each experiment requires.
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "_audit_barcode_statistics.json"
import sys
sys.path.insert(0, str(ROOT))
from utils import ACTIVE_TDA_PROTOCOL_BUCKETS, TDA_PROCESS_REGISTRY

BUCKETS = ACTIVE_TDA_PROTOCOL_BUCKETS

EXPECTED_H0 = [f"g{i}_0" for i in range(1, 13)]
EXPECTED_H1 = [f"g{i}_1" for i in range(1, 13)]

G_H0 = re.compile(r"^g(\d{1,2})_0$")
G_H1 = re.compile(r"^g(\d{1,2})_1$")


def win(path: Path) -> str:
    raw = os.path.abspath(str(path))
    if os.name == "nt" and not raw.startswith("\\\\?\\"):
        return "\\\\?\\" + raw
    return raw


def classify_columns(columns):
    h0, h1, other = [], [], []
    has_label = False
    for col in columns:
        name = str(col)
        if name == "label":
            has_label = True
            continue
        if G_H0.fullmatch(name) or "(Dim 0)" in name:
            h0.append(name)
        elif G_H1.fullmatch(name) or "(Dim 1)" in name:
            h1.append(name)
        else:
            other.append(name)
    return {
        "n_h0": len(h0),
        "n_h1": len(h1),
        "n_other": len(other),
        "has_label": has_label,
        "n_features": len(h0) + len(h1) + len(other),
        "h0": h0,
        "h1": h1,
        "other": other,
    }


def expected_profile(bucket: str, experiment: str):
    spec = TDA_PROCESS_REGISTRY.get(bucket) or {}
    if experiment == "1_PH_Default_Parameters":
        return "h0_only" if spec.get("homology") == "H0" else "full_24"
    if experiment in {"3_H0_Only", "5_Linear_Regression_For_Prediction"}:
        return "h0_only"
    if experiment == "4_Dropping_Correlated_Barcode_Statistics_Columns":
        return "decorrelated_subset"
    if experiment == "9_Revised_Snapshot_Protocol":
        return "full_24"
    return "unknown"


def verdict_for(profile: str, info: dict) -> str:
    n0, n1, n_other = info["n_h0"], info["n_h1"], info["n_other"]
    if profile == "full_24":
        if n0 == 12 and n1 == 12 and n_other == 0 and info["has_label"]:
            return "pass"
        if n0 == 12 and n1 == 0:
            return "fail_h0_only_but_needed_h0_h1"
        return "fail_unexpected_columns"
    if profile == "h0_only":
        if n0 == 12 and n1 == 0 and n_other == 0 and info["has_label"]:
            return "pass"
        if n1 > 0:
            return "fail_still_has_h1"
        return "fail_unexpected_columns"
    if profile == "decorrelated_subset":
        total = n0 + n1 + n_other
        if not info["has_label"]:
            return "fail_missing_label"
        if total >= 24 and n0 == 12 and n1 == 12:
            return "fail_nothing_dropped"
        if total == 12 and n1 == 0 and n0 == 12:
            return "warn_looks_like_h0_only"
        if total < 24 and total >= 1:
            return "pass"
        return "fail_unexpected_columns"
    return "skip"


def h1_all_zero(frame: pd.DataFrame, h1_cols) -> bool:
    if not h1_cols:
        return False
    numeric = frame[h1_cols].apply(pd.to_numeric, errors="coerce")
    return bool((numeric.fillna(0) == 0).all().all())


def inspect_csv(path: Path) -> dict:
    frame = pd.read_csv(win(path), nrows=None)
    info = classify_columns(list(frame.columns))
    info["n_rows"] = int(len(frame))
    info["h1_all_zero"] = h1_all_zero(frame, info["h1"]) if info["h1"] else False
    missing_h0 = [c for c in EXPECTED_H0 if c not in frame.columns]
    missing_h1 = [c for c in EXPECTED_H1 if c not in frame.columns]
    # Descriptive-name tables will report all g-codes missing; that is OK.
    info["missing_g_h0"] = missing_h0
    info["missing_g_h1"] = missing_h1
    return info


def strip_long(path: Path | str) -> Path:
    text = str(path)
    if text.startswith("\\\\?\\"):
        text = text[4:]
    return Path(text)


def rel(path: Path) -> str:
    path = strip_long(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


rows = []
tda_root = ROOT / "1_Data" / "TDA_Datasets"
for dirpath, dirnames, filenames in os.walk(str(tda_root)):
    parts = Path(dirpath).parts
    if "Archives" in parts:
        continue
    for name in filenames:
        if not name.endswith(".csv") or not name.startswith("data_"):
            continue
        full = Path(dirpath) / name
        rel_parts = Path(os.path.relpath(str(full), str(tda_root))).parts
        if len(rel_parts) < 3:
            continue
        bucket, experiment, dataset = rel_parts[0], rel_parts[1], rel_parts[2]
        if bucket not in BUCKETS:
            continue
        try:
            info = inspect_csv(full)
        except Exception as exc:
            rows.append({
                "kind": "tda_dataset",
                "bucket": bucket,
                "experiment": experiment,
                "dataset": dataset,
                "file": rel(full),
                "verdict": "fail_unreadable",
                "error": str(exc),
            })
            continue
        profile = expected_profile(bucket, experiment)
        verdict = verdict_for(profile, info)
        rows.append({
            "kind": "tda_dataset",
            "bucket": bucket,
            "experiment": experiment,
            "dataset": dataset,
            "file": rel(full),
            "basename": name,
            "profile": profile,
            "verdict": verdict,
            "n_rows": info["n_rows"],
            "n_h0": info["n_h0"],
            "n_h1": info["n_h1"],
            "n_other": info["n_other"],
            "n_features": info["n_features"],
            "has_label": info["has_label"],
            "h1_all_zero": info["h1_all_zero"],
            "other": info["other"][:8],
        })

barcode_rows = []
barcode_root = ROOT / "1_Data" / "Barcode_Statistics"
if barcode_root.exists():
    for dirpath, dirnames, filenames in os.walk(str(barcode_root)):
        parts = Path(dirpath).parts
        if "Archives" in parts:
            continue
        for name in filenames:
            if not name.endswith(".csv") or not name.startswith("barcode_stats_"):
                continue
            full = Path(dirpath) / name
            rel_parts = Path(os.path.relpath(str(full), str(barcode_root))).parts
            if len(rel_parts) < 3:
                continue
            bucket, experiment, dataset = rel_parts[0], rel_parts[1], rel_parts[2]
            if bucket not in BUCKETS:
                continue
            try:
                info = inspect_csv(full)
            except Exception as exc:
                barcode_rows.append({
                    "kind": "barcode_stats",
                    "bucket": bucket,
                    "experiment": experiment,
                    "dataset": dataset,
                    "file": rel(full),
                    "verdict": "fail_unreadable",
                    "error": str(exc),
                })
                continue
            profile = "full_24"
            verdict = verdict_for(profile, info)
            barcode_rows.append({
                "kind": "barcode_stats",
                "bucket": bucket,
                "experiment": experiment,
                "dataset": dataset,
                "file": rel(full),
                "basename": name,
                "profile": profile,
                "verdict": verdict,
                "n_rows": info["n_rows"],
                "n_h0": info["n_h0"],
                "n_h1": info["n_h1"],
                "n_other": info["n_other"],
                "n_features": info["n_features"],
                "has_label": info["has_label"],
                "h1_all_zero": info["h1_all_zero"],
            })

exp7_rows = []
results_root = ROOT / "6_Results"
for bucket in BUCKETS:
    for dataset_dir in (results_root / bucket / "7_Snapshot_Mean_Variance").glob("*"):
        csv_path = dataset_dir / "snapshot_mean_variance.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(win(csv_path))
        feature_col = None
        for candidate in ("feature", "statistic", "column", "name"):
            if candidate in frame.columns:
                feature_col = candidate
                break
        if feature_col is None:
            feature_col = frame.columns[0]
        features = [str(v) for v in frame[feature_col].unique()]
        info = classify_columns(features)
        # these files are long-form; label may be absent
        n0, n1 = info["n_h0"], info["n_h1"]
        if n0 == 0 and n1 == 0:
            # feature names may be stored without dim suffix; count unique stats another way
            verdict = "warn_could_not_parse_features"
        elif n0 == 12 and n1 == 12:
            verdict = "pass"
        elif n0 == 12 and n1 == 0:
            verdict = "fail_h0_only_but_needed_h0_h1"
        else:
            verdict = "fail_unexpected_columns"
        exp7_rows.append({
            "kind": "exp7_summary",
            "bucket": bucket,
            "experiment": "7_Snapshot_Mean_Variance",
            "dataset": dataset_dir.name,
            "file": rel(csv_path),
            "verdict": verdict,
            "n_h0": n0,
            "n_h1": n1,
            "n_other": info["n_other"],
            "n_unique_features": len(features),
            "feature_col": feature_col,
        })

# Script-level source check: consumers should read Exp 1, not Exp 3.
script_rows = []
exp_root = ROOT / "5_Experiments"
source_re = re.compile(r'SOURCE_EXPERIMENT\s*=\s*"([^"]+)"')
homology_re = re.compile(r"HOMOLOGY_DIM\s*=\s*(\d+)")
h0_filter_re = re.compile(r'endswith\("_0"\)|\(Dim 0\)')
for bucket in BUCKETS:
    for experiment in (
        "1_PH_Default_Parameters",
        "2_PH_Tuned_Parameters",
        "3_H0_Only",
        "4_Dropping_Correlated_Barcode_Statistics_Columns",
        "5_Linear_Regression_For_Prediction",
        "7_Snapshot_Mean_Variance",
        "8_Null_Hypothesis_Algorithm2",
        "9_Revised_Snapshot_Protocol",
    ):
        folder = exp_root / bucket / experiment
        if not folder.exists():
            continue
        for py in folder.rglob("*.py"):
            if py.name in {"visualize_results.py", "run.py", "run_all.py"}:
                continue
            if "Archives" in py.parts:
                continue
            script_path = win(py)
            if not os.path.exists(script_path):
                script_rows.append({
                    "kind": "script",
                    "bucket": bucket,
                    "experiment": experiment,
                    "file": rel(py),
                    "verdict": "fail_unreadable",
                    "issues": ["path_too_long_or_missing"],
                })
                continue
            with open(script_path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
            sources = source_re.findall(text)
            dims = homology_re.findall(text)
            has_h0_filter = bool(h0_filter_re.search(text))
            expected_source = None
            if experiment in {
                "2_PH_Tuned_Parameters",
                "3_H0_Only",
                "4_Dropping_Correlated_Barcode_Statistics_Columns",
                "5_Linear_Regression_For_Prediction",
                "7_Snapshot_Mean_Variance",
                "8_Null_Hypothesis_Algorithm2",
            }:
                expected_source = "1_PH_Default_Parameters"
            issues = []
            if expected_source and sources and any(s != expected_source for s in sources):
                issues.append("wrong_source_experiment")
            if experiment == "1_PH_Default_Parameters" and dims and any(d != "2" for d in dims):
                issues.append("homology_dim_not_2")
            if experiment in {"3_H0_Only", "5_Linear_Regression_For_Prediction"} and not has_h0_filter:
                issues.append("missing_h0_filter")
            if experiment in {"1_PH_Default_Parameters", "2_PH_Tuned_Parameters", "7_Snapshot_Mean_Variance", "8_Null_Hypothesis_Algorithm2"} and has_h0_filter:
                issues.append("unexpected_h0_filter")
            if experiment == "4_Dropping_Correlated_Barcode_Statistics_Columns" and has_h0_filter:
                issues.append("h0_filter_on_decorrelate")
            script_rows.append({
                "kind": "script",
                "bucket": bucket,
                "experiment": experiment,
                "file": rel(py),
                "source_experiment": sources[0] if sources else None,
                "homology_dim": dims[0] if dims else None,
                "has_h0_filter": has_h0_filter,
                "verdict": "pass" if not issues else "fail_" + ",".join(issues),
                "issues": issues,
            })

all_rows = rows + barcode_rows + exp7_rows + script_rows


def tally(items):
    counts = defaultdict(int)
    for item in items:
        counts[item.get("verdict", "unknown")] += 1
    return dict(counts)


summary = {
    "tda_datasets": {"n": len(rows), "verdicts": tally(rows)},
    "barcode_stats": {"n": len(barcode_rows), "verdicts": tally(barcode_rows)},
    "exp7": {"n": len(exp7_rows), "verdicts": tally(exp7_rows)},
    "scripts": {"n": len(script_rows), "verdicts": tally(script_rows)},
}

by_experiment = defaultdict(lambda: defaultdict(int))
for item in rows:
    by_experiment[item["experiment"]][item["verdict"]] += 1

payload = {
    "summary": summary,
    "by_experiment_tda": {k: dict(v) for k, v in by_experiment.items()},
    "failures": [r for r in all_rows if str(r.get("verdict", "")).startswith("fail")],
    "warnings": [r for r in all_rows if str(r.get("verdict", "")).startswith("warn")],
    "tda_rows": rows,
    "barcode_rows": barcode_rows,
    "exp7_rows": exp7_rows,
    "script_rows": script_rows,
}

OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("failures", len(payload["failures"]))
print("warnings", len(payload["warnings"]))
print("wrote", OUT)
