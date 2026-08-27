"""Compare Exp 1 barcode VALUES across the four protocol arms."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "_compare_barcode_values_across_arms.json"

ARMS = [
    ("Early_Split_And_Undersample_H0_And_H1", "Early split and undersample, using both H0 and H1", True),
    ("Early_Split_No_Undersample_H0_And_H1", "Early split, no undersample, using both H0 and H1", True),
    ("Late_Split_And_Undersample_H0_And_H1", "Late split and undersample (the original historical run), using both H0 and H1", False),
    ("Late_Split_No_Undersample_H0_And_H1", "Late split, no undersample, using both H0 and H1", False),
]

DATASETS = {
    "Default_Of_Credit_Card_Client_Data": [5, 15],
    "Statlog_German_Credit_Data": [30, 60],
    "PKDD_Czech_Financial": [10, 20],
    "Polish_Bankruptcy_3Year": [10, 20],
    "Taiwan_Bankruptcy": [10, 20],
    "South_German_Credit": [10, 20],
}

HEADLINE = ["g2_0", "g3_0", "g2_1", "g3_1"]  # mean death / mean persistence, H0 and H1


def win(path: Path) -> str:
    raw = os.path.abspath(str(path))
    if os.name == "nt" and not raw.startswith("\\\\?\\"):
        return "\\\\?\\" + raw
    return raw


def load_exp1(bucket: str, dataset: str, percent: int, early: bool) -> pd.DataFrame | None:
    base = ROOT / "1_Data" / "TDA_Datasets" / bucket / "1_PH_Default_Parameters" / dataset
    token = str(percent)
    if early:
        train = base / "train" / f"data_L{token}.csv"
        test = base / "test" / f"data_L{token}.csv"
        if os.path.exists(win(train)) and os.path.exists(win(test)):
            a = pd.read_csv(win(train))
            b = pd.read_csv(win(test))
            a["__split"] = "train"
            b["__split"] = "test"
            return pd.concat([a, b], ignore_index=True)
        return None
    path = base / f"data_L{token}.csv"
    if os.path.exists(win(path)):
        frame = pd.read_csv(win(path))
        frame["__split"] = "late"
        return frame
    return None


def file_digest(frame: pd.DataFrame) -> str:
    cols = [c for c in frame.columns if c != "__split"]
    payload = frame[cols].to_csv(index=False).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


def summarise(frame: pd.DataFrame) -> dict:
    feats = [c for c in frame.columns if c not in {"label", "__split"}]
    labels = frame["label"].value_counts().to_dict() if "label" in frame.columns else {}
    n_default = int(labels.get(1, labels.get(1.0, 0)))
    n_non = int(labels.get(0, labels.get(0.0, 0)))
    means = {}
    for col in HEADLINE:
        if col in frame.columns:
            means[col] = float(pd.to_numeric(frame[col], errors="coerce").mean())
        else:
            means[col] = None
    vector = frame[HEADLINE].apply(pd.to_numeric, errors="coerce").mean().to_numpy(dtype=float) if all(
        c in frame.columns for c in HEADLINE
    ) else None
    return {
        "n_rows": int(len(frame)),
        "n_default": n_default,
        "n_nondefault": n_non,
        "n_features": len(feats),
        "means": means,
        "vector": vector.tolist() if vector is not None else None,
        "digest": file_digest(frame),
        "n_train": int((frame["__split"] == "train").sum()) if "__split" in frame.columns else None,
        "n_test": int((frame["__split"] == "test").sum()) if "__split" in frame.columns else None,
    }


records = []
for dataset, percents in DATASETS.items():
    for percent in percents:
        loaded = {}
        for bucket, label, early in ARMS:
            frame = load_exp1(bucket, dataset, percent, early)
            loaded[bucket] = None if frame is None else summarise(frame)
        # pairwise distance between mean vectors
        buckets = [a[0] for a in ARMS]
        pairwise = {}
        identical = []
        for i, left in enumerate(buckets):
            for right in buckets[i + 1 :]:
                a = loaded[left]
                b = loaded[right]
                key = f"{left} vs {right}"
                if a is None or b is None or a["vector"] is None or b["vector"] is None:
                    pairwise[key] = None
                    continue
                dist = float(np.linalg.norm(np.array(a["vector"]) - np.array(b["vector"])))
                pairwise[key] = dist
                if a["digest"] == b["digest"]:
                    identical.append(key)
        records.append({
            "dataset": dataset,
            "percent": percent,
            "arms": {
                bucket: {
                    "label": label,
                    **(loaded[bucket] or {"missing": True}),
                }
                for bucket, label, _ in ARMS
            },
            "pairwise_mean_l2": pairwise,
            "identical_tables": identical,
        })

# compact headline table
headline_rows = []
for rec in records:
    row = {"dataset": rec["dataset"], "L": rec["percent"]}
    for bucket, label, _ in ARMS:
        arm = rec["arms"][bucket]
        if arm.get("missing"):
            row[label] = None
            continue
        row[label] = {
            "n": arm["n_rows"],
            "n_default": arm["n_default"],
            "n_nondefault": arm["n_nondefault"],
            "g2_0": arm["means"].get("g2_0"),
            "g3_1": arm["means"].get("g3_1"),
        }
    dists = [v for v in rec["pairwise_mean_l2"].values() if v is not None]
    row["min_arm_distance"] = min(dists) if dists else None
    row["max_arm_distance"] = max(dists) if dists else None
    row["any_identical"] = bool(rec["identical_tables"])
    headline_rows.append(row)

payload = {
    "headline": headline_rows,
    "records": records,
    "any_identical_pair": any(r["identical_tables"] for r in records),
}
OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("any identical pair across arms:", payload["any_identical_pair"])
print()
for row in headline_rows:
    print(f"{row['dataset']} L{row['L']}  identical={row['any_identical']}  minL2={row['min_arm_distance']:.6g}  maxL2={row['max_arm_distance']:.6g}")
    for bucket, label, _ in ARMS:
        cell = row[label]
        if cell is None:
            print(f"  MISSING {label}")
            continue
        print(
            f"  {label:28} n={cell['n']:5}  def={cell['n_default']:5} non={cell['n_nondefault']:5}  "
            f"mean_death_H0={cell['g2_0']:.5f}  mean_pers_H1={cell['g3_1']:.5f}"
        )
print("wrote", OUT)
