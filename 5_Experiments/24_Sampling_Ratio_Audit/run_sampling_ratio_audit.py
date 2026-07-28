import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Experiment 24 — Sampling ratio audit (n, t, l and Zaniar checklist ratios).

Uses processed datasets + the landmark percentages / file counts from the
main TDA experiments. Does not regenerate landmarks.
"""

import json
import warnings
from pathlib import Path

import pandas as pd

from utils import compute_sampling_ratio_audit, store_data_as_csv_or_json

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "6_Results" / "24_Sampling_Ratio_Audit"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    # After class balancing in TDA scripts: minority count drives both sides
    {
        "dataset": "Default_Of_Credit_Card_Client_Data",
        "source": ROOT / "1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/processed_data.xlsx",
        "target": "default payment next month",
        "positive": 1,
        "landmarks": [5, 15],
        "l": 500,
        "note": "n1=n2=minority count after undersampling (as in Exp 3)",
    },
    {
        "dataset": "Statlog_German_Credit_Data",
        "source": ROOT / "1_Data/Processed_Datasets/Statlog_German_Credit_Data/processed_data.xlsx",
        "target": "Class",
        "positive": 1,
        "landmarks": [30, 60],
        "l": 500,
        "note": "n1=n2=minority count after undersampling (as in Exp 3)",
    },
]

rows = []
all_audits = {}

for cfg in CONFIGS:
    df = pd.read_excel(cfg["source"])
    y = df[cfg["target"]]
    n_pos = int((y == cfg["positive"]).sum())
    n_neg = int((y != cfg["positive"]).sum())
    # Matching experiment practice: undersample to minority
    n1 = n2 = min(n_pos, n_neg)
    dataset_audits = {"raw_n_pos": n_pos, "raw_n_neg": n_neg, "balanced_n1": n1, "balanced_n2": n2, "landmarks": {}}

    for pct in cfg["landmarks"]:
        t = int(n1 * pct / 100)
        audit = compute_sampling_ratio_audit(
            n1=n1, n2=n2, t=t, l=cfg["l"], landmark_percent=pct
        )
        audit["dataset"] = cfg["dataset"]
        audit["note"] = cfg["note"]
        dataset_audits["landmarks"][f"L{pct}"] = audit
        rows.append(audit)
        print(
            f"{cfg['dataset']} L{pct}: t={t}, t/n1={audit['t_over_n1']:.4f}, "
            f"(t*l)/n1={audit['naive_tl_over_n1']:.2f}, "
            f"ok_naive={audit['suggested_naive_near_or_below_1']}"
        )

    all_audits[cfg["dataset"]] = dataset_audits

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT / "sampling_ratio_audit.csv", index=False)
store_data_as_csv_or_json(
    path=str(OUT),
    csv=False,
    save_as=["sampling_ratio_audit"],
    data_object=[all_audits],
)

# Suggested l so that (t*l)/n1 ≈ 1  =>  l ≈ n1/t
suggestions = []
for cfg in CONFIGS:
    n1 = all_audits[cfg["dataset"]]["balanced_n1"]
    for pct in cfg["landmarks"]:
        t = int(n1 * pct / 100)
        l_star = max(1, int(round(n1 / t))) if t else None
        suggestions.append(
            {
                "dataset": cfg["dataset"],
                "landmark": f"L{pct}",
                "t": t,
                "current_l": cfg["l"],
                "l_for_tl_over_n1_approx_1": l_star,
                "current_tl_over_n1": (t * cfg["l"]) / n1 if n1 and t else None,
            }
        )
pd.DataFrame(suggestions).to_csv(OUT / "suggested_l_values.csv", index=False)
print(f"\nSaved audit to {OUT}")
