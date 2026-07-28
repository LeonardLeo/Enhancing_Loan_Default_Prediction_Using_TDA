import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -*- coding: utf-8 -*-
"""
Experiment 25 — Snapshot mean / variance of barcode statistics.

For each existing TDA barcode matrix (from paper Exp 3 by default), record:
  - per-feature mean and variance across snapshots
  - per-class summaries
  - lambda_bar_proxy = empirical mean barcode-statistic vector
    (feature-space proxy for the landscape average \\bar\\lambda in Chazal et al.
     arXiv:1406.1901 / Frontiers TDA survey §6.3.1)

Full persistence-landscape \\bar\\lambda remains optional/costly; this experiment
locks down the statistics already available from our snapshot pipeline.
"""

import warnings
from pathlib import Path

import pandas as pd

from utils import summarize_snapshot_statistics, store_data_as_csv_or_json

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "6_Results" / "25_Snapshot_Mean_Variance"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ROOT / "1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L5.csv",
    ROOT / "1_Data/TDA_Datasets/Default_Of_Credit_Card_Client_Data/3_PH_Default_Parameters/data_L15.csv",
    ROOT / "1_Data/TDA_Datasets/Statlog_German_Credit_Data/3_PH_Default_Parameters/data_L30.csv",
    ROOT / "1_Data/TDA_Datasets/Statlog_German_Credit_Data/3_PH_Default_Parameters/data_L60.csv",
]

all_summaries = {}
flat_rows = []

for path in SOURCES:
    if not path.exists():
        print(f"⚠️ Missing (skip): {path}")
        continue
    key = f"{path.parent.parent.name}/{path.parent.name}/{path.name}"
    summary = summarize_snapshot_statistics(str(path))
    all_summaries[key] = summary

    for feat, mean_v in summary["global_mean"].items():
        flat_rows.append(
            {
                "source": key,
                "feature": feat,
                "mean": mean_v,
                "variance": summary["global_variance"][feat],
                "n_snapshots": summary["n_snapshots"],
            }
        )
    print(f"✅ {key}: n={summary['n_snapshots']} features={len(summary['feature_columns'])}")

if flat_rows:
    pd.DataFrame(flat_rows).to_csv(OUT / "snapshot_mean_variance.csv", index=False)

store_data_as_csv_or_json(
    path=str(OUT),
    csv=False,
    save_as=["snapshot_mean_variance_full"],
    data_object=[all_summaries],
)
print(f"\nSaved to {OUT}")
