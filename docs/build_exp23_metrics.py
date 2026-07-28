# -*- coding: utf-8 -*-
"""Extract Experiment 23 metrics into CSV and Markdown."""
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "6_Results" / "23_Early_Train_Test_Split"
OUT_CSV = BASE / "experiment_23_metrics.csv"
OUT_MD = ROOT / "docs" / "Experiment_23_Results.md"

rows = []
for ds, label in [
    ("Default_Of_Credit_Card_Client_Data", "Default of Credit Card Client"),
    ("Statlog_German_Credit_Data", "Statlog German Credit"),
]:
    for kind in ["default", "tuned"]:
        p = BASE / ds / f"model_results_{kind}.pkl"
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        obj = joblib.load(p)
        for data_key, models in obj.items():
            sampling = str(data_key).replace("data_", "").replace(".csv", "")
            for model_name, metrics in models.items():
                rows.append(
                    {
                        "Dataset": label,
                        "Mode": kind,
                        "Sampling": sampling,
                        "Model": model_name,
                        "Accuracy": round(float(metrics.get("accuracy", float("nan"))), 4),
                        "Precision": round(float(metrics.get("precision", float("nan"))), 4),
                        "Recall": round(float(metrics.get("recall", float("nan"))), 4),
                        "F1": round(float(metrics.get("f1_score", float("nan"))), 4),
                    }
                )

df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(df.to_string(index=False))

lines = [
    "# Experiment 23 Results — Early Train/Test Split",
    "",
    "Protocol: 80/20 stratified split on processed tabular data → PCA fit on train only → independent train/test landmarks and barcodes → train on train barcodes, evaluate on test barcodes.",
    "",
    "## Metrics",
    "",
    "| Dataset | Mode | Sampling | Model | Accuracy | Precision | Recall | F1 |",
    "|---|---|---|---|---:|---:|---:|---:|",
]
for _, r in df.iterrows():
    lines.append(
        f"| {r['Dataset']} | {r['Mode']} | {r['Sampling']} | {r['Model']} | "
        f"{r['Accuracy']} | {r['Precision']} | {r['Recall']} | {r['F1']} |"
    )
lines.extend(
    [
        "",
        "## Notes",
        "",
        "- Most Statlog models sit near 0.50 accuracy with recall ≈ 1.0 (predicting the positive class).",
        "- This differs sharply from older full-data (leaky) barcode experiments and is an important early-split finding.",
        "",
        "Artefacts: `6_Results/23_Early_Train_Test_Split/`",
        "",
    ]
)
OUT_MD.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT_MD}")
