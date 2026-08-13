# -*- coding: utf-8 -*-
"""
Experiment 25 — run every dataset folder.

Debugging: open the script inside the dataset folder, not this file.

    5_Experiments/25_Snapshot_Mean_Variance/<Dataset>/run_snapshot_mean_variance.py

Needs Experiment 3 `data_L*.csv` for that dataset.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = [
    "Default_Of_Credit_Card_Client_Data",
    "Statlog_German_Credit_Data",
    "PKDD_Czech_Financial",
    "Polish_Bankruptcy_3Year",
    "Taiwan_Bankruptcy",
    "South_German_Credit",
]


def main() -> int:
    failed = []
    for folder in DATASETS:
        script = HERE / folder / "run_snapshot_mean_variance.py"
        print("=" * 72)
        print(f"Experiment 25 — {folder}")
        print(script)
        print("=" * 72)
        code = subprocess.call([sys.executable, str(script)], cwd=str(script.parent))
        if code != 0:
            failed.append(folder)
            print(f"[FAIL] {folder} exit {code}")
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    print("Experiment 25 finished for all six datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
