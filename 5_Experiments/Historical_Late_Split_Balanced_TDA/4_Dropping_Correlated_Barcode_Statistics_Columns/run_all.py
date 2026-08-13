# -*- coding: utf-8 -*-
"""Run 4_Dropping_Correlated_Barcode_Statistics_Columns for every dataset in Historical_Late_Split_Balanced_TDA."""
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
        script = HERE / folder / "run.py"
        if not script.exists():
            scripts = sorted((HERE / folder).glob("*.py"))
            script = scripts[0] if scripts else None
        if script is None:
            print(f"[MISS] no script in {HERE / folder}")
            failed.append(folder)
            continue
        print("=" * 72)
        print(f"Historical_Late_Split_Balanced_TDA / 4_Dropping_Correlated_Barcode_Statistics_Columns — {folder}")
        print(script)
        print("=" * 72)
        code = subprocess.call([sys.executable, str(script)], cwd=str(script.parent))
        if code != 0:
            failed.append(folder)
            print(f"[FAIL] {folder} exit {code}")
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    print("4_Dropping_Correlated_Barcode_Statistics_Columns finished for all six datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
