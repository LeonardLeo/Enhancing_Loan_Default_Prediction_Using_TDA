# -*- coding: utf-8 -*-
"""Run 2_PH_Tuned_Parameters for every dataset in Early_Split_TDA_And_No_Undersampling."""
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
        print(f"Early_Split_TDA_And_No_Undersampling / 2_PH_Tuned_Parameters — {folder}")
        print(script)
        print("=" * 72)
        code = subprocess.call([sys.executable, str(script)], cwd=str(script.parent))
        if code != 0:
            failed.append(folder)
            print(f"[FAIL] {folder} exit {code}")
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    print("2_PH_Tuned_Parameters finished for all six datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
