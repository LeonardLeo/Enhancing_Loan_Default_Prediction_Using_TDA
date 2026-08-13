# -*- coding: utf-8 -*-
"""
Experiment 9 — Revised Snapshot Protocol
Dataset: Default of Credit Card Clients (UCI)

This folder is the dataset-specific entry point. The protocol itself
(no undersampling, fixed t, train/test snapshot counts 60/15, formula
versus reuse reported separately) is implemented once in:

    protocol_lib.py   — sampling, overlap, models
    run_protocol.py   — stages: design, split_ml, full_ml, all

Stages (in order)
-----------------
1. design   estimate intrinsic dimension, choose a joint t, print reuse
2. split_ml early train/test split, independent snapshots, fit models
3. full_ml  DCCCD-only non-split arm (run_protocol skips it unless needed)

Outputs land under 6_Results/Early_Split_TDA/9_Revised_Snapshot_Protocol/Default_Of_Credit_Card_Client_Data/
Narrative: docs/Revised_Snapshot_Protocol_Deep_Report.md
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

DATASET_KEY = "credit_card_default"
FOLDER = "Default_Of_Credit_Card_Client_Data"
STAGE = "all"  # design | split_ml | full_ml | all
SCRIPT = Path(__file__).resolve().parents[4] / "run_protocol.py"

print("=" * 72)
print(f"Experiment 9 — {FOLDER}")
print(f"Dataset key : {DATASET_KEY}")
print(f"Stage       : {STAGE}")
print(f"Engine      : {SCRIPT}")
print("=" * 72)

if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(SCRIPT), "--datasets", DATASET_KEY, "--stage", STAGE],
            cwd=str(ROOT),
        )
    )
