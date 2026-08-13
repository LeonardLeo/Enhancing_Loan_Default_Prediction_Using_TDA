# -*- coding: utf-8 -*-
"""
Experiment 28 — Revised Snapshot Protocol
Dataset: {title}

This is *not* another copy of the persistent-homology engine.  The meeting
rules (no undersampling, fixed t, train/test l = 60/15, reuse reported
separately from the formula) live in:

    5_Experiments/28_Revised_Snapshot_Protocol/protocol_lib.py
    5_Experiments/28_Revised_Snapshot_Protocol/run_protocol.py

This folder script is the dataset-specific entry point so the layout
matches every other experiment:

    5_Experiments/28_.../{folder}/{stem}_protocol.py
    6_Results/28_.../{folder}/

Stages run (in order)
---------------------
1. design   — estimate intrinsic dimension, pick a joint t, report reuse
2. split_ml — early train/test split, independent snapshots, fit models
3. full_ml  — DCCCD-only non-split arm (skipped for this dataset unless
              run_protocol is told otherwise)

Deep write-up: docs/Revised_Snapshot_Protocol_Deep_Report.md
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DATASET_KEY = "{key}"
FOLDER = "{folder}"
STAGE = "all"  # design | split_ml | full_ml | all
SCRIPT = Path(__file__).resolve().parents[1] / "run_protocol.py"

print("=" * 72)
print(f"Experiment 28 — {FOLDER}")
print(f"Dataset key : {DATASET_KEY}")
print(f"Stage       : {STAGE}")
print(f"Engine      : {SCRIPT}")
print("=" * 72)

if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(SCRIPT),
                "--datasets",
                DATASET_KEY,
                "--stage",
                STAGE,
            ],
            cwd=str(ROOT),
        )
    )
