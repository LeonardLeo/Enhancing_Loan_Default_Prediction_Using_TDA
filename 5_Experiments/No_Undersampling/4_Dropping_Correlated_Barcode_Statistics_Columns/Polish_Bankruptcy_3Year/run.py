# -*- coding: utf-8 -*-
"""
No_Undersampling / 4_Dropping_Correlated_Barcode_Statistics_Columns
Dataset: Polish_Bankruptcy_3Year

Protocol knobs and artefact paths are resolved in utils.run_protocol_experiment.
This file is a complete, runnable entry point — not a stub.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import run_protocol_experiment

if __name__ == "__main__":
    run_protocol_experiment(
        dataset_key='polish_bankruptcy',
        protocol_bucket='No_Undersampling',
        experiment='4_Dropping_Correlated_Barcode_Statistics_Columns',
        skip_existing_barcodes=True,
    )
