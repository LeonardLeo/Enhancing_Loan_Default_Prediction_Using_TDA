# -*- coding: utf-8 -*-
"""
No_Undersampling / 5_Linear_Regression_For_Prediction
Dataset: South_German_Credit

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
        dataset_key='south_german_credit',
        protocol_bucket='No_Undersampling',
        experiment='5_Linear_Regression_For_Prediction',
        skip_existing_barcodes=True,
    )
