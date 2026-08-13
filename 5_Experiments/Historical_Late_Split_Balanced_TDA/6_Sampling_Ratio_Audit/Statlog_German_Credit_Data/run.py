# -*- coding: utf-8 -*-
"""
Historical_Late_Split_Balanced_TDA / 6_Sampling_Ratio_Audit
Dataset: Statlog_German_Credit_Data

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
        dataset_key='statlog_german',
        protocol_bucket='Historical_Late_Split_Balanced_TDA',
        experiment='6_Sampling_Ratio_Audit',
        skip_existing_barcodes=True,
    )
