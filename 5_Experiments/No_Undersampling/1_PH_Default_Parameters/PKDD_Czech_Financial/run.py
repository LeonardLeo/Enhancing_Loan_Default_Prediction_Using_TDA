# -*- coding: utf-8 -*-
"""
No_Undersampling / 1_PH_Default_Parameters
Dataset: PKDD_Czech_Financial

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
        dataset_key='pkdd_czech',
        protocol_bucket='No_Undersampling',
        experiment='1_PH_Default_Parameters',
        skip_existing_barcodes=True,
    )
