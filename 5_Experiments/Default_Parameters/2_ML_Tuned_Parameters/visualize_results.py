# -*- coding: utf-8 -*-
"""Visualize artefacts for Default_Parameters / 2_ML_Tuned_Parameters.

Runnable from this experiment directory:

    python visualize_results.py

Run this script; figures land in `6_Results/Default_Parameters/2_ML_Tuned_Parameters/Visualizations/`.
If results do not exist yet, the script exits with a clear
"results not generated yet" message naming the expected path.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import ResultsNotGeneratedError, visualize_experiment_folder

if __name__ == "__main__":
    try:
        visualize_experiment_folder(
            protocol_bucket='Default_Parameters',
            experiment='2_ML_Tuned_Parameters',
        )
    except ResultsNotGeneratedError as exc:
        raise SystemExit(str(exc)) from exc
