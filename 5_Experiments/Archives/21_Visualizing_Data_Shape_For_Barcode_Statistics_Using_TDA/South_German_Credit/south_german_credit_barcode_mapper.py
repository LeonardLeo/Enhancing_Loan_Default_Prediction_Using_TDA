# -*- coding: utf-8 -*-
"""
Experiment 21 — Mapper graphs on barcode-statistic snapshots

Dataset
-------
South German Credit

Question
--------
If we treat each 24-number barcode row as a point, what shape does that
cloud have?  Mapper (Singh, Memoli, Carlsson) covers a 2D lens (PCA or
UMAP) with overlapping bins and clusters inside each bin.  The resulting
graph is a cartoon of the data shape, coloured by default vs non-default.

What this script does (in order)
--------------------------------
1. Load Experiment 3 `data_L10.csv` / `data_L20.csv` from the repo root.
2. Shuffle rows (reproducible).
3. Build a Mapper grid: resolution {20,30,40} x overlap {0.3,0.4,0.5}
   x lens {PCA, UMAP} x k-means (2 clusters).
4. Colour nodes by default status and write HTML under 6_Results.

Prerequisite: Experiment 3 `data_L*.csv`.
Results: 6_Results/Archives/21_Visualizing_Data_Shape_For_Barcode_Statistics_Using_TDA/South_German_Credit/
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import build_mapper_viz

warnings.filterwarnings("ignore")

# =============================================================================
# Dataset settings (this folder only)
# =============================================================================
FOLDER = "South_German_Credit"
PERCENTAGES = [10, 20]
TDA_DIR = ROOT / "1_Data" / "TDA_Datasets" / FOLDER / "3_PH_Default_Parameters"
SAVE_ROOT = (
    ROOT
    / "6_Results"
    / "21_Visualizing_Data_Shape_For_Barcode_Statistics_Using_TDA"
    / FOLDER
)

# =============================================================================
# Stage 1 — Mapper grid per landmark percent
# =============================================================================
for pct in PERCENTAGES:
    src = TDA_DIR / f"data_L{pct}.csv"
    if not src.exists():
        print(f"Missing Experiment 3 matrix: {src}")
        continue
    data = pd.read_csv(src).sample(frac=1, random_state=0).reset_index(drop=True)
    features = data.drop(columns="label")
    labels = data["label"]
    save_path = SAVE_ROOT / f"L{pct}"
    print(f"Mapper grid for L{pct} -> {save_path}")
    build_mapper_viz(
        data=features,
        resampled_data_label=labels,
        resolution=[20, 30, 40],
        percentage_overlap=[0.3, 0.4, 0.5],
        clustering_grid={"kmeans": [{"n_clusters": 2}]},
        lens_methods=["pca", "umap"],
        lens_params={
            "pca": {"n_components": 2},
            "umap": {"n_components": 2, "random_state": 42},
        },
        color_functions=["labels"],
        color_function_name=["Default Status"],
        output_dir=str(save_path),
        n_jobs=-1,
    )

print("Experiment 21 finished.")
