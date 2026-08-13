# -*- coding: utf-8 -*-
"""
Experiment 17 — Do the two classes occupy different regions of barcode space?

Dataset
-------
Polish Companies Bankruptcy (3-year horizon)

Question
--------
After Experiment 3 squashes each snapshot into 24 numbers, do defaults and
non-defaults form two visible clouds, or do they sit on top of each other?
If they overlap completely, topology is not giving the classifier a
separable signal.

What this script does (in order)
--------------------------------
1. Load Experiment 3 matrices (data_L10.csv / data_L20.csv) using paths
   anchored at the repo root (safe regardless of the working directory).
2. Project those 24 columns to 2D with PCA, t-SNE, and UMAP.
3. Repeat with a kernel-density overlay so overlap is easier to see.
4. Repeat in 3D (static figures only). Rotating MP4/GIF exports are skipped:
   they need ffmpeg, take a long time, and are optional extras in the
   original Statlog script.

Prerequisite: Experiment 3 `data_L*.csv`.
Results: 6_Results/Archives/17_Distribution_For_Each_Class/Polish_Bankruptcy_3Year/
"""

# =============================================================================
# Import Libraries
# =============================================================================
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from utils import visualize_class_separability

warnings.filterwarnings("ignore")

# =============================================================================
# Dataset settings (this folder only)
# =============================================================================
FOLDER = "Polish_Bankruptcy_3Year"
PERCENTAGES = [10, 20]
TDA_DIR = ROOT / "1_Data" / "TDA_Datasets" / FOLDER / "3_PH_Default_Parameters"
SAVE_PATH = ROOT / "6_Results" / "17_Distribution_For_Each_Class" / FOLDER

data_paths = [str(TDA_DIR / f"data_L{pct}.csv") for pct in PERCENTAGES]
missing = [p for p in data_paths if not Path(p).exists()]
if missing:
    raise SystemExit(
        "Missing Experiment 3 matrices:\n  "
        + "\n  ".join(missing)
        + f"\nRun 5_Experiments/Historical_Late_Split_Balanced_TDA/1_PH_Default_Parameters/{FOLDER}/ first."
    )

# =============================================================================
# Stage 1 — 2D scatter (PCA / t-SNE / UMAP)
# =============================================================================
for method in ("tsne", "pca", "umap"):
    visualize_class_separability(
        dataset_paths=data_paths,
        method=method,
        label_column="label",
        save_path=str(SAVE_PATH),
        title="TDA Class Separability",
    )
    visualize_class_separability(
        dataset_paths=data_paths,
        method=method,
        label_column="label",
        save_path=str(SAVE_PATH / "kernel_density"),
        density_overlay=True,
        title="TDA Class Separability",
    )

# =============================================================================
# Stage 2 — Static 3D (no animation — see docstring)
# =============================================================================
for method in ("pca", "tsne", "umap"):
    visualize_class_separability(
        dataset_paths=data_paths,
        method=method,
        plot_3d=True,
        density_overlay=False,
        save_path=str(SAVE_PATH / "3D"),
        show_legend=True,
        use_color_palette=True,
        animate_3d=False,
        title="3D TDA Class Separability Using",
    )

print(f"Experiment 17 figures written under {SAVE_PATH}")
