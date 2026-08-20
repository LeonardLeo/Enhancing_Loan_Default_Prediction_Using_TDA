# -*- coding: utf-8 -*-
"""
Draw the 2_Points_Per_Snapshot_Sweep figures.

Figures land in 6_Results/Snapshot_Sample_Size/2_Points_Per_Snapshot_Sweep/Visualizations/.
Run the dataset scripts in this folder first so the CSV tables exist.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from utils import ITEM_FOLDERS, export_experiment_tables, visualize_item

ITEM = "2"
summary_path = os.path.join(
    REPO_ROOT, "6_Results", "Snapshot_Sample_Size", ITEM_FOLDERS[ITEM], "all_summary.csv"
)
if not os.path.exists(summary_path):
    print("all_summary.csv missing; rebuilding item tables from shared repeats.")
    export_experiment_tables(ITEM)
written = visualize_item(ITEM)
print("Wrote", len(written), "figure(s) -> 6_Results/Snapshot_Sample_Size/" + ITEM_FOLDERS[ITEM] + "/Visualizations/")
for path in written:
    print(" ", path)
