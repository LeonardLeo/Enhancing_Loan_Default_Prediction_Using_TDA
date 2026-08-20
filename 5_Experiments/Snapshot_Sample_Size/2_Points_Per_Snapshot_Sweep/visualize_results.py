# -*- coding: utf-8 -*-
"""Visualize Snapshot_Sample_Size / 2_Points_Per_Snapshot_Sweep.

Figures land in 6_Results/Snapshot_Sample_Size/2_Points_Per_Snapshot_Sweep/Visualizations/.
If the shared grid has not been run, the script exits with
"results not generated yet" and the expected path.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

BUCKET = Path(__file__).resolve().parents[1]
ROOT = BUCKET.parents[1]
sys.path.insert(0, str(BUCKET))
sys.path.insert(0, str(ROOT))

from sample_size_lib import ITEM_FOLDERS, export_experiment_tables, visualize_item  # noqa: E402

ITEM = "2"


if __name__ == "__main__":
    try:
        export_experiment_tables(ITEM)
        written = visualize_item(ITEM)
    except FileNotFoundError as exc:
        raise SystemExit(f"results not generated yet: {exc}") from exc
    print(f"Wrote {len(written)} figure(s) -> 6_Results/Snapshot_Sample_Size/{ITEM_FOLDERS[ITEM]}/Visualizations/")
    for path in written:
        print(f"  {path}")
