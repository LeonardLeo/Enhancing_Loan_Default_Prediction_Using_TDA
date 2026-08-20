# -*- coding: utf-8 -*-
"""Readable shared-pool builder for the snapshot sample-size study.

Items 1, 2, and 4 consume this grid. Per-dataset copies live under
0_Shared_Pools/<Dataset>/ and show the same steps with that table's knobs.

    .\tda_env\Scripts\python.exe 5_Experiments/Snapshot_Sample_Size/build_shared_pools.py
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_shared.py")), run_name="__main__")
