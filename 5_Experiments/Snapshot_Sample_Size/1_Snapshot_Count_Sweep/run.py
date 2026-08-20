# -*- coding: utf-8 -*-
"""Item 1 — Number of snapshots on the x-axis; each cloud uses the dataset-aware
default point count (held fixed). Not item 2 (which holds 60 snapshots and
moves points per snapshot).

This experiment is a view of the shared sample-size grid. Item 3 is the
study made of items 1, 2, and 4; it is not a third independent grid.

    .\\tda_env\\Scripts\\python.exe 5_Experiments/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/run.py
    .\\tda_env\\Scripts\\python.exe 5_Experiments/Snapshot_Sample_Size/1_Snapshot_Count_Sweep/run.py --protocol Early_Split_TDA --datasets pkdd_czech
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUCKET = HERE.parent
REPO_ROOT = BUCKET.parents[1]
sys.path.insert(0, str(BUCKET))
sys.path.insert(0, str(REPO_ROOT))

from sample_size_lib import (  # noqa: E402
    DATASET_RUN_ORDER,
    ITEM_FOLDERS,
    PROTOCOLS,
    export_experiment_tables,
    parse_cli_list,
    run_shared_grid,
    write_master_design_table,
)

ITEM = "1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["design", "generate", "export", "all"], default="all")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--protocol", dest="protocols", nargs="*", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()
    datasets = parse_cli_list(args.datasets, DATASET_RUN_ORDER)
    protocols = parse_cli_list(args.protocols, PROTOCOLS.keys())
    skip_existing = not args.no_skip_existing
    if args.stage in ("design", "all", "generate"):
        write_master_design_table(datasets=datasets, protocols=protocols)
    if args.stage in ("generate", "all"):
        run_shared_grid(
            datasets=datasets,
            protocols=protocols,
            skip_existing=skip_existing,
            write_exports=True,
        )
    if args.stage in ("export", "all"):
        export_experiment_tables(ITEM)
        print(f"Exported {ITEM_FOLDERS[ITEM]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
