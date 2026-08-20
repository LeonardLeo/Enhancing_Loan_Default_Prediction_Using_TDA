# -*- coding: utf-8 -*-
"""Generate barcodes once, evaluate the nested grid, export items 1/2/4.

Item 3 is this sample-size study — not a third independent grid.

Run from repo root:

    .\\tda_env\\Scripts\\python.exe 5_Experiments/Snapshot_Sample_Size/run_shared.py
    .\\tda_env\\Scripts\\python.exe 5_Experiments/Snapshot_Sample_Size/run_shared.py --stage design
    .\\tda_env\\Scripts\\python.exe 5_Experiments/Snapshot_Sample_Size/run_shared.py --stage generate --datasets pkdd_czech --protocol Early_Split_TDA
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from sample_size_lib import (  # noqa: E402
    DATASET_RUN_ORDER,
    PROTOCOLS,
    export_all_experiment_tables,
    parse_cli_list,
    run_shared_grid,
    write_master_design_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot sample-size shared grid (items 1, 2, and 4)."
    )
    parser.add_argument(
        "--stage",
        choices=["design", "generate", "export", "all"],
        default="all",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--protocol", dest="protocols", nargs="*", default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    datasets = parse_cli_list(args.datasets, DATASET_RUN_ORDER)
    protocols = parse_cli_list(args.protocols, PROTOCOLS.keys())
    skip_existing = not args.no_skip_existing

    if args.stage in ("design", "all", "generate"):
        grid = write_master_design_table(datasets=datasets, protocols=protocols)
        print(grid.to_string(index=False))

    if args.stage in ("generate", "all"):
        run_shared_grid(
            datasets=datasets,
            protocols=protocols,
            skip_existing=skip_existing,
            write_exports=True,
        )

    if args.stage in ("export", "all"):
        export_all_experiment_tables()
        print("Exported item 1 / 2 / 4 tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
