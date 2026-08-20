# Sequential Snapshot_Sample_Size queue. Resume-safe via skip_existing.

# Separate from _ripser_queue.py / _consumer_queue.py so an in-flight
# historical queue is not interleaved destructively.

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "tda_env" / "Scripts" / "python.exe"
LOG = HERE / "_snapshot_sample_size_queue.log"
ENGINE = ROOT / "5_Experiments" / "Snapshot_Sample_Size" / "run_shared.py"
VIZ = [
    ROOT / "5_Experiments" / "Snapshot_Sample_Size" / "1_Snapshot_Count_Sweep" / "visualize_results.py",
    ROOT / "5_Experiments" / "Snapshot_Sample_Size" / "2_Points_Per_Snapshot_Sweep" / "visualize_results.py",
    ROOT / "5_Experiments" / "Snapshot_Sample_Size" / "3_Snapshot_Count_Across_Cloud_Sizes" / "visualize_results.py",
]

DATASETS = [
    "pkdd_czech",
    "south_german_credit",
    "statlog_german",
    "taiwan_bankruptcy",
    "polish_bankruptcy",
    "credit_card_default",
]
PROTOCOLS = [
    "Historical_Late_Split_Balanced_TDA",
    "Early_Split_TDA",
    "No_Undersampling",
    "Early_Split_TDA_And_No_Undersampling",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd) -> int:
    log("RUN " + " ".join(str(c) for c in cmd))
    code = subprocess.call(cmd, cwd=str(ROOT))
    log(f"EXIT {code} :: {' '.join(str(c) for c in cmd)}")
    return code


def main() -> int:
    failed = []
    run([str(PY), str(ENGINE), "--stage", "design"])
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            code = run(
                [
                    str(PY),
                    str(ENGINE),
                    "--stage",
                    "generate",
                    "--datasets",
                    dataset,
                    "--protocol",
                    protocol,
                ]
            )
            if code != 0:
                failed.append(f"{protocol}/{dataset}")
    run([str(PY), str(ENGINE), "--stage", "export"])
    for script in VIZ:
        code = run([str(PY), str(script)])
        if code != 0:
            failed.append(str(script.name))
    if failed:
        log("QUEUE FINISHED WITH FAILURES: " + ", ".join(failed))
        return 1
    log("QUEUE FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
