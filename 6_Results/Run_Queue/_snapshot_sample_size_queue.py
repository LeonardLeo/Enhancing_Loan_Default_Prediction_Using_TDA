# Sequential Snapshot_Sample_Size queue. Resume-safe via skip_existing inside each script.
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

DATASETS = [
    ("PKDD_Czech_Financial", "pkdd_czech_financial_sample_size.py"),
    ("South_German_Credit", "south_german_credit_sample_size.py"),
    ("Statlog_German_Credit_Data", "statlog_german_credit_sample_size.py"),
    ("Taiwan_Bankruptcy", "taiwan_bankruptcy_sample_size.py"),
    ("Polish_Bankruptcy_3Year", "polish_bankruptcy_3year_sample_size.py"),
    ("Default_Of_Credit_Card_Client_Data", "default_of_credit_card_client_sample_size.py"),
]
ITEMS = [
    "1_Snapshot_Count_Sweep",
    "2_Points_Per_Snapshot_Sweep",
    "3_Snapshot_Count_Across_Cloud_Sizes",
]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd):
    log("RUN " + " ".join(str(c) for c in cmd))
    code = subprocess.call(cmd, cwd=str(ROOT))
    log(f"EXIT {code} :: {' '.join(str(c) for c in cmd)}")
    return code


def main():
    failed = []
    for item in ITEMS:
        for folder, fname in DATASETS:
            script = ROOT / "5_Experiments" / "Snapshot_Sample_Size" / item / folder / fname
            code = run([str(PY), str(script)])
            if code != 0:
                failed.append(f"{item}/{folder}")
        viz = ROOT / "5_Experiments" / "Snapshot_Sample_Size" / item / "visualize_results.py"
        code = run([str(PY), str(viz)])
        if code != 0:
            failed.append(str(viz.name))
    if failed:
        log("FAILED: " + ", ".join(failed))
        return 1
    log("Snapshot sample size queue finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
