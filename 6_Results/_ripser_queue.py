# Sequential Ripser / protocol queue. Resume-safe: Exp 1 skip_existing; Exp 9 skip by ml_results run_key.
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).name == "_ripser_queue.py" else Path.cwd()
PY = ROOT / "tda_env" / "Scripts" / "python.exe"
LOG = ROOT / "6_Results" / "_ripser_queue.log"

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def run(cmd, cwd=None):
    log("RUN " + " ".join(str(c) for c in cmd))
    code = subprocess.call(cmd, cwd=str(cwd or ROOT))
    log(f"EXIT {code} :: {' '.join(str(c) for c in cmd)}")
    return code

# --- Exp 9 first (fixed t, 60/15, no Zaniar sweep) ---
EXP9_ARMS = [
    "Historical_Late_Split_Balanced_TDA",
    "Early_Split_TDA",
    "No_Undersampling",
]
EXP9_DATASETS = [
    "pkdd_czech",
    "south_german_credit",
    "statlog_german",
    "taiwan_bankruptcy",
    "polish_bankruptcy",
    "credit_card_default",
]
for arm in EXP9_ARMS:
    script = ROOT / "5_Experiments" / arm / "9_Revised_Snapshot_Protocol" / "run_protocol.py"
    for ds in EXP9_DATASETS:
        run([str(PY), str(script), "--stage", "split_ml", "--no-sweep", "--datasets", ds])

# --- Exp 1 historical L / l=500 for missing arms ---
EXP1 = [
    ("pkdd_czech", "No_Undersampling"),
    ("pkdd_czech", "Early_Split_TDA"),
    ("pkdd_czech", "Early_Split_TDA_And_No_Undersampling"),
    ("south_german_credit", "No_Undersampling"),
    ("south_german_credit", "Early_Split_TDA"),
    ("south_german_credit", "Early_Split_TDA_And_No_Undersampling"),
    ("statlog_german", "No_Undersampling"),
    ("statlog_german", "Early_Split_TDA_And_No_Undersampling"),
    ("taiwan_bankruptcy", "No_Undersampling"),
    ("taiwan_bankruptcy", "Early_Split_TDA"),
    ("taiwan_bankruptcy", "Early_Split_TDA_And_No_Undersampling"),
    ("polish_bankruptcy", "No_Undersampling"),
    ("polish_bankruptcy", "Early_Split_TDA"),
    ("polish_bankruptcy", "Early_Split_TDA_And_No_Undersampling"),
    ("credit_card_default", "No_Undersampling"),
    ("credit_card_default", "Early_Split_TDA_And_No_Undersampling"),
]
CONSUMERS = [
    "2_PH_Tuned_Parameters",
    "3_H0_Only",
    "4_Dropping_Correlated_Barcode_Statistics_Columns",
    "5_Linear_Regression_For_Prediction",
    "6_Sampling_Ratio_Audit",
    "7_Snapshot_Mean_Variance",
    "8_Null_Hypothesis_Algorithm2",
]
for ds, arm in EXP1:
    run([str(PY), "-c",
         f"from utils import run_protocol_experiment; run_protocol_experiment({ds!r}, {arm!r}, '1_PH_Default_Parameters', skip_existing_barcodes=True)"])
    for exp in CONSUMERS:
        run([str(PY), "-c",
             f"from utils import run_protocol_experiment; run_protocol_experiment({ds!r}, {arm!r}, {exp!r})"])

# Historical consumers already have Exp 1; run remaining cheap/consumer jobs if missing
for ds in ("pkdd_czech", "south_german_credit", "statlog_german", "taiwan_bankruptcy", "polish_bankruptcy", "credit_card_default"):
    for exp in CONSUMERS:
        run([str(PY), "-c",
             f"from utils import run_protocol_experiment; run_protocol_experiment({ds!r}, 'Historical_Late_Split_Balanced_TDA', {exp!r})"])
    # Early_Split DCCCD + Statlog already have Exp 1
    if ds in ("credit_card_default", "statlog_german"):
        for exp in CONSUMERS:
            run([str(PY), "-c",
                 f"from utils import run_protocol_experiment; run_protocol_experiment({ds!r}, 'Early_Split_TDA', {exp!r})"])

log("QUEUE FINISHED")
