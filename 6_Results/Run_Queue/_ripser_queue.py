# Sequential Ripser / protocol queue. Resume-safe: Exp 1 skip_existing; Exp 9 skip by ml_results run_key.
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "tda_env" / "Scripts" / "python.exe"
LOG = HERE / "_ripser_queue.log"

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
PROTOCOL_SCRIPTS = {
    "pkdd_czech": ("PKDD_Czech_Financial", "pkdd_czech_financial_protocol.py"),
    "south_german_credit": ("South_German_Credit", "south_german_credit_protocol.py"),
    "statlog_german": ("Statlog_German_Credit_Data", "statlog_german_credit_protocol.py"),
    "taiwan_bankruptcy": ("Taiwan_Bankruptcy", "taiwan_bankruptcy_protocol.py"),
    "polish_bankruptcy": ("Polish_Bankruptcy_3Year", "polish_bankruptcy_3year_protocol.py"),
    "credit_card_default": ("Default_Of_Credit_Card_Client_Data", "default_of_credit_card_client_protocol.py"),
}
for arm in EXP9_ARMS:
    for ds in EXP9_DATASETS:
        folder, fname = PROTOCOL_SCRIPTS[ds]
        script = ROOT / "5_Experiments" / arm / "9_Revised_Snapshot_Protocol" / folder / fname
        run([str(PY), str(script)])

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
PH_SCRIPTS = {
    "pkdd_czech": ("PKDD_Czech_Financial", "pkdd_czech_financial"),
    "south_german_credit": ("South_German_Credit", "south_german_credit"),
    "statlog_german": ("Statlog_German_Credit_Data", "statlog_german_credit_data"),
    "taiwan_bankruptcy": ("Taiwan_Bankruptcy", "taiwan_bankruptcy"),
    "polish_bankruptcy": ("Polish_Bankruptcy_3Year", "polish_bankruptcy_3year"),
    "credit_card_default": ("Default_Of_Credit_Card_Client_Data", "default_of_credit_cards_client"),
}
CONSUMER_SUFFIX = {
    "1_PH_Default_Parameters": "_PH.py",
    "2_PH_Tuned_Parameters": "_PH_tuned.py",
    "3_H0_Only": "_H0_only.py",
    "4_Dropping_Correlated_Barcode_Statistics_Columns": "_PH.py",
    "5_Linear_Regression_For_Prediction": "_PH.py",
    "6_Sampling_Ratio_Audit": "_audit.py",
    "7_Snapshot_Mean_Variance": "_mean_variance.py",
    "8_Null_Hypothesis_Algorithm2": "_algorithm2.py",
}


def dataset_script(arm, dataset_key, experiment):
    folder, stem = PH_SCRIPTS[dataset_key]
    return ROOT / "5_Experiments" / arm / experiment / folder / (stem + CONSUMER_SUFFIX[experiment])


for ds, arm in EXP1:
    run([str(PY), str(dataset_script(arm, ds, "1_PH_Default_Parameters"))])
    for exp in CONSUMERS:
        run([str(PY), str(dataset_script(arm, ds, exp))])

# Historical consumers already have Exp 1; run remaining cheap/consumer jobs if missing
for ds in ("pkdd_czech", "south_german_credit", "statlog_german", "taiwan_bankruptcy", "polish_bankruptcy", "credit_card_default"):
    for exp in CONSUMERS:
        run([str(PY), str(dataset_script("Historical_Late_Split_Balanced_TDA", ds, exp))])
    if ds in ("credit_card_default", "statlog_german"):
        for exp in CONSUMERS:
            run([str(PY), str(dataset_script("Early_Split_TDA", ds, exp))])

log("QUEUE FINISHED")
