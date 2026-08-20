# Resume-safe: train Exp 1 models (barcodes already exist) and missing consumers.
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "tda_env" / "Scripts" / "python.exe"
LOG = HERE / "_consumer_queue.log"

ARMS = (
    "Historical_Late_Split_Balanced_TDA",
    "Early_Split_TDA",
    "No_Undersampling",
    "Early_Split_TDA_And_No_Undersampling",
)
DATASETS = (
    "pkdd_czech",
    "south_german_credit",
    "statlog_german",
    "taiwan_bankruptcy",
    "polish_bankruptcy",
    "credit_card_default",
)
EXPERIMENTS = (
    "1_PH_Default_Parameters",
    "2_PH_Tuned_Parameters",
    "3_H0_Only",
    "4_Dropping_Correlated_Barcode_Statistics_Columns",
    "5_Linear_Regression_For_Prediction",
    "7_Snapshot_Mean_Variance",
    "8_Null_Hypothesis_Algorithm2",
)
FOLDERS = {
    "pkdd_czech": "PKDD_Czech_Financial",
    "south_german_credit": "South_German_Credit",
    "statlog_german": "Statlog_German_Credit_Data",
    "taiwan_bankruptcy": "Taiwan_Bankruptcy",
    "polish_bankruptcy": "Polish_Bankruptcy_3Year",
    "credit_card_default": "Default_Of_Credit_Card_Client_Data",
}
STEMS = {
    "pkdd_czech": "pkdd_czech_financial",
    "south_german_credit": "south_german_credit",
    "statlog_german": "statlog_german_credit_data",
    "taiwan_bankruptcy": "taiwan_bankruptcy",
    "polish_bankruptcy": "polish_bankruptcy_3year",
    "credit_card_default": "default_of_credit_cards_client",
}
SUFFIX = {
    "1_PH_Default_Parameters": "_PH.py",
    "2_PH_Tuned_Parameters": "_PH_tuned.py",
    "3_H0_Only": "_H0_only.py",
    "4_Dropping_Correlated_Barcode_Statistics_Columns": "_PH.py",
    "5_Linear_Regression_For_Prediction": "_PH.py",
    "7_Snapshot_Mean_Variance": "_mean_variance.py",
    "8_Null_Hypothesis_Algorithm2": "_algorithm2.py",
}
PICKLES = {
    "1_PH_Default_Parameters": "model_results.pkl",
    "2_PH_Tuned_Parameters": "model_results.pkl",
    "3_H0_Only": "model_results.pkl",
    "4_Dropping_Correlated_Barcode_Statistics_Columns": "model_results.pkl",
    "5_Linear_Regression_For_Prediction": "model_results.pkl",
    "7_Snapshot_Mean_Variance": "snapshot_mean_variance_full.pkl",
    "8_Null_Hypothesis_Algorithm2": "algorithm2_permutation_results.pkl",
}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def already_done(arm, exp, dataset_key):
    folder = FOLDERS[dataset_key]
    pickle_name = PICKLES[exp]
    path = ROOT / "6_Results" / arm / exp / folder / pickle_name
    return path.exists()


def main():
    for arm in ARMS:
        for dataset_key in DATASETS:
            for exp in EXPERIMENTS:
                if already_done(arm, exp, dataset_key):
                    log(f"SKIP {arm} {dataset_key} {exp}")
                    continue
                folder = FOLDERS[dataset_key]
                script = ROOT / "5_Experiments" / arm / exp / folder / (STEMS[dataset_key] + SUFFIX[exp])
                log("RUN " + str(script))
                code = subprocess.call([str(PY), str(script)], cwd=str(ROOT))
                log(f"EXIT {code} :: {script}")


if __name__ == "__main__":
    main()
