# Resume-safe: train default models (barcodes already exist) and missing consumers.
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "tda_env" / "Scripts" / "python.exe"
LOG = HERE / "_consumer_queue.log"

H0H1_ARMS = (
    "Early_Split_And_Undersample_H0_And_H1",
    "Early_Split_No_Undersample_H0_And_H1",
    "Late_Split_And_Undersample_H0_And_H1",
    "Late_Split_No_Undersample_H0_And_H1",
)
H0_ARMS = (
    "Early_Split_And_Undersample_H0",
    "Early_Split_No_Undersample_H0",
    "Late_Split_And_Undersample_H0",
    "Late_Split_No_Undersample_H0",
)
H0H1_EXPERIMENTS = (
    "1_PH_Default_Parameters",
    "2_PH_Tuned_Parameters",
    "6_Sampling_Ratio_Audit",
    "8_Null_Hypothesis_Algorithm2",
)
H0_EXPERIMENTS = (
    "1_PH_Default_Parameters",
    "2_PH_Tuned_Parameters",
    "6_Sampling_Ratio_Audit",
    "8_Null_Hypothesis_Algorithm2",
    "9_Revised_Snapshot_Protocol",
)
DATASETS = (
    "statlog_german",
    "credit_card_default",
)
FOLDERS = {
    "statlog_german": "Statlog_German_Credit_Data",
    "credit_card_default": "Default_Of_Credit_Card_Client_Data",
}
STEMS = {
    "statlog_german": "statlog_german_credit_data",
    "credit_card_default": "default_of_credit_cards_client",
}
H0H1_SUFFIX = {
    "1_PH_Default_Parameters": "_PH.py",
    "2_PH_Tuned_Parameters": "_PH_tuned.py",
    "6_Sampling_Ratio_Audit": "_audit.py",
    "8_Null_Hypothesis_Algorithm2": "_algorithm2.py",
}
H0_SUFFIX = {
    "1_PH_Default_Parameters": "_H0_only.py",
    "2_PH_Tuned_Parameters": "_PH_tuned.py",
    "6_Sampling_Ratio_Audit": "_audit.py",
    "8_Null_Hypothesis_Algorithm2": "_algorithm2.py",
    "9_Revised_Snapshot_Protocol": "_protocol.py",
}
PROTOCOL_SCRIPTS = {
    "statlog_german": ("Statlog_German_Credit_Data", "statlog_german_credit_protocol.py"),
    "credit_card_default": ("Default_Of_Credit_Card_Client_Data", "default_of_credit_card_client_protocol.py"),
}
PICKLES = {
    "1_PH_Default_Parameters": "model_results.pkl",
    "2_PH_Tuned_Parameters": "model_results.pkl",
    "6_Sampling_Ratio_Audit": "sampling_ratio_audit.csv",
    "8_Null_Hypothesis_Algorithm2": "algorithm2_permutation_results.pkl",
    "9_Revised_Snapshot_Protocol": "ml_results.csv",
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


def dataset_script(arm, dataset_key, exp, suffixes):
    if exp == "9_Revised_Snapshot_Protocol":
        folder, fname = PROTOCOL_SCRIPTS[dataset_key]
        return ROOT / "5_Experiments" / arm / exp / folder / fname
    folder = FOLDERS[dataset_key]
    return ROOT / "5_Experiments" / arm / exp / folder / (STEMS[dataset_key] + suffixes[exp])


def run_jobs(arms, experiments, suffixes):
    for arm in arms:
        for dataset_key in DATASETS:
            for exp in experiments:
                if already_done(arm, exp, dataset_key):
                    log(f"SKIP {arm} {dataset_key} {exp}")
                    continue
                script = dataset_script(arm, dataset_key, exp, suffixes)
                log("RUN " + str(script))
                code = subprocess.call([str(PY), str(script)], cwd=str(ROOT))
                log(f"EXIT {code} :: {script}")


def main():
    run_jobs(H0H1_ARMS, H0H1_EXPERIMENTS, H0H1_SUFFIX)
    run_jobs(H0_ARMS, H0_EXPERIMENTS, H0_SUFFIX)


if __name__ == "__main__":
    main()
