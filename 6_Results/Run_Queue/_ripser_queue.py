# Sequential Ripser / protocol queue. Resume-safe: Exp 1 skip_existing; Exp 9 skip by ml_results run_key.
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / "tda_env" / "Scripts" / "python.exe"
LOG = HERE / "_ripser_queue.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd, cwd=None):
    log("RUN " + " ".join(str(c) for c in cmd))
    code = subprocess.call(cmd, cwd=str(cwd or ROOT))
    log(f"EXIT {code} :: {' '.join(str(c) for c in cmd)}")
    return code


H0H1_ARMS = [
    "Late_Split_And_Undersample_H0_And_H1",
    "Early_Split_And_Undersample_H0_And_H1",
    "Late_Split_No_Undersample_H0_And_H1",
    "Early_Split_No_Undersample_H0_And_H1",
]
H0_OF = {
    "Late_Split_And_Undersample_H0_And_H1": "Late_Split_And_Undersample_H0",
    "Early_Split_And_Undersample_H0_And_H1": "Early_Split_And_Undersample_H0",
    "Late_Split_No_Undersample_H0_And_H1": "Late_Split_No_Undersample_H0",
    "Early_Split_No_Undersample_H0_And_H1": "Early_Split_No_Undersample_H0",
}
DATASETS = [
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
PH_SCRIPTS = {
    "pkdd_czech": ("PKDD_Czech_Financial", "pkdd_czech_financial"),
    "south_german_credit": ("South_German_Credit", "south_german_credit"),
    "statlog_german": ("Statlog_German_Credit_Data", "statlog_german_credit_data"),
    "taiwan_bankruptcy": ("Taiwan_Bankruptcy", "taiwan_bankruptcy"),
    "polish_bankruptcy": ("Polish_Bankruptcy_3Year", "polish_bankruptcy_3year"),
    "credit_card_default": ("Default_Of_Credit_Card_Client_Data", "default_of_credit_cards_client"),
}
H0H1_CONSUMERS = {
    "2_PH_Tuned_Parameters": "_PH_tuned.py",
    "6_Sampling_Ratio_Audit": "_audit.py",
    "8_Null_Hypothesis_Algorithm2": "_algorithm2.py",
}


def dataset_script(arm, dataset_key, experiment, suffix):
    folder, stem = PH_SCRIPTS[dataset_key]
    return ROOT / "5_Experiments" / arm / experiment / folder / (stem + suffix)


for arm in H0H1_ARMS:
    for ds in DATASETS:
        folder, fname = PROTOCOL_SCRIPTS[ds]
        script = ROOT / "5_Experiments" / arm / "9_Revised_Snapshot_Protocol" / folder / fname
        run([str(PY), str(script)])

for arm in H0H1_ARMS:
    for ds in DATASETS:
        run([str(PY), str(dataset_script(arm, ds, "1_PH_Default_Parameters", "_PH.py"))])
        for exp, suffix in H0H1_CONSUMERS.items():
            run([str(PY), str(dataset_script(arm, ds, exp, suffix))])
        h0 = H0_OF[arm]
        run([str(PY), str(dataset_script(h0, ds, "1_PH_Default_Parameters", "_H0_only.py"))])
        run([str(PY), str(dataset_script(h0, ds, "8_Null_Hypothesis_Algorithm2", "_algorithm2.py"))])

log("QUEUE FINISHED")
