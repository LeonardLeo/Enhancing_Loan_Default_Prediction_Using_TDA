"""Regression checks for dataset registry + four registry datasets."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils import get_dataset_config, get_dataset_folder  # noqa: E402

INGEST = ROOT / "1_Data" / "ingest_registry_datasets.py"
_spec = importlib.util.spec_from_file_location("ingest_registry_datasets", INGEST)
_ingest = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ingest)

parse_yymmdd = _ingest.parse_yymmdd
RAW = ROOT / "1_Data" / "Datasets"
PROCESSED = ROOT / "1_Data" / "Processed_Datasets"


def _read_processed(key: str) -> pd.DataFrame:
    folder = get_dataset_config(key).folder_name
    return pd.read_csv(PROCESSED / folder / "processed_data.csv")


def test_registry_preserves_legacy_aliases_and_adds_new_datasets():
    assert get_dataset_folder("Statlog") == "Statlog_German_Credit_Data"
    assert get_dataset_folder("defaultofcreditcard") == "Default_Of_Credit_Card_Client_Data"
    assert get_dataset_config("pkdd").key == "pkdd_czech"
    assert get_dataset_config("polish3year").key == "polish_bankruptcy"
    assert get_dataset_config("taiwan").key == "taiwan_bankruptcy"
    assert get_dataset_config("southgerman").key == "south_german_credit"
    assert get_dataset_config("pkdd").raw_relative_path.startswith("1_Data/Datasets/")


def test_pkdd_transaction_filter_is_strictly_before_loan_date():
    base = RAW / "PKDD_Czech_Financial"
    transactions = pd.read_csv(base / "trans.asc", sep=";", usecols=["account_id", "date"])
    loans = pd.read_csv(base / "loan.asc", sep=";", usecols=["loan_id", "account_id", "date"])
    transactions["transaction_date"] = parse_yymmdd(transactions.pop("date"))
    loans["loan_date"] = parse_yymmdd(loans.pop("date"))
    joined = transactions.merge(loans, on="account_id", how="inner")
    selected = joined[joined["transaction_date"] < joined["loan_date"]]
    assert len(selected) == 54_694
    assert (selected["transaction_date"] >= selected["loan_date"]).sum() == 0
    assert selected.groupby("loan_id")["transaction_date"].max().lt(
        loans.set_index("loan_id")["loan_date"]
    ).all()


def test_confirmed_target_mappings_in_processed_outputs():
    pkdd = _read_processed("pkdd_czech")
    south = _read_processed("south_german_credit")
    assert pkdd["target"].value_counts().to_dict() == {0: 606, 1: 76}
    assert south["target"].value_counts().to_dict() == {0: 700, 1: 300}


def test_polish_uses_only_three_year_file_and_retains_missing_values_for_train_fit():
    polish = _read_processed("polish_bankruptcy")
    assert polish.shape == (10_503, 65)
    assert polish.drop(columns="target").isna().sum().sum() == 9_888
    assert polish["target"].value_counts().to_dict() == {0: 10_008, 1: 495}


def test_taiwan_constant_and_large_value_evidence_is_preserved_pre_cleaning():
    taiwan = _read_processed("taiwan_bankruptcy")
    assert taiwan["Net Income Flag"].nunique() == 1
    assert (taiwan.drop(columns="target").abs() >= 1e9).any().any()
    assert taiwan["target"].value_counts().to_dict() == {0: 6599, 1: 220}


def test_experiment_scripts_use_utils_not_pipeline():
    script = (
        ROOT / "5_Experiments" / "1_ML_Default_Parameters"
        / "PKDD_Czech_Financial" / "pkdd_czech_financial.py"
    )
    text = script.read_text(encoding="utf-8")
    assert "from utils import" in text
    assert "from pipeline import" not in text
    assert "5_Experiments/common" not in text
    assert "train_dataset" in text
    assert "data_preprocessing_pipeline" in text
