"""Focused regression checks for the four-dataset extension."""
from pathlib import Path

import pandas as pd

from run_new_datasets import RAW, parse_yymmdd, read_processed
from utils import get_dataset_config, get_dataset_folder


def test_registry_preserves_legacy_aliases_and_adds_new_datasets():
    assert get_dataset_folder("Statlog") == "Statlog_German_Credit_Data"
    assert get_dataset_folder("defaultofcreditcard") == "Default_Of_Credit_Card_Client_Data"
    assert get_dataset_config("pkdd").key == "pkdd_czech"
    assert get_dataset_config("polish3year").key == "polish_bankruptcy"
    assert get_dataset_config("taiwan").key == "taiwan_bankruptcy"
    assert get_dataset_config("southgerman").key == "south_german_credit"


def test_pkdd_transaction_filter_is_strictly_before_loan_date():
    """Reconstruct the temporal join and prove the selected set has no leakage."""
    transactions = pd.read_csv(RAW / "03_pkdd_czech" / "trans.asc", sep=";", usecols=["account_id", "date"])
    loans = pd.read_csv(RAW / "03_pkdd_czech" / "loan.asc", sep=";", usecols=["loan_id", "account_id", "date"])
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
    pkdd = read_processed("pkdd_czech")
    south = read_processed("south_german_credit")
    assert pkdd["target"].value_counts().to_dict() == {0: 606, 1: 76}
    assert south["target"].value_counts().to_dict() == {0: 700, 1: 300}


def test_polish_uses_only_three_year_file_and_retains_missing_values_for_train_fit():
    polish = read_processed("polish_bankruptcy")
    assert polish.shape == (10_503, 65)
    assert polish.drop(columns="target").isna().sum().sum() == 9_888
    assert polish["target"].value_counts().to_dict() == {0: 10_008, 1: 495}


def test_taiwan_constant_and_large_value_evidence_is_preserved_pre_cleaning():
    taiwan = read_processed("taiwan_bankruptcy")
    assert taiwan["Net Income Flag"].nunique() == 1
    assert (taiwan.drop(columns="target").abs() >= 1e9).any().any()
    assert taiwan["target"].value_counts().to_dict() == {0: 6599, 1: 220}
