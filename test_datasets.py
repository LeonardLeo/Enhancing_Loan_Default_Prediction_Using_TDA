"""Regression checks for the live Default of Credit Card Client table."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils import get_dataset_config, get_dataset_folder  # noqa: E402


def test_registry_keeps_credit_card_default():
    assert get_dataset_folder("defaultofcreditcard") == "Default_Of_Credit_Card_Client_Data"
    assert get_dataset_config("credit_card_default").key == "credit_card_default"
    assert get_dataset_config("credit_card_default").landmark_percentages == (5.0, 15.0)


def test_unknown_retired_aliases_are_rejected():
    for alias in ("pkdd", "polish3year", "taiwan", "southgerman", "statlog", "statlog_german"):
        try:
            get_dataset_config(alias)
        except ValueError:
            continue
        raise AssertionError(f"retired alias still registered: {alias}")


def test_experiment_scripts_use_utils_not_pipeline():
    script = (
        ROOT
        / "5_Experiments"
        / "Default_Parameters"
        / "1_ML_Default_Parameters"
        / "Default_Of_Credit_Card_Client_Data"
        / "default_of_credit_cards_client_data.py"
    )
    text = script.read_text(encoding="utf-8")
    assert "from utils import" in text
    assert "from pipeline import" not in text
    assert "5_Experiments/common" not in text
    assert "train_dataset" in text or "data_preprocessing_pipeline" in text
