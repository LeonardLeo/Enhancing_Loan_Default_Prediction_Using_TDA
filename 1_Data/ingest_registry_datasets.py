# -*- coding: utf-8 -*-
"""
Build processed_data.csv for the four registry datasets from raw files under
1_Data/Datasets/{Folder}/.

Run from repo root:
  python 1_Data/ingest_registry_datasets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import get_dataset_config

RAW = ROOT / "1_Data" / "Datasets"
PROCESSED = ROOT / "1_Data" / "Processed_Datasets"

CATEGORICAL_SOUTH = {
    "laufkont", "moral", "verw", "sparkont", "beszeit", "rate", "famges",
    "buerge", "wohnzeit", "verm", "weitkred", "wohn", "bishkred", "beruf",
    "pers", "telef", "gastarb",
}


# =============================================================================
# Helpers
# =============================================================================
def parse_yymmdd(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    yy = text.str[:2].astype(int)
    yyyy = np.where(yy >= 30, 1900 + yy, 2000 + yy)
    return pd.to_datetime(
        pd.Series(yyyy, index=values.index).astype(str) + text.str[2:],
        format="%Y%m%d",
        errors="coerce",
    )


def write_audit(key: str, df: pd.DataFrame, ingestion: dict) -> None:
    folder = PROCESSED / get_dataset_config(key).folder_name
    audits = folder / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    X = df.drop(columns="target")
    payload = {
        "dataset": key,
        "rows": len(df),
        "columns_including_target": len(df.columns),
        "target_counts": df["target"].value_counts().sort_index().to_dict(),
        "missing_cells": int(X.isna().sum().sum()),
        "leakage_risks": ingestion,
    }
    (audits / f"{key}_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "feature": X.columns,
            "missing_count": X.isna().sum().values,
            "unique_count": X.nunique(dropna=False).values,
        }
    ).to_csv(audits / f"{key}_feature_audit.csv", index=False)


# =============================================================================
# PKDD'99 Czech Financial
# =============================================================================
def load_pkdd() -> tuple[pd.DataFrame, dict]:
    base = RAW / "PKDD_Czech_Financial"
    tables = {p.stem: pd.read_csv(p, sep=";", low_memory=False) for p in base.glob("*.asc")}
    loan = tables["loan"].copy()
    loan["loan_date"] = parse_yymmdd(loan["date"])
    loan["target"] = loan["status"].map({"A": 0, "C": 0, "B": 1, "D": 1})

    trans = tables["trans"].copy()
    trans["trans_date"] = parse_yymmdd(trans["date"])
    tx = trans.merge(loan[["loan_id", "account_id", "loan_date"]], on="account_id", how="inner")
    tx = tx[tx["trans_date"] < tx["loan_date"]].copy()
    tx["is_credit"] = tx["type"].eq("PRIJEM").astype(int)
    tx["is_debit"] = (~tx["type"].eq("PRIJEM")).astype(int)
    tx = tx.sort_values(["loan_id", "trans_date", "trans_id"])
    grouped = tx.groupby("loan_id", sort=True)
    agg = grouped.agg(
        tx_count=("trans_id", "count"),
        tx_amount_sum=("amount", "sum"),
        tx_amount_mean=("amount", "mean"),
        tx_amount_std=("amount", "std"),
        tx_amount_max=("amount", "max"),
        tx_balance_mean=("balance", "mean"),
        tx_balance_min=("balance", "min"),
        tx_balance_max=("balance", "max"),
        tx_last_balance=("balance", "last"),
        tx_credit_count=("is_credit", "sum"),
        tx_debit_count=("is_debit", "sum"),
        last_trans_date=("trans_date", "max"),
    ).reset_index()
    out = loan.merge(agg, on="loan_id", how="left")
    out["days_since_last_transaction"] = (out["loan_date"] - out["last_trans_date"]).dt.days

    account = tables["account"].rename(
        columns={"date": "account_open_raw", "district_id": "account_district_id"}
    )
    account["account_open_date"] = parse_yymmdd(account["account_open_raw"])
    out = out.merge(account, on="account_id", how="left", suffixes=("", "_account"))
    out["account_tenure_days"] = (out["loan_date"] - out["account_open_date"]).dt.days

    owner = tables["disp"][tables["disp"]["type"].eq("OWNER")].copy()
    client = tables["client"].rename(columns={"district_id": "client_district_id"}).copy()
    birth = client["birth_number"].astype(str).str.zfill(6)
    month = birth.str[2:4].astype(int)
    client["sex"] = np.where(month > 50, "female", "male")
    adjusted = (
        birth.str[:2]
        + (month.where(month <= 50, month - 50)).astype(str).str.zfill(2)
        + birth.str[4:]
    )
    client["birth_date"] = parse_yymmdd(adjusted)
    owner_client = owner.merge(client, on="client_id", how="left", suffixes=("", "_client"))
    out = out.merge(owner_client, on="account_id", how="left")
    out["client_age_years"] = (out["loan_date"] - out["birth_date"]).dt.days / 365.25

    district = tables["district"].rename(columns={"A1": "account_district_id"})
    out = out.merge(district, on="account_district_id", how="left")

    cards = tables["card"].merge(owner[["disp_id", "account_id"]], on="disp_id", how="inner")
    cards["issued_date"] = pd.to_datetime(cards["issued"].str[:6], format="%y%m%d", errors="coerce")
    cards = cards.merge(loan[["loan_id", "account_id", "loan_date"]], on="account_id")
    cards = cards[cards["issued_date"] <= cards["loan_date"]]
    card_agg = cards.groupby("loan_id").agg(
        preloan_card_count=("card_id", "count"),
        earliest_card_date=("issued_date", "min"),
        preloan_card_type=("type", "first"),
    ).reset_index()
    out = out.merge(card_agg, on="loan_id", how="left")
    out["card_tenure_days"] = (out["loan_date"] - out["earliest_card_date"]).dt.days

    max_violation = int((tx["trans_date"] >= tx["loan_date"]).sum())
    temporal = {
        "strict_rule": "transaction_date < loan_date",
        "transactions_joined_preloan": len(tx),
        "post_or_same_day_transactions_included": max_violation,
        "orders_included": False,
        "loan_status_mapping": {"A": 0, "C": 0, "B": 1, "D": 1},
    }
    drop = [
        "loan_id", "account_id", "date", "status", "loan_date", "last_trans_date",
        "account_open_raw", "account_open_date", "disp_id", "client_id",
        "birth_number", "birth_date", "client_district_id", "account_district_id",
        "earliest_card_date",
    ]
    out = out.drop(columns=[c for c in drop if c in out], errors="ignore")
    if max_violation:
        raise AssertionError("PKDD temporal leakage check failed")
    return out, temporal


# =============================================================================
# Other registry datasets
# =============================================================================
def load_polish() -> tuple[pd.DataFrame, dict]:
    raw, _ = arff.loadarff(RAW / "Polish_Bankruptcy_3Year" / "3year.arff")
    df = pd.DataFrame(raw)
    df["target"] = df.pop("class").astype(str).str.extract(r"([01])")[0].astype(int)
    return df.apply(pd.to_numeric, errors="coerce"), {"source_subset": "3year.arff only"}


def load_taiwan() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW / "Taiwan_Bankruptcy" / "data.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Bankrupt?": "target"})
    return df, {"note": "winsorization applied later in experiment scripts if needed"}


def load_south_german() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW / "South_German_Credit" / "SouthGermanCredit.asc", sep=r"\s+")
    df["target"] = df.pop("kredit").map({0: 1, 1: 0})
    for col in CATEGORICAL_SOUTH & set(df.columns):
        df[col] = df[col].astype(str)
    return df, {
        "framing": "updated-German sensitivity analysis",
        "target_mapping": {"0_bad": 1, "1_good": 0},
    }


LOADERS = {
    "pkdd_czech": load_pkdd,
    "polish_bankruptcy": load_polish,
    "taiwan_bankruptcy": load_taiwan,
    "south_german_credit": load_south_german,
}


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    keys = sys.argv[1:] or list(LOADERS)
    for key in keys:
        print(f"Ingesting {key}...")
        df, meta = LOADERS[key]()
        folder = PROCESSED / get_dataset_config(key).folder_name
        folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(folder / "processed_data.csv", index=False)
        write_audit(key, df, meta)
        print(f"  -> {folder / 'processed_data.csv'} ({len(df)} rows)")
    print("Done.")
