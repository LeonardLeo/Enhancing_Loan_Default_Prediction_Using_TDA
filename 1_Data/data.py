# -*- coding: utf-8 -*-
"""Convenience loaders for TDA barcode matrices and class-wise barcode statistics.

Canonical default: Historical_Late_Split_Balanced_TDA / 1_PH_Default_Parameters
Layout: 1_Data/{kind}/{protocol}/{experiment}/{dataset}/...

Pass ``protocol=`` to load another arm. Experiment 2 (tuned models) does not
rewrite barcode matrices — those aliases resolve to experiment 1. Experiment 3
(H0-only) writes sliced matrices under 3_H0_Only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_DATA_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _DATA_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import (  # noqa: E402
    ACTIVE_TDA_PROTOCOL_BUCKETS,
    tda_artefact_dir,
)

DEFAULT_PROTOCOL = "Historical_Late_Split_Balanced_TDA"
EXP_DEFAULT = "1_PH_Default_Parameters"
EXP_TUNED = "2_PH_Tuned_Parameters"
EXP_H0 = "3_H0_Only"

STATLOG = "Statlog_German_Credit_Data"
DCCCD = "Default_Of_Credit_Card_Client_Data"


def available_protocols() -> tuple:
    return ACTIVE_TDA_PROTOCOL_BUCKETS


def load_tda_matrix(
    dataset_folder: str,
    filename: str,
    *,
    protocol: str = DEFAULT_PROTOCOL,
    experiment: str = EXP_DEFAULT,
    split: str | None = None,
) -> pd.DataFrame:
    extra = (split, filename) if split else (filename,)
    path = tda_artefact_dir("TDA_Datasets", protocol, experiment, dataset_folder, *extra)
    return pd.read_csv(path)


def load_class_barcode_stats(
    dataset_folder: str,
    filename: str,
    *,
    protocol: str = DEFAULT_PROTOCOL,
    experiment: str = EXP_DEFAULT,
    split: str | None = None,
) -> pd.DataFrame:
    extra = (split, filename) if split else (filename,)
    path = tda_artefact_dir("Barcode_Statistics", protocol, experiment, dataset_folder, *extra)
    return pd.read_csv(path)


def _matrix(folder: str, filename: str, experiment: str = EXP_DEFAULT) -> pd.DataFrame:
    return load_tda_matrix(folder, filename, protocol=DEFAULT_PROTOCOL, experiment=experiment)


def _stats(folder: str, filename: str, experiment: str = EXP_DEFAULT) -> pd.DataFrame:
    return load_class_barcode_stats(folder, filename, protocol=DEFAULT_PROTOCOL, experiment=experiment)


def _optional_stats(folder: str, filename: str, experiment: str) -> pd.DataFrame | None:
    """H0-only does not rewrite class-wise barcode CSVs; skip if absent."""
    extra = (filename,)
    path = tda_artefact_dir("Barcode_Statistics", DEFAULT_PROTOCOL, experiment, folder, *extra)
    if not path.exists():
        return None
    return pd.read_csv(path)


# =============================================================================
# STATLOG GERMAN CREDIT DATASET
# =============================================================================

# Class barcode statistics — Historical / 1_PH_Default_Parameters
sgcd_barcode_stats_default_L30 = _stats(STATLOG, "barcode_stats_default_L30.csv")
sgcd_barcode_stats_default_L60 = _stats(STATLOG, "barcode_stats_default_L60.csv")
sgcd_barcode_stats_non_default_L30 = _stats(STATLOG, "barcode_stats_non-default_L30.csv")
sgcd_barcode_stats_non_default_L60 = _stats(STATLOG, "barcode_stats_non-default_L60.csv")

# Entire barcode matrices — Historical / 1_PH_Default_Parameters
sgcd_data_L30 = _matrix(STATLOG, "data_L30.csv")
sgcd_data_L60 = _matrix(STATLOG, "data_L60.csv")

# Tuned-model aliases: Exp 2 trains on the Exp 1 barcode matrix (no rewrite).
sgcd_barcode_stats_default_L30_4_PH_Tuned_Parameters = sgcd_barcode_stats_default_L30
sgcd_barcode_stats_default_L60_4_PH_Tuned_Parameters = sgcd_barcode_stats_default_L60
sgcd_barcode_stats_non_default_L30_4_PH_Tuned_Parameters = sgcd_barcode_stats_non_default_L30
sgcd_barcode_stats_non_default_L60_4_PH_Tuned_Parameters = sgcd_barcode_stats_non_default_L60
sgcd_data_L30_4_PH_Tuned_Parameters = sgcd_data_L30
sgcd_data_L60_4_PH_Tuned_Parameters = sgcd_data_L60

# H0-only combined matrices — Historical / 3_H0_Only
# Class-wise barcode CSVs are not rewritten by the H0 consumer; optional.
sgcd_barcode_stats_default_L30_6_Experiment_Impact_of_H0_Only = _optional_stats(
    STATLOG, "barcode_stats_default_L30.csv", EXP_H0
)
sgcd_barcode_stats_default_L60_6_Experiment_Impact_of_H0_Only = _optional_stats(
    STATLOG, "barcode_stats_default_L60.csv", EXP_H0
)
sgcd_barcode_stats_non_default_L30_6_Experiment_Impact_of_H0_Only = _optional_stats(
    STATLOG, "barcode_stats_non-default_L30.csv", EXP_H0
)
sgcd_barcode_stats_non_default_L60_6_Experiment_Impact_of_H0_Only = _optional_stats(
    STATLOG, "barcode_stats_non-default_L60.csv", EXP_H0
)
sgcd_data_L30_6_Experiment_Impact_of_H0_Only = _matrix(STATLOG, "data_L30.csv", experiment=EXP_H0)
sgcd_data_L60_6_Experiment_Impact_of_H0_Only = _matrix(STATLOG, "data_L60.csv", experiment=EXP_H0)


# =============================================================================
# DEFAULT OF CREDIT CARD CLIENT DATASET
# =============================================================================

dcccd_barcode_stats_default_L5_3_PH_Default_Parameters = _stats(DCCCD, "barcode_stats_default_L5.csv")
dcccd_barcode_stats_default_L15_3_PH_Default_Parameters = _stats(DCCCD, "barcode_stats_default_L15.csv")
dcccd_barcode_stats_non_default_L5_3_PH_Default_Parameters = _stats(DCCCD, "barcode_stats_non-default_L5.csv")
dcccd_barcode_stats_non_default_L15_3_PH_Default_Parameters = _stats(DCCCD, "barcode_stats_non-default_L15.csv")

dcccd_data_L5_3_PH_Default_Parameters = _matrix(DCCCD, "data_L5.csv")
dcccd_data_L153_PH_Default_Parameters = _matrix(DCCCD, "data_L15.csv")

dcccd_barcode_stats_default_L5_4_PH_Tuned_Parameters = dcccd_barcode_stats_default_L5_3_PH_Default_Parameters
dcccd_barcode_stats_default_L15_4_PH_Tuned_Parameters = dcccd_barcode_stats_default_L15_3_PH_Default_Parameters
dcccd_barcode_stats_non_default_L5_4_PH_Tuned_Parameters = dcccd_barcode_stats_non_default_L5_3_PH_Default_Parameters
dcccd_barcode_stats_non_default_L15_4_PH_Tuned_Parameters = dcccd_barcode_stats_non_default_L15_3_PH_Default_Parameters
dcccd_data_L5_4_PH_Tuned_Parameters = dcccd_data_L5_3_PH_Default_Parameters
dcccd_data_L15_4_PH_Tuned_Parameters = dcccd_data_L153_PH_Default_Parameters

dcccd_barcode_stats_default_L5_6_Experiment_Impact_of_H0_Only = _optional_stats(
    DCCCD, "barcode_stats_default_L5.csv", EXP_H0
)
dcccd_barcode_stats_default_L15_6_Experiment_Impact_of_H0_Only = _optional_stats(
    DCCCD, "barcode_stats_default_L15.csv", EXP_H0
)
dcccd_barcode_stats_non_default_L5_6_Experiment_Impact_of_H0_Only = _optional_stats(
    DCCCD, "barcode_stats_non-default_L5.csv", EXP_H0
)
dcccd_barcode_stats_non_default_L15_6_Experiment_Impact_of_H0_Only = _optional_stats(
    DCCCD, "barcode_stats_non-default_L15.csv", EXP_H0
)
dcccd_data_L5_6_Experiment_Impact_of_H0_Only = _matrix(DCCCD, "data_L5.csv", experiment=EXP_H0)
dcccd_data_L15_6_Experiment_Impact_of_H0_Only = _matrix(DCCCD, "data_L15.csv", experiment=EXP_H0)
