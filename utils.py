# -*- coding: utf-8 -*-
"""
Functions utilised towards analysing loan defaults/credit card defaults of customers, while saving the best model.
"""

# Import Libraries
import os
import glob
import time
import logging
import re
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import kmapper as km
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyvis.network import Network
from itertools import product, combinations
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Sequence, Iterable
import textwrap
from ripser import ripser
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from kmapper import KeplerMapper
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import mannwhitneyu
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split, 
                                     GridSearchCV, 
                                     StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             classification_report,
                             confusion_matrix,
                             average_precision_score,
                             balanced_accuracy_score,
                             roc_auc_score)

# =============================================================================
# UMAP
# =============================================================================
# Optional: import UMAP only if installed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def win_long_path(path) -> Path:
    """Windows path that can be created/opened beyond MAX_PATH (260)."""
    raw = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not raw.startswith("\\\\?\\"):
        if raw.startswith("\\\\"):
            raw = "\\\\?\\UNC\\" + raw[2:]
        else:
            raw = "\\\\?\\" + raw
    return Path(raw)


def _percent_token(percent: float) -> str:
    """Canonical landmark-percent filename token: 10.0 → '10', 1.36 → '1.36'."""
    return str(int(percent)) if float(percent).is_integer() else str(percent)


def _ensure_dir(path) -> Path:
    """Create ``path`` (and parents). Returns an absolute Path.

    Always resolve relative folders against the current working directory so
    ``savefig`` / Mapper HTML writes do not depend on mixed slash tricks.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = win_long_path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

# =============================================================================
# Barcode Statistics Column Descriptions (for Dimensions 0 and 1)
# =============================================================================
# Format: gX_Y
# Where:
#   - X is the statistic number (1–12)
#   - Y is the homology dimension (0 or 1)

COLUMN_DESCRIPTIONS = {
    'g1_0':  'Mean Birth (Dim 0)',
    'g2_0':  'Mean Death (Dim 0)',
    'g3_0':  'Mean Persistence (Dim 0)',
    'g4_0':  'Mean Gap to Max Death (Dim 0)',
    'g5_0':  'Median Birth (Dim 0)',
    'g6_0':  'Median Death (Dim 0)',
    'g7_0':  'Median Persistence (Dim 0)',
    'g8_0':  'Median Gap to Max Death (Dim 0)',
    'g9_0':  'Std Birth (Dim 0)',
    'g10_0': 'Std Death (Dim 0)',
    'g11_0': 'Std Persistence (Dim 0)',
    'g12_0': 'Std Gap to Max Death (Dim 0)',

    'g1_1':  'Mean Birth (Dim 1)',
    'g2_1':  'Mean Death (Dim 1)',
    'g3_1':  'Mean Persistence (Dim 1)',
    'g4_1':  'Mean Gap to Max Death (Dim 1)',
    'g5_1':  'Median Birth (Dim 1)',
    'g6_1':  'Median Death (Dim 1)',
    'g7_1':  'Median Persistence (Dim 1)',
    'g8_1':  'Median Gap to Max Death (Dim 1)',
    'g9_1':  'Std Birth (Dim 1)',
    'g10_1': 'Std Death (Dim 1)',
    'g11_1': 'Std Persistence (Dim 1)',
    'g12_1': 'Std Gap to Max Death (Dim 1)'
}



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Repository root (this file lives at the project root). Landmark / barcode
# writers resolve against this so script depth (bucket vs archive) cannot
# break artefact paths.
REPO_ROOT = Path(__file__).resolve().parent

# Eight live processes: split × undersample × (just H0 vs both H0 and H1).
# Public names always use "and", never "+". Figures and paper tables must
# call process_display_name() so labels cannot drift from this registry.
TDA_PROCESS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Early_Split_And_Undersample_H0": {
        "display_name": "Early split and undersample, using just H0",
        "split_timing": "early",
        "undersample": True,
        "homology": "H0",
        "historical": False,
        "description": "Customers split first; undersample inside each split; homology-0 barcode statistics only.",
    },
    "Early_Split_And_Undersample_H0_And_H1": {
        "display_name": "Early split and undersample, using both H0 and H1",
        "split_timing": "early",
        "undersample": True,
        "homology": "H0_and_H1",
        "historical": False,
        "description": "Customers split first; undersample inside each split; homology 0 and 1 barcode statistics.",
    },
    "Early_Split_No_Undersample_H0": {
        "display_name": "Early split, no undersample, using just H0",
        "split_timing": "early",
        "undersample": False,
        "homology": "H0",
        "historical": False,
        "description": "Customers split first; full class pools; homology-0 barcode statistics only.",
    },
    "Early_Split_No_Undersample_H0_And_H1": {
        "display_name": "Early split, no undersample, using both H0 and H1",
        "split_timing": "early",
        "undersample": False,
        "homology": "H0_and_H1",
        "historical": False,
        "description": "Customers split first; full class pools; homology 0 and 1 barcode statistics.",
    },
    "Late_Split_And_Undersample_H0": {
        "display_name": "Late split and undersample (the original historical run), using just H0",
        "split_timing": "late",
        "undersample": True,
        "homology": "H0",
        "historical": True,
        "description": "PCA on the full table; undersample; homology-0 barcode statistics only.",
    },
    "Late_Split_And_Undersample_H0_And_H1": {
        "display_name": "Late split and undersample (the original historical run), using both H0 and H1",
        "split_timing": "late",
        "undersample": True,
        "homology": "H0_and_H1",
        "historical": True,
        "description": "PCA on the full table; undersample; homology 0 and 1 barcode statistics.",
    },
    "Late_Split_No_Undersample_H0": {
        "display_name": "Late split, no undersample, using just H0",
        "split_timing": "late",
        "undersample": False,
        "homology": "H0",
        "historical": False,
        "description": "PCA on the full table; no undersample; homology-0 barcode statistics only.",
    },
    "Late_Split_No_Undersample_H0_And_H1": {
        "display_name": "Late split, no undersample, using both H0 and H1",
        "split_timing": "late",
        "undersample": False,
        "homology": "H0_and_H1",
        "historical": False,
        "description": "PCA on the full table; no undersample; homology 0 and 1 barcode statistics.",
    },
}

_H0_H0H1_PAIRS = (
    ("Early_Split_And_Undersample_H0", "Early_Split_And_Undersample_H0_And_H1"),
    ("Early_Split_No_Undersample_H0", "Early_Split_No_Undersample_H0_And_H1"),
    ("Late_Split_And_Undersample_H0", "Late_Split_And_Undersample_H0_And_H1"),
    ("Late_Split_No_Undersample_H0", "Late_Split_No_Undersample_H0_And_H1"),
)
for _h0_slug, _h0h1_slug in _H0_H0H1_PAIRS:
    TDA_PROCESS_REGISTRY[_h0_slug]["barcode_source_bucket"] = _h0h1_slug
    TDA_PROCESS_REGISTRY[_h0_slug]["h0_and_h1_bucket"] = _h0h1_slug
    TDA_PROCESS_REGISTRY[_h0h1_slug]["barcode_source_bucket"] = _h0h1_slug
    TDA_PROCESS_REGISTRY[_h0h1_slug]["h0_and_h1_bucket"] = _h0h1_slug

LEGACY_PROTOCOL_BUCKETS = {
    "Historical_Late_Split_Balanced_TDA": "Late_Split_And_Undersample_H0_And_H1",
    "Early_Split_TDA": "Early_Split_And_Undersample_H0_And_H1",
    "No_Undersampling": "Late_Split_No_Undersample_H0_And_H1",
    "Early_Split_TDA_And_No_Undersampling": "Early_Split_No_Undersample_H0_And_H1",
}

ACTIVE_TDA_PROTOCOL_BUCKETS = tuple(TDA_PROCESS_REGISTRY.keys())
TDA_PROTOCOL_SPECS = TDA_PROCESS_REGISTRY

ACTIVE_TDA_EXPERIMENT_NAMES = (
    "1_PH_Default_Parameters",
    "2_PH_Tuned_Parameters",
    "6_Sampling_Ratio_Audit",
    "8_Null_Hypothesis_Algorithm2",
    "9_Revised_Snapshot_Protocol",
)

ARCHIVED_NESTED_EXPERIMENT_NAMES = (
    "3_H0_Only",
    "4_Dropping_Correlated_Barcode_Statistics_Columns",
    "5_Linear_Regression_For_Prediction",
    "7_Snapshot_Mean_Variance",
)


def resolve_protocol_bucket(protocol_bucket: str) -> str:
    """Map a live or legacy protocol folder name onto the eight-process slug."""
    if not protocol_bucket:
        return protocol_bucket
    if protocol_bucket in TDA_PROCESS_REGISTRY:
        return protocol_bucket
    return LEGACY_PROTOCOL_BUCKETS.get(protocol_bucket, protocol_bucket)


def process_display_name(protocol_bucket: str) -> str:
    """Public process name for figures, reports, and paper tables."""
    key = resolve_protocol_bucket(protocol_bucket)
    spec = TDA_PROCESS_REGISTRY.get(key)
    if spec:
        return str(spec["display_name"])
    return str(protocol_bucket).replace("_", " ")


def process_figure_title(protocol_bucket: str, title: str) -> str:
    """Prefix a figure title with the registry display name when this is a live TDA process."""
    key = resolve_protocol_bucket(protocol_bucket)
    if key not in TDA_PROCESS_REGISTRY:
        return title
    name = process_display_name(key)
    if not name or name in title:
        return title
    return f"{name} — {title}"


def barcode_source_bucket(protocol_bucket: str) -> str:
    """Folder that owns Ripser output. H0 processes read the matching H0-and-H1 run."""
    key = resolve_protocol_bucket(protocol_bucket)
    spec = TDA_PROCESS_REGISTRY.get(key) or {}
    return str(spec.get("barcode_source_bucket") or key)


def tda_artefact_dir(
    kind: str,
    protocol_bucket: str,
    experiment_name: str,
    dataset_folder: str,
    *extra: str,
) -> Path:
    """Protocol-mirrored artefact path under 1_Data/{kind}/.

    kind is Landmark_Sets, Barcode_Statistics, or TDA_Datasets.
    Layout: 1_Data/{kind}/{ProtocolBucket}/{ExperimentName}/{DatasetFolder}/[extra]
    """
    if kind not in {"Landmark_Sets", "Barcode_Statistics", "TDA_Datasets"}:
        raise ValueError(f"Unknown TDA artefact kind: {kind}")
    path = REPO_ROOT / "1_Data" / kind / resolve_protocol_bucket(protocol_bucket) / experiment_name / dataset_folder
    for part in extra:
        if part:
            path = path / part
    return win_long_path(path)


def tda_results_dir(protocol_bucket: str, experiment_name: str, dataset_folder: str) -> Path:
    return win_long_path(
        REPO_ROOT / "6_Results" / resolve_protocol_bucket(protocol_bucket) / experiment_name / dataset_folder
    )


def get_tda_protocol(protocol_bucket: str) -> Dict[str, Any]:
    key = resolve_protocol_bucket(protocol_bucket)
    if key not in TDA_PROTOCOL_SPECS:
        raise ValueError(
            f"Unknown TDA protocol bucket '{protocol_bucket}'. "
            f"Known: {', '.join(TDA_PROTOCOL_SPECS)}"
        )
    spec = dict(TDA_PROTOCOL_SPECS[key])
    spec["bucket"] = key
    return spec

# =============================================================================
# Dataset registry
# =============================================================================
@dataclass(frozen=True)
class DatasetConfig:
    """Canonical metadata used by both legacy and registry-driven pipelines.

    ``pca_variance`` (default 0.90) is the *target* for new tables, not a
    guarantee of Exp 3. ``notes["pca_n_components_exp3"]`` is the rank Exp 3
    actually used (7 / 15 / 10). ``landmark_percentages`` are dataset-specific
    on purpose — see ``notes["landmark_reason"]`` and docs/Design_Decisions.md.
    """

    key: str
    display_name: str
    folder_name: str
    aliases: Tuple[str, ...]
    target_column: str
    positive_label: int = 1
    raw_relative_path: Optional[str] = None
    pca_variance: float = 0.90
    landmark_percentages: Tuple[float, ...] = (10.0, 20.0)
    notes: Dict[str, Any] = field(default_factory=dict)


DATASET_REGISTRY: Dict[str, DatasetConfig] = {}
DATASET_ALIASES: Dict[str, str] = {}


def register_dataset(config: DatasetConfig, overwrite: bool = False) -> DatasetConfig:
    """Register a dataset and normalized aliases."""
    key = config.key.strip().lower()
    if key in DATASET_REGISTRY and not overwrite:
        raise ValueError(f"Dataset already registered: {config.key}")
    DATASET_REGISTRY[key] = config
    for alias in (config.key, config.folder_name, *config.aliases):
        normalized = re.sub(r"[^a-z0-9]", "", alias.lower())
        if normalized in DATASET_ALIASES and DATASET_ALIASES[normalized] != key and not overwrite:
            raise ValueError(f"Dataset alias already registered: {alias}")
        DATASET_ALIASES[normalized] = key
    return config


def get_dataset_config(dataset: str) -> DatasetConfig:
    """Resolve a canonical key, folder name, or backwards-compatible alias."""
    normalized = re.sub(r"[^a-z0-9]", "", str(dataset).lower())
    key = DATASET_ALIASES.get(normalized)
    if key is None:
        options = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset}'. Registered datasets: {options}")
    return DATASET_REGISTRY[key]


def get_dataset_folder(dataset: str) -> str:
    return get_dataset_config(dataset).folder_name


for _dataset_config in (
    DatasetConfig(
        key="statlog_german",
        display_name="Statlog German Credit",
        folder_name="Statlog_German_Credit_Data",
        aliases=("dataset1", "sgcd", "statlog", "german", "statloggerman"),
        target_column="Class",
        pca_variance=0.90,
        landmark_percentages=(30.0, 60.0),
        notes={
            "pca_n_components_exp3": 15,
            "landmark_reason": "Original paper percents. Minority class count=300, so 30%/60% are required to get 90/180 points per snapshot.",
        },
    ),
    DatasetConfig(
        key="credit_card_default",
        display_name="Default of Credit Card Client",
        folder_name="Default_Of_Credit_Card_Client_Data",
        aliases=("dataset2", "dccdd", "default", "defaultofcreditcard", "defaultofcreditcardclientdata"),
        target_column="default payment next month",
        pca_variance=0.90,
        landmark_percentages=(5.0, 15.0),
        notes={
            "pca_n_components_exp3": 7,
            "landmark_reason": "Original paper percents. Minority class count=6630, so 5% already gives 331 points per snapshot.",
        },
    ),
    DatasetConfig(
        key="pkdd_czech",
        display_name="PKDD'99 Czech Financial",
        folder_name="PKDD_Czech_Financial",
        aliases=("pkdd", "czech", "berka"),
        target_column="target",
        raw_relative_path="1_Data/Datasets/PKDD_Czech_Financial",
        pca_variance=0.90,
        landmark_percentages=(10.0, 20.0),
        notes={
            "pca_n_components_exp3": 10,
            "landmark_reason": "Shared new-table percents. 5% would give 3 points per snapshot on a minority class of 76 (too small for PH); 30% would over-reuse.",
        },
    ),
    DatasetConfig(
        key="polish_bankruptcy",
        display_name="Polish Companies Bankruptcy (3 year)",
        folder_name="Polish_Bankruptcy_3Year",
        aliases=("polish", "3year", "polish3year"),
        target_column="target",
        raw_relative_path="1_Data/Datasets/Polish_Bankruptcy_3Year/3year.arff",
        pca_variance=0.90,
        landmark_percentages=(10.0, 20.0),
        notes={
            "pca_n_components_exp3": 10,
            "missing_indicators": True,
            "landmark_reason": "Shared new-table percents so PKDD/Polish/Taiwan/South German stay comparable.",
        },
    ),
    DatasetConfig(
        key="taiwan_bankruptcy",
        display_name="Taiwanese Bankruptcy Prediction",
        folder_name="Taiwan_Bankruptcy",
        aliases=("taiwan", "taiwanese"),
        target_column="target",
        raw_relative_path="1_Data/Datasets/Taiwan_Bankruptcy/data.csv",
        pca_variance=0.90,
        landmark_percentages=(10.0, 20.0),
        notes={
            "pca_n_components_exp3": 10,
            "train_only_winsor_quantiles": (0.005, 0.995),
            "landmark_reason": "Shared new-table percents so PKDD/Polish/Taiwan/South German stay comparable.",
        },
    ),
    DatasetConfig(
        key="south_german_credit",
        display_name="South German Credit (updated-German sensitivity)",
        folder_name="South_German_Credit",
        aliases=("southgerman", "sgc", "updatedgerman"),
        target_column="target",
        raw_relative_path="1_Data/Datasets/South_German_Credit/SouthGermanCredit.asc",
        pca_variance=0.90,
        landmark_percentages=(10.0, 20.0),
        notes={
            "pca_n_components_exp3": 10,
            "landmark_reason": "Shared new-table percents. Not Statlog's 30/60: this is a sensitivity table, kept on the same 10%/20% snapshot-size grid as the other new sets.",
            "target_mapping": {"0_bad": 1, "1_good": 0},
            "sensitivity_analysis": True,
        },
    ),
):
    register_dataset(_dataset_config)

# Defining Functions
def fix_string(word: str):
    import string
    punctuations = string.punctuation
    
    for each_letter in word:
        if each_letter in punctuations:
            word = word.replace(each_letter, " ")
    
    return word


def eda(dataset: pd.DataFrame,
        bin_size: int or list = None,
        graphs: bool = False,
        hue: str = None,
        markers: list = None,
        only_graphs: bool = False,
        hist_figsize: tuple = (15, 10),
        corr_heatmap_figsize: tuple = (15, 10),
        pairplot_figsize: tuple = (15, 10)) -> dict:

    if only_graphs != True:
        data_unique = {}
        data_category_count = {}
        data_numeric_count = {}
        dataset.info()
        data_head = dataset.head()
        data_tail = dataset.tail()
        data_mode = dataset.mode().iloc[0]
        data_descriptive_stats = dataset.describe()
        data_more_descriptive_stats = dataset.describe(include = "all")
        data_correlation_matrix = dataset.corr(numeric_only = True)
        data_distinct_count = dataset.nunique()
        data_count_duplicates = dataset.duplicated().sum()
        data_duplicates = dataset[dataset.duplicated()]
        data_count_null = dataset.isnull().sum()
        # data_null = dataset[any(dataset.isna())]
        data_total_null = dataset.isnull().sum().sum()
        for each_column in dataset.columns: # Loop through each column and get the unique values
            data_unique[each_column] = dataset[each_column].unique()
        for each_column in dataset.select_dtypes(object).columns:
            # Loop through the categorical columns and count how many values are in each category
            data_category_count[each_column] = dataset[each_column].value_counts()
        for each_column in dataset.select_dtypes(exclude = object).columns:
            # Loop through the numeric columns and count how many values are in each category
            data_numeric_count[each_column] = dataset[each_column].value_counts()

    if graphs == True:
        # Visualising Histograms
        dataset.hist(figsize = hist_figsize, bins = bin_size)
        plt.show()

        if only_graphs != False:
            # Creating a heatmap for the correlation matrix
            plt.figure(figsize = corr_heatmap_figsize)
            sns.heatmap(data_correlation_matrix, annot = True, cmap = 'coolwarm')
            plt.show()

        # Creating the pairplot for the dataset
        plt.figure(figsize = pairplot_figsize)
        sns.pairplot(dataset, hue = hue, markers = markers) # Graph of correlation across each numerical feature
        plt.show()

    if only_graphs != True:
        result = {"data_head": data_head,
                  "data_tail": data_tail,
                  "data_mode": data_mode,
                  "data_descriptive_stats": data_descriptive_stats,
                  "data_more_descriptive_stats": data_more_descriptive_stats,
                  "data_correlation_matrix": data_correlation_matrix,
                  "data_distinct_count": data_distinct_count,
                  "data_count_duplicates": data_count_duplicates,
                  "data_count_null": data_count_null,
                  "data_total_null": data_total_null,
                  "data_unique": data_unique,
                  "data_duplicates": data_duplicates,
                  # "data_null": data_null,
                  "data_category_count": data_category_count,
                  "data_numeric_count": data_numeric_count,
                  }
        return result

def data_preprocessing_pipeline(dataset: pd.DataFrame,
                                drop_columns: list = None,
                                log_col: list = None,
                                dummy_col: list = None,
                                replace_val: dict = None):
    # DATA CLEANING AND TRANSFORMATION
    # Converting the card type to numeric
    if replace_val is not None:
        dataset = dataset.replace(replace_val)
    
    # Fix dummy variables
    if dummy_col is not None:
        dataset = pd.get_dummies(dataset,
                                 columns = dummy_col,
                                 drop_first = True,
                                 dtype = np.int64)

    # Creating logrithmic columns
    if log_col is not None:
        for each_col in log_col:
            dataset[f"Log_{each_col}"] = np.log1p(dataset[each_col])

    # Dropping columns
    if drop_columns is not None:
        dataset = dataset.drop(drop_columns, axis = 1)

    return dataset

def save_python_object_using_joblib(python_object, 
                                   dataset_to_use: str,
                                   save_item: str,
                                   save_name: str,
                                   experiment_name: str):
    # Preparing dataset save
    dataset = dataset_to_use.strip().lower()
    
    # Preparing what to save (2 options - EDA and Feature Info) 
    save_item = save_item.strip().lower()
    save_item_options_1 = ["option1", "eda", "explore", "exploratory", "data_analysis", "dataanalysis"]
    save_item_options_2 = ["option2", "feature_info", "feature", "featureinfo"]
    save_item_options_3 = ["option3", "processeddata", "processed_data", "clean_data", "processed"]
    
    # Confirm dataset chosen
    dataset_string = get_dataset_folder(dataset)
        
    # Confirm save item
    if save_item in save_item_options_1:
        save_item_string = "EDA"
        path = f"../../../3_Python_Objects/{dataset_string}/{experiment_name}/{save_item_string}" 
    elif save_item in save_item_options_2:
        save_item_string = "Feature_Information"
        path = f"../../../3_Python_Objects/{dataset_string}/{experiment_name}/{save_item_string}" 
    elif save_item in save_item_options_3:
        save_item_string = "Processed_Data"
        path = f"../../../1_Data/Processed_Datasets/{dataset_string}/{experiment_name}" 
    
    # Preparing save item
    os.makedirs(path, exist_ok = True)
    joblib.dump(python_object, f"{path}/{save_name}")
        
def train_dataset(X_resampled, 
                  y_resampled, 
                  X_test, 
                  y_test,
                  **kwargs):
    # Step 1: Initialize models
    models = {
        "svm": SVC(**kwargs.get("svm", {})),
        "knn": KNeighborsClassifier(**kwargs.get("knn", {})),
        "xgb": XGBClassifier(**kwargs.get("xgb", {})),
        "logistic": LogisticRegression(**kwargs.get("logistic", {})),
        "random_forest": RandomForestClassifier(**kwargs.get("random_forest", {}))
    }

    # Step 2: Train models with default parameters and evaluate
    results = {}

    for model_name, model in models.items():
        print(f"\n\nTraining {model_name}...")
        if model_name == "knn":
            # Ensure input data is in the correct format
            X_resampled = X_resampled.to_numpy() if isinstance(X_resampled, pd.DataFrame) else X_resampled
            y_resampled = y_resampled.to_numpy().ravel() if isinstance(y_resampled, pd.DataFrame) else y_resampled
            X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
            y_test = y_test.to_numpy().ravel() if isinstance(y_test, pd.DataFrame) else y_test

        # Fit the model with default parameters
        model.fit(X_resampled, y_resampled)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Store results
        results[model_name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

        print(f"{model_name} training completed.")

    return results

def select_landmarks(data: pd.DataFrame, 
                     percentage: int | float,  # What percentage of the original data will be sampled (L5, L10, L15, etc) 
                     n_files: int,  # How many number of samples to be taken
                     dataset_to_use: str,
                     save_label_dir: str,  # Class Path to store results
                     experiment_name: str,
                     add_optional_path: str = None,
                     verbose: bool = False,
                     protocol_bucket: str = None):

    # Print current working directory for debugging
    if verbose:
        print("Current working directory:", os.getcwd())

    # Preparing dataset save
    dataset = dataset_to_use.strip().lower()
    n_landmarks = max(2, int(len(data) * percentage / 100))
    if n_landmarks > len(data):
        raise ValueError(f"Requested {n_landmarks} landmarks from only {len(data)} rows")

    # Confirm dataset chosen
    dataset_string = get_dataset_folder(dataset)

    # Protocol buckets write
    # 1_Data/Landmark_Sets/{ProtocolBucket}/{ExperimentName}/{DatasetFolder}/...
    # Legacy / archive calls keep DatasetFolder/ExperimentName.
    if protocol_bucket:
        output_dir = str(
            tda_artefact_dir(
                "Landmark_Sets",
                protocol_bucket,
                experiment_name,
                dataset_string,
                *([add_optional_path] if add_optional_path else []),
                save_label_dir,
            )
        )
    elif add_optional_path is None:
        output_dir = str(
            REPO_ROOT / "1_Data" / "Landmark_Sets" / dataset_string / experiment_name / save_label_dir
        )
    else:
        output_dir = str(
            REPO_ROOT / "1_Data" / "Landmark_Sets" / dataset_string / experiment_name / add_optional_path / save_label_dir
        )
        

    # Convert relative path to absolute path (Windows long-path safe)
    absolute_output_dir = str(win_long_path(output_dir))
    if verbose:
        print("Saving to:", absolute_output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(absolute_output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Check if the directory was successfully created
    if not os.path.exists(absolute_output_dir):
        print(f"[ERR] Directory does not exist: {absolute_output_dir}")

    # Saving landmarks
    for i in range(n_files):
        # Use pandas' sample method to select random landmarks
        landmarks = data.sample(n=n_landmarks, random_state=i)  # Use random_state for reproducibility
        
        # Construct file path for each landmark file
        file_path = os.path.join(absolute_output_dir, f"landmarks_{percentage}_{i}.csv")
        
        # Print the file path to debug
        if verbose:
            print(f"Saving to: {file_path}")
        
        # Save the landmarks to a CSV file
        try:
            landmarks.to_csv(file_path, index=False)
            if verbose:
                print(f"[OK] Complete for landmarks_{percentage}_{i}")
        except Exception as e:
            raise RuntimeError(f"Error saving landmarks_{percentage}_{i}: {e}") from e
    print(
        f"Saved {n_files} landmark files ({n_landmarks} points each) to "
        f"{absolute_output_dir}"
    )

def generate_landmark_sets(class_label_and_data: dict, # Dictionary containing split classes and their data which is used to generate samples
                           landmark_percentages: list,
                           dataset_to_use: str,
                           experiment_name: str,
                           add_optional_path: str = None,
                           n_files_per_percentage: int = 500,
                           protocol_bucket: str = None): # Number of landmark sets to generate per percentage
    
    # Generate landmarks for default and non-default separately
    for each_name, each_data in class_label_and_data.items():
        for each_percentage in landmark_percentages:
            print(f"\n\nBuilding Landmark Set for L{each_percentage} - Label ({each_name})")
            select_landmarks(data = each_data, 
                             percentage = each_percentage, 
                             n_files = n_files_per_percentage,
                             dataset_to_use = dataset_to_use,
                             save_label_dir = f"{each_name}_L{each_percentage}",
                             add_optional_path = add_optional_path,
                             experiment_name = experiment_name,
                             protocol_bucket = protocol_bucket)

    print("Landmark selection complete!")

def generate_landmark_sets_v2(class_label_and_data: dict,
                             landmark_percentages: list,
                             dataset_to_use: str,
                             experiment_name: str,
                             num_files_per_class: dict,
                             add_optional_path: str = None,
                             protocol_bucket: str = None):
    """
    Generates landmark sets with flexible number of files per class.

    Parameters:
    - class_label_and_data: dict of {class_name: DataFrame}
    - landmark_percentages: list of percentages like [5, 10, 15]
    - dataset_to_use: dataset identifier string
    - experiment_name: name for saving results
    - num_files_per_class: dict of {class_name: number_of_files}
    """
    for class_name, class_data in class_label_and_data.items():
        n_files = num_files_per_class.get(class_name, 0)
        for percentage in landmark_percentages:
            print(f"\n\nBuilding Landmark Set for L{percentage} - Label ({class_name}) with {n_files} files")
            select_landmarks(data=class_data,
                             percentage=percentage,
                             n_files=n_files,
                             dataset_to_use=dataset_to_use,
                             save_label_dir=f"{class_name}_L{_percent_token(percentage)}",
                             add_optional_path = add_optional_path,
                             experiment_name=experiment_name,
                             protocol_bucket=protocol_bucket)
    print("Landmark selection complete!")

def compute_barcode_statistics(diagram):
    """
    Compute 12 barcode statistics for a persistence diagram.
    Handles cases with invalid or empty diagrams.
    Returns a list of 12 statistics (g1 to g12).
    """
    if len(diagram) == 0:
        # If no persistence pairs, return zeros
        return [0] * 12

    # Ensure the diagram is a NumPy array
    diagram = np.array(diagram)

    # Separate birth and death times
    birth_times = diagram[:, 0]
    death_times = diagram[:, 1]

    # Compute persistence
    persistence = death_times - birth_times

    # Filter valid entries: both birth and death are finite, and persistence is non-negative
    valid_indices = (
        np.isfinite(birth_times)
        & np.isfinite(death_times)
        & (persistence >= 0)
    )
    birth_times = birth_times[valid_indices]
    death_times = death_times[valid_indices]
    persistence = persistence[valid_indices]

    # If no valid entries remain, return zeros
    if len(birth_times) == 0:
        return [0] * 12

    # Compute ymax (maximum death time)
    ymax = np.nanmax(death_times) if len(death_times) > 0 else 0

    # Safely compute statistics
    try:
        stats = [
            np.nanmean(birth_times),
            np.nanmean(death_times),
            np.nanmean(persistence),
            np.nanmean(ymax - death_times) if len(death_times) > 0 else 0,
            np.nanmedian(birth_times),
            np.nanmedian(death_times),
            np.nanmedian(persistence),
            np.nanmedian(ymax - death_times) if len(death_times) > 0 else 0,
            np.nanstd(birth_times),
            np.nanstd(death_times),
            np.nanstd(persistence),
            np.nanstd(ymax - death_times) if len(death_times) > 0 else 0,
        ]
    except Exception as e:
        print(f"Error in statistical computation: {e}")
        stats = [0] * 12

    return stats

def rename_barcode_statistics_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames barcode statistic column names like 'g1_0', 'g12_1' to descriptive names,
    while being robust to missing, shuffled, or filtered columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with potentially partially-renamed or subset columns.

    Returns:
    - pd.DataFrame: New DataFrame with renamed columns where applicable.
    """
    
    # Mapping from g-codes to descriptive names
    feature_map = {
        "g1": "Mean Birth",
        "g2": "Mean Death",
        "g3": "Mean Persistence",
        "g4": "Mean Gap to Max Death",
        "g5": "Median Birth",
        "g6": "Median Death",
        "g7": "Median Persistence",
        "g8": "Median Gap to Max Death",
        "g9": "Std Birth",
        "g10": "Std Death",
        "g11": "Std Persistence",
        "g12": "Std Gap to Max Death"
    }

    # Function to rename one column
    def rename_column(col):
        match = re.fullmatch(r"(g\d{1,2})_(\d)", col)
        if match:
            g_code, dim = match.groups()
            if g_code in feature_map:
                return f"{feature_map[g_code]} (Dim {dim})"
        return col  # leave unchanged if not matching the pattern

    # Rename and return
    new_columns = [rename_column(col) for col in df.columns]
    renamed_df = df.copy()
    renamed_df.columns = new_columns
    return renamed_df

def compute_barcodes_from_multiple_landmarks(landmark_percentages: List[int | float],
                                             landmark_dir: str,
                                             barcode_output_dir: str,
                                             dim: int,
                                             label: Dict[int, str]) -> None:
    # Set parameters
    barcode_output_dir = str(win_long_path(barcode_output_dir))
    landmark_dir = str(win_long_path(landmark_dir))
    os.makedirs(barcode_output_dir, exist_ok=True)  # Ensure the output directory exists

    for percentage in landmark_percentages:
        for each_class, label_name in label.items():
            print(f"\n\nComputing Barcode Statistics for L{percentage} - Label ({label_name})")
            create_barcode_statistics(
                landmark_dir = os.path.abspath(os.path.join(landmark_dir, f"{label_name}_L{_percent_token(percentage)}")),  # Use abspath for landmark_dir
                output_file = os.path.abspath(os.path.join(barcode_output_dir, f"barcode_stats_{label_name}_L{_percent_token(percentage)}.csv")),  # Ensure the file path is absolute
                label = each_class,
                dim = dim
            )

    print("Barcode computation complete!")


def create_barcode_statistics(landmark_dir: str, 
                              output_file: str,
                              dim: int,
                              label: int):
    rows = []
    files = [f for f in os.listdir(landmark_dir) if f.endswith(".csv")]

    for file in files:
        file_path = os.path.join(landmark_dir, file)
        try:
            landmarks = pd.read_csv(file_path).values
            result = ripser(landmarks)
            diagrams = result['dgms']

            if dim > len(diagrams):
                raise ValueError(f"Requested dim={dim}, but only {len(diagrams)} diagrams found.")

            row = []
            for d in range(dim):
                row.extend(compute_barcode_statistics(diagrams[d]))

            row.append(label)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Error processing file {file_path}: {e}")
            continue

    columns = [f"g{i}_{j}" for j in range(dim) for i in range(1, 13)] + ["label"]
    pd.DataFrame(rows, columns=columns).to_csv(output_file, index=False)
    print(f"Processed {len(rows)} valid files in {landmark_dir}. \nResults saved to {output_file}")

def combine_barcode_statistics_per_group(barcode_dir: str, 
                                         percentage: int,
                                         label: Dict[int, str]):
    
    """
    Combine barcode statistics for default and non-default for a specific percentage group.
    """
    # Load default and non-default statistics
    file_storage = []
    
    # Convert the relative path to an absolute path
    barcode_dir_abs = os.path.abspath(barcode_dir)
    
    for each_class, label_name in label.items():
        # Construct absolute path for the barcode files
        token = _percent_token(percentage)
        files = glob.glob(os.path.join(barcode_dir_abs, f"barcode_stats_{label_name}_L{token}.csv"))
        if not files:
            files = glob.glob(os.path.join(barcode_dir_abs, f"barcode_stats_{label_name}_L{percentage}.csv"))
        file_storage.extend(files)
    
    print(f"Number of files found: {len(file_storage)}\n\n")
   
    combined_data = []
    
    for each_file in file_storage:
        data = pd.read_csv(each_file)
        combined_data.append(data)
    
    # Concatenate barcode statistics for this group
    combined_df = pd.concat(combined_data, ignore_index=True)
    return combined_df

def build_final_barcode_statistics_data(landmark_percentages: List[int | float],
                                        barcode_dir: str,
                                        output_dir: str,
                                        label: Dict[int, str]):

    output_dir = str(win_long_path(output_dir))
    barcode_dir = str(win_long_path(barcode_dir))
    os.makedirs(output_dir, exist_ok=True)
    print("\n\nStarting barcode statistics generation...")

    for each_percentage in landmark_percentages:
        print(f"\nProcessing {each_percentage}% landmarks...")

        start_time = time.time()

        # Combine barcode statistics
        barcode_stats_full_data = combine_barcode_statistics_per_group(
            barcode_dir=barcode_dir, 
            percentage=each_percentage,
            label=label
        )
        token = _percent_token(each_percentage)
        barcode_stats_full_data.to_csv(
            os.path.abspath(os.path.join(output_dir, f"data_L{token}.csv")),
            index=False
        )
        print(f"Saved: data_L{token}.csv | Shape: {barcode_stats_full_data.shape}")

        # Time per iteration
        end_time = time.time()
        elapsed_seconds = int(end_time - start_time)
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"[TIME] Time taken for {each_percentage}%: {hours}h {minutes}m {seconds}s")

    print("\nAll barcode statistics generation completed.")

def train_dataset_tda(data: str,
                      y_col_name: str,
                      test_size: float = 0.2,
                      random_state: int = 42,
                      **kwargs):
    # Convert to absolute path
    abs_data_path = os.path.abspath(data)
    print(f"[LOAD] Loading dataset from: {abs_data_path}")
    
    dataset = pd.read_csv(abs_data_path)
    
    # Shuffle
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split
    X = dataset.drop(columns=[y_col_name]).values
    y = dataset[y_col_name].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Models
    models = {
        "svm": SVC(**kwargs.get("svm", {})),
        "knn": KNeighborsClassifier(**kwargs.get("knn", {})),
        "xgb": XGBClassifier(**kwargs.get("xgb", {})),
        "logistic": LogisticRegression(**kwargs.get("logistic", {})),
        "random_forest": RandomForestClassifier(**kwargs.get("random_forest", {})),
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[model_name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        print(f"[OK] Trained {model_name}")
    
    return results

def train_dataset_tda_drop_correlated(X_train: pd.DataFrame,
                                      y_train: pd.Series,
                                      X_test: pd.DataFrame,
                                      y_test: pd.Series,
                                      test_size: float = 0.2,
                                      random_state: int = 42,
                                      **kwargs):    
    models = {
        "svm": SVC(**kwargs.get("svm", {})),
        "knn": KNeighborsClassifier(**kwargs.get("knn", {})),
        "xgb": XGBClassifier(**kwargs.get("xgb", {})),
        "logistic": LogisticRegression(**kwargs.get("logistic", {})),
        "random_forest": RandomForestClassifier(**kwargs.get("random_forest", {})),
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        result = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

        # Feature importance extraction
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = model.coef_[0]  # for binary classification
            else:
                importances = None

            if importances is not None:
                feature_importance_df = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": importances
                }).sort_values(by="importance", ascending=False).reset_index(drop=True)
                result["feature_importance"] = feature_importance_df
            else:
                result["feature_importance"] = None
        except Exception as e:
            print(f"[WARN] Could not extract feature importance for {model_name}: {e}")
            result["feature_importance"] = None

        results[model_name] = result
        print(f"[OK] Trained {model_name}")
    
    return results

def train_dataset_tda_linear_regression(data: str,
                                        y_col_name: str,
                                        test_size: float = 0.2,
                                        random_state: int = 42,
                                        **kwargs):
    # Convert to absolute path
    abs_data_path = os.path.abspath(data)
    print(f"[LOAD] Loading dataset from: {abs_data_path}")
    
    dataset = pd.read_csv(abs_data_path)
    
    # Shuffle
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split
    X = dataset.drop(columns=[y_col_name]).values
    y = dataset[y_col_name].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Models
    models = {
        "linear": LinearRegression(**kwargs.get("linear", {}))
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        # LinearRegression can predict outside {0,1}. Clip after rounding so
        # sklearn binary metrics do not see a spurious third class.
        y_pred = np.clip(np.round(model.predict(X_test)), 0, 1).astype(int)
        
        results[model_name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        print(f"[OK] Trained {model_name}")
    
    return results


def train_knn_with_grid_search(data: str,
                                y_col_name: str,
                                output_path: str,
                                test_size: float = 0.2,
                                random_state: int = 42,
                                k_range: range = range(1, 21)):

    # Convert to absolute path
    output_path = Path(os.path.abspath(output_path))
    abs_data_path = os.path.abspath(data)
    print(f"[LOAD] Loading dataset from: {abs_data_path}")
    dataset = pd.read_csv(abs_data_path)

    # Shuffle
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split
    X = dataset.drop(columns=[y_col_name]).values
    y = dataset[y_col_name].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Grid search on KNN
    param_grid = {'n_neighbors': list(k_range)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_k = grid_search.best_params_['n_neighbors']
    print(f"[BEST] Best k found: {best_k}")

    # Plot elbow diagram
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), scores, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    elbow_path = os.path.join(output_path, "elbow_curve.png")
    plt.savefig(elbow_path)
    print(f"[PLOT] Elbow curve saved to: {elbow_path}")

    # Evaluate best model
    best_model = KNeighborsClassifier(n_neighbors=best_k)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    results = {
        "best_k": best_k,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Save results and model
    os.makedirs(output_path, exist_ok=True)

    results_path = os.path.join(output_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[SAVE] Results saved to: {results_path}")

    model_path = os.path.join(output_path, "best_knn_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"[MODEL] Trained KNN model saved to: {model_path}")

    return results

def train_multiple_dataset_tda(path_datasets: list,
                               y_col_name: str,
                               test_size: float = 0.2,
                               random_state: int = 42,
                               **kwargs):
    model_results = {}
    overall_start = time.time()

    for count, path in enumerate(path_datasets, 1):
        abs_path = os.path.abspath(path)
        print(f"\n\n[RUN] Training on dataset {count}: {abs_path}")
        
        start = time.time()
        results = train_dataset_tda(abs_path,
                                    y_col_name=y_col_name,
                                    test_size=test_size,
                                    random_state=random_state,
                                    **kwargs)
        elapsed = int(time.time() - start)
        h, m, s = divmod(elapsed, 60), *divmod(elapsed % 60, 60)
        print(f"[TIME] Finished in {h}h {m}m {s}s")

        model_results[os.path.basename(path)] = results

    total_time = int(time.time() - overall_start)
    t_h, t_m, t_s = divmod(total_time, 60), *divmod(total_time % 60, 60)
    print(f"\n[OK] All datasets completed in {t_h}h {t_m}m {t_s}s")
    return model_results

def train_multiple_dataset_tda_drop_correlated(data_objects: dict,
                                               test_size: float = 0.2,
                                               random_state: int = 42,
                                               **kwargs):
    model_results = {}
    overall_start = time.time()

    # for count, data in enumerate(data_objects, 1):
    for data, each_data_object in data_objects.items():
        full_data = each_data_object["data"]
        X_train = full_data.drop("label", axis = 1)
        y_train = full_data.label
        X_test = each_data_object["X_test"]
        y_test = each_data_object["y_test"]
        
        print(f"\n\n[RUN] Training on dataset {data}")        
        
        start = time.time()
        results = train_dataset_tda_drop_correlated(X_train=X_train,
                                                    y_train=y_train,
                                                    X_test=X_test,
                                                    y_test=y_test,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    **kwargs)
        elapsed = int(time.time() - start)
        h, m, s = divmod(elapsed, 60), *divmod(elapsed % 60, 60)
        print(f"[TIME] Finished in {h}h {m}m {s}s")

        model_results[data] = results

    total_time = int(time.time() - overall_start)
    t_h, t_m, t_s = divmod(total_time, 60), *divmod(total_time % 60, 60)
    print(f"\n[OK] All datasets completed in {t_h}h {t_m}m {t_s}s")
    return model_results

def train_multiple_dataset_tda_linear_regression(path_datasets: list,
                                                 y_col_name: str,
                                                 test_size: float = 0.2,
                                                 random_state: int = 42,
                                                 **kwargs):
    model_results = {}
    overall_start = time.time()

    for count, path in enumerate(path_datasets, 1):
        abs_path = os.path.abspath(path)
        print(f"\n\n[RUN] Training on dataset {count}: {abs_path}")
        
        start = time.time()
        results = train_dataset_tda_linear_regression(abs_path,
                                                      y_col_name=y_col_name,
                                                      test_size=test_size,
                                                      random_state=random_state,
                                                      **kwargs)
        elapsed = int(time.time() - start)
        h, m, s = divmod(elapsed, 60), *divmod(elapsed % 60, 60)
        print(f"[TIME] Finished in {h}h {m}m {s}s")

        model_results[os.path.basename(path)] = results

    total_time = int(time.time() - overall_start)
    t_h, t_m, t_s = divmod(total_time, 60), *divmod(total_time % 60, 60)
    print(f"\n[OK] All datasets completed in {t_h}h {t_m}m {t_s}s")
    return model_results

def train_multiple_knn_datasets(path_datasets: list,
                                 y_col_name: str,
                                 base_output_path: str,
                                 test_size: float = 0.2,
                                 random_state: int = 42,
                                 k_range: range = range(1, 21)):
    model_results = {}
    overall_start = time.time()

    for count, dataset_path in enumerate(path_datasets, 1):
        abs_path = os.path.abspath(dataset_path)
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = os.path.join(base_output_path, dataset_name)
        os.makedirs(output_path, exist_ok=True)

        print(f"\n\n[RUN] Training on dataset {count}: {abs_path}")
        start = time.time()

        results = train_knn_with_grid_search(
            data=abs_path,
            y_col_name=y_col_name,
            output_path=output_path,
            test_size=test_size,
            random_state=random_state,
            k_range=k_range
        )

        elapsed = int(time.time() - start)
        h, m, s = divmod(elapsed, 60), *divmod(elapsed % 60, 60)
        print(f"[TIME] Finished in {h}h {m}m {s}s")

        model_results[dataset_name] = results

    total_time = int(time.time() - overall_start)
    t_h, t_m, t_s = divmod(total_time, 60), *divmod(total_time % 60, 60)
    print(f"\n[OK] All datasets completed in {t_h}h {t_m}m {t_s}s")

    return model_results


def store_results(path: str, save_name: str, result_object: dict):
    abs_save_dir = win_long_path(path)
    abs_save_dir.mkdir(parents=True, exist_ok=True)

    save_file = abs_save_dir / f"{save_name}.pkl"
    joblib.dump(result_object, save_file)

    print(f"Results saved to: {save_file}")

def store_data_as_csv_or_json(
    path: str,
    csv: bool,
    save_as: List[str],
    data_object: List[Union[pd.DataFrame, pd.Series, dict]]
):
    abs_save_dir = win_long_path(path)
    abs_save_dir.mkdir(parents=True, exist_ok=True)

    if len(save_as) != len(data_object):
        raise ValueError("Length of 'save_as' must match length of 'data_object'")

    for name, obj in zip(save_as, data_object):
        save_file = abs_save_dir / f"{name}.{'csv' if csv else 'json'}"

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            if csv:
                obj.to_csv(save_file, index=False)
            else:
                obj.to_json(save_file, orient="records", lines=False)
        
        elif isinstance(obj, dict):
            if csv:
                raise TypeError(f"[ERR] Cannot save a dict as CSV: {name}")
            else:
                with open(save_file, "w") as f:
                    json.dump(obj, f, indent=4)

        else:
            raise TypeError(f"Unsupported data type for: {name} ({type(obj)})")

        print(f"Saved: {save_file}")
    
def _model_results_entry(model_results: Dict[str, Any], data_name: str) -> Any:
    """Match a barcode CSV path to the pickle key used when that file was trained."""
    data_key = os.path.basename(str(data_name).replace("\\\\?\\", "").replace("//", "/"))
    candidates = [data_key]
    stem, ext = os.path.splitext(data_key)
    if ext.lower() == ".csv":
        candidates.append(stem)
    else:
        candidates.append(data_key + ".csv")
    for key in candidates:
        if key in model_results:
            return model_results[key]
    raise KeyError(
        f"{data_key!r} not in model_results (available: {list(model_results.keys())})"
    )


def perform_cross_validation_tda(datasets: List[str],
                                 model_results: Dict[str, Dict[str, Dict[str, Any]]],
                                 n_splits: int = 10,
                                 shuffle: bool = True,
                                 random_state: int = 42,
                                 test_size: float = 0.2) -> Dict[str, Dict[str, Any]]:
    """
    Perform stratified K-fold cross-validation on multiple datasets and models.
    """
    skf = StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
    all_results = {}

    for data_name in datasets:
        data_key = os.path.basename(data_name)
        print(f"\nEvaluating dataset: {data_key}")
        dataset = pd.read_csv(os.path.abspath(data_name))
        X = dataset.drop(columns=["label"]).values
        y = dataset["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, random_state = random_state, stratify = y
        )

        results = {}
        for model_name, model_info in _model_results_entry(model_results, data_name).items():
            print(f"  Cross-validating model: {model_name}")
            scores = cross_val_score(
                estimator = model_info["model"],
                X = X_train,
                y = y_train,
                cv = skf,
                n_jobs = -1
            )
            results[model_name] = {
                "cross_val_scores": scores,
                "mean_accuracy": np.mean(scores),
                "std_accuracy": np.std(scores)
            }
            print(f"    {model_name} — Mean: {results[model_name]['mean_accuracy']:.4f}, Std: {results[model_name]['std_accuracy']:.4f}")

        all_results[data_name] = results

    return all_results

def train_models_on_dataset(data_path,
                            model_configs,
                            target_column: str = 'label',
                            test_size: float = 0.2,
                            scoring_metric: str = 'f1',
                            scale_features: bool = True,
                            random_state: int = 42,
                            n_splits_kfold: int = 5,
                            **kwargs):
    """
    Trains multiple models on a dataset with hyperparameter tuning and evaluation.
    Accepts kwargs for:
        - train_test_split: prefix `split__`
        - scaler: prefix `scaler__`
        - StratifiedKFold: prefix `cv__`
        - GridSearchCV: prefix `grid__`
        - model instantiation: prefix `model__` (applied globally to all models)
    """
    import inspect

    # Helper: filter kwargs by prefix
    def extract_kwargs(prefix):
        return {k.replace(f"{prefix}__", ""): v for k, v in kwargs.items() if k.startswith(f"{prefix}__")}

    # --- Parse kwargs ---
    split_kwargs = extract_kwargs("split")
    scaler_kwargs = extract_kwargs("scaler")
    cv_kwargs = extract_kwargs("cv")
    grid_kwargs = extract_kwargs("grid")
    model_common_kwargs = extract_kwargs("model")

    # Load dataset
    dataset = pd.read_csv(data_path).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Separate features and labels
    X = dataset.drop(columns=[target_column]).values
    y = dataset[target_column].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        **split_kwargs
    )

    # Feature scaling
    if scale_features:
        scaler = MinMaxScaler(**scaler_kwargs)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    results = {}

    for model_name, config in model_configs.items():
        base_model = config['model']
        param_grid = config['params']

        # Re-instantiate model if class was passed instead of object
        if inspect.isclass(base_model):
            model = base_model(**model_common_kwargs)
        else:
            model = base_model  # Assume already instantiated

        # Cross-validation strategy
        cv = StratifiedKFold(n_splits=n_splits_kfold, shuffle=True, random_state=random_state, **cv_kwargs)

        # Grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring_metric,
            cv=cv,
            n_jobs=-1,
            **grid_kwargs
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        results[model_name] = {
            "model": best_model,
            "best_params": grid_search.best_params_,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

        print(f"[OK] Trained: {model_name}")

    return results

def train_models_on_multiple_datasets(data_paths: list,
                                       model_configs: dict,
                                       target_column: str = "label",
                                       test_size: float = 0.2,
                                       scoring_metric: str = "f1",
                                       scale_features: bool = True,
                                       random_state: int = 42,
                                       n_splits_kfold: int = 5,
                                       **kwargs):
    """
    Trains multiple models across multiple datasets using grid search and evaluation.
    Passes kwargs to underlying train_models_on_dataset() function, including:
        - model__<arg>         → Model instantiation args
        - grid__<arg>          → GridSearchCV args
        - cv__<arg>            → StratifiedKFold args
        - scaler__<arg>        → Scaler args
        - split__<arg>         → train_test_split args
    """
    all_results = {}
    
    overall_start = time.time()
    
    for idx, data_path in enumerate(data_paths, start=1):
        print(f"\nTraining on Dataset {idx}: {os.path.basename(data_path)}")

        start_time = time.time()
        
        results = train_models_on_dataset(
            data_path = data_path,
            model_configs = model_configs,
            target_column = target_column,
            test_size = test_size,
            scoring_metric = scoring_metric,
            scale_features = scale_features,
            random_state = random_state,
            n_splits_kfold = n_splits_kfold,
            **kwargs  # Pass all additional control parameters
        )
        
        end_time = time.time()
        elapsed = int(end_time - start_time)
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        print(f"[TIME] Dataset {idx} finished in {hrs}h {mins}m {secs}s")
        
        file_key = os.path.basename(data_path)
        all_results[file_key] = results
    
    total_time = int(time.time() - overall_start)
    t_hrs, t_rem = divmod(total_time, 3600)
    t_mins, t_secs = divmod(t_rem, 60)
    print(f"\n[OK] All datasets completed in {t_hrs}h {t_mins}m {t_secs}s")
    
    return all_results

def perform_pca_analysis(dataset_dict, 
                         output_dir="PCA_Results", 
                         save_visuals=True, 
                         skip_existing=True,
                         n_components=None,
                         target_column=None):
    
    os.makedirs(output_dir, exist_ok=True)
    
    reduced_datasets = {}
    pca_metadata = {}
    summary_stats = {}

    for name, df in dataset_dict.items():
        logging.info(f"Processing dataset: {name}")

        # Validation
        if not isinstance(df, pd.DataFrame):
            logging.warning(f"Skipping {name}: Not a DataFrame.")
            continue
        if df.empty:
            logging.warning(f"Skipping {name}: Empty DataFrame.")
            continue

        # Drop target column if specified
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])

        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            logging.warning(f"Skipping {name}: No numeric columns after dropping target.")
            continue

        out_file = os.path.join(output_dir, f"{name}_PCA.csv")
        if skip_existing and os.path.exists(out_file):
            logging.info(f"Skipping {name}: PCA results already exist.")
            continue

        # Standardize
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df_numeric)

        # PCA
        pca = PCA(n_components=n_components) if n_components is not None else PCA()
        components = pca.fit_transform(standardized_data)
        reduced_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(components.shape[1])])
        reduced_datasets[name] = reduced_df

        # Save reduced dataset
        reduced_df.to_csv(out_file, index=False)

        # Save loadings
        loadings = pd.DataFrame(np.round(pca.components_, 4), columns=df_numeric.columns)
        loadings.to_csv(os.path.join(output_dir, f"{name}_PCA_loadings.csv"), index=False)

        # Metadata
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        eigenvalues = pca.explained_variance_

        metadata = {
            "explained_variance_ratio": np.round(explained_variance_ratio, 4).tolist(),
            "cumulative_explained_variance": np.round(cumulative_variance, 4).tolist(),
            "eigenvalues": np.round(eigenvalues, 4).tolist(),
            "n_components": int(pca.n_components_)
        }
        pca_metadata[name] = metadata
        pd.DataFrame(metadata).to_csv(os.path.join(output_dir, f"{name}_PCA_metadata.csv"), index=False)

        # Summary stats
        num_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        summary_stats[name] = {
            "Total Components": int(pca.n_components_),
            "Components for 95% Variance": int(num_components_95),
            "Max Explained Variance (%)": round(float(np.max(explained_variance_ratio) * 100), 2)
        }

        # Visualization
        if save_visuals:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, pca.n_components_ + 1), explained_variance_ratio, marker='o', label='Explained Variance Ratio')
            plt.plot(range(1, pca.n_components_ + 1), cumulative_variance, marker='s', label='Cumulative Variance')
            plt.title(f"Scree Plot for {name}")
            plt.xlabel("Principal Components")
            plt.ylabel("Variance Explained")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_ScreePlot.png"))
            plt.close()

            if pca.n_components_ >= 2:
                plt.figure(figsize=(8, 6))
                plt.scatter(components[:, 0], components[:, 1], alpha=0.7)
                plt.title(f"PCA Scatter Plot: {name}")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_PCA_2Dscatter.png"))
                plt.close()

    return reduced_datasets, pca_metadata, summary_stats

def run_experiments_with_pca_components(data_path: str,
                                        target_column: str,
                                        components_list: list,
                                        percentages: list,
                                        experiment_name: str,
                                        add_optional_path: bool,
                                        landmark_dir: str,
                                        base_output_dir: str,
                                        output_dir: str,
                                        results_save_path: str,
                                        dataset_to_use: str,
                                        homology_dimension: int = 2,
                                        test_size: float = 0.2,
                                        random_state: int = 42):

    # Load and prepare the base data
    data = pd.read_excel(os.path.abspath(data_path))
    X = data.drop(columns=[target_column, "Unnamed: 0"], errors="ignore")
    y = data[target_column]
    
    all_results = {}
    
    # Save state
    save_landmark_dir = landmark_dir
    save_base_output_dir = base_output_dir
    save_output_dir = output_dir

    for n_components in components_list:
        if add_optional_path:
            add_optional_path = f"Using_{n_components}_Components"
            landmark_dir = os.path.join(landmark_dir, f"Using_{n_components}_Components")
            base_output_dir = os.path.join(base_output_dir, f"Using_{n_components}_Components")
            output_dir = os.path.join(output_dir, f"Using_{n_components}_Components")
            print(landmark_dir)
            print(base_output_dir)
            print(output_dir)
        print(f"\n\n🚀 Running pipeline with PCA components = {n_components}")

        # Normalize
        scaler = MinMaxScaler()
        X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # PCA
        pca = PCA(n_components = n_components)
        X_reduced = pd.DataFrame(pca.fit_transform(X_normalized), columns=[f"PCA_{i+1}" for i in range(n_components)])
        variance_ratio = pca.explained_variance_ratio_.sum()
        print(f"[OK] Variance retained: {variance_ratio:.2%}")
        print("PCA Computed")

        # Reattach target
        reduced_data = X_reduced.copy()
        reduced_data["Class"] = y

        # Balance dataset
        default_data = reduced_data[reduced_data["Class"] == 1].reset_index(drop=True)
        non_default_data = reduced_data[reduced_data["Class"] == 0].reset_index(drop=True)
        n_samples = len(default_data)
        balanced_non_default = non_default_data.sample(n=n_samples, random_state=random_state)

        # Step 1: Generate Landmarks
        if add_optional_path:
            generate_landmark_sets(
                class_label_and_data={"default": default_data.drop("Class", axis=1),
                                      "non-default": balanced_non_default.drop("Class", axis=1)},
                landmark_percentages=percentages,
                dataset_to_use=dataset_to_use,
                experiment_name=experiment_name,
                add_optional_path=add_optional_path,
            )
            print("Landmark Sets Generated")
        else:
            generate_landmark_sets(
                class_label_and_data={"default": default_data.drop("Class", axis=1),
                                      "non-default": balanced_non_default.drop("Class", axis=1)},
                landmark_percentages=percentages,
                dataset_to_use=dataset_to_use,
                experiment_name=experiment_name
            )
            print("Landmark Sets Generated")

        # Step 2: Compute Barcodes
        compute_barcodes_from_multiple_landmarks(
            landmark_percentages=percentages,
            landmark_dir=landmark_dir,
            barcode_output_dir=base_output_dir,
            dim=homology_dimension,
            label={1: "default", 0: "non-default"}
        )
        print("Barcode Statistics Generated")

        # Step 3: Merge to form TDA datasets
        build_final_barcode_statistics_data(
            landmark_percentages=percentages,
            barcode_dir=base_output_dir,
            output_dir=output_dir,
            label={1: "default", 
                   0: "non-default"}
        )
        print("TDA Dataset Generated")

        # Step 4: Train models on each landmark-percentage dataset
        dataset_paths = [os.path.join(output_dir, f"data_L{p}.csv") for p in percentages]

        model_results = train_multiple_dataset_tda(
            path_datasets=dataset_paths,
            y_col_name="label",
            test_size=test_size,
            random_state=random_state,
            xgb={"eval_metric": "logloss"}
        )

        # Step 5: Store results
        store_results(
            path=results_save_path,
            save_name=f"model_results_using_{n_components}_components",
            result_object=model_results
        )

        # Save in main collection
        all_results[f"model_results_using_{n_components}_components"] = {
            "variance_retained": variance_ratio,
            "results": model_results
        }
        
        landmark_dir = save_landmark_dir
        base_output_dir = save_base_output_dir
        output_dir = save_output_dir

    return all_results

def plot_all_metrics_vs_pca_components(
    all_results: dict, 
    model_key: str, 
    save_path: str = None, 
    separate_plots: bool = False
):
    """
    Plots Accuracy, Precision, Recall, and F1 vs PCA components for each dataset (e.g., data_L5.csv).
    
    Parameters:
    - all_results: dict from `run_experiments_with_pca_components()`
    - model_key: str - the name of the model to analyze (e.g., 'knn', 'xgb', 'svm')
    - save_path: optional str - base file path or folder to save the figure(s)
    - separate_plots: bool - whether to create separate plots per dataset
    """
    # Structure: metrics_by_dataset[metric][dataset] = list of (pca_component, value)
    metrics_by_dataset = {
        "accuracy": defaultdict(list),
        "precision": defaultdict(list),
        "recall": defaultdict(list),
        "f1_score": defaultdict(list),
    }

    for experiment_name, result in all_results.items():
        try:
            pca_comp = int(experiment_name.split("_")[3])  # e.g., "PCA_5_Components" -> 5
        except (IndexError, ValueError):
            continue

        model_results = result.get("results", {})

        for dataset_name, stat in model_results.items():
            model_stat = stat.get(model_key, {})
            for metric in metrics_by_dataset:
                value = model_stat.get(metric)
                if value is not None:
                    metrics_by_dataset[metric][dataset_name].append((pca_comp, value))

    metric_titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]

    if separate_plots:
        for dataset_name in next(iter(metrics_by_dataset.values())).keys():
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"[SAVE] {model_key.upper()} Performance vs PCA Components ({dataset_name})", fontsize=16)

            for idx, ax in enumerate(axs.flat):
                metric = metric_keys[idx]
                values = metrics_by_dataset[metric][dataset_name]
                if not values:
                    continue
                sorted_vals = sorted(values, key=lambda x: x[0])
                pca_vals, metric_vals = zip(*sorted_vals)
                ax.plot(pca_vals, metric_vals, marker='o', label=dataset_name)
                ax.set_title(metric_titles[idx])
                ax.set_xlabel("PCA Components")
                ax.set_ylabel(metric_titles[idx])
                ax.set_xticks(pca_vals)
                ax.grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            if save_path:
                save_path = os.path.abspath(save_path)
                os.makedirs(save_path, exist_ok = True)
                dataset_name = dataset_name.split(".")[0]
                plt.savefig(f"{save_path}/{dataset_name}_{model_key}.png", bbox_inches="tight")
                print(f"Plot saved to: {save_path}_{dataset_name}.png")
            else:
                plt.show()
    else:
        # Combined plot (each line is a dataset)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"[SAVE] {model_key.upper()} Performance vs PCA Components (All Datasets)", fontsize=16)

        for idx, ax in enumerate(axs.flat):
            metric = metric_keys[idx]
            for dataset_name, values in metrics_by_dataset[metric].items():
                if not values:
                    continue
                sorted_vals = sorted(values, key=lambda x: x[0])
                pca_vals, metric_vals = zip(*sorted_vals)
                ax.plot(pca_vals, metric_vals, marker='o', label=dataset_name)
                ax.set_title(metric_titles[idx])
                ax.set_xlabel("PCA Components")
                ax.set_ylabel(metric_titles[idx])
                ax.set_xticks(sorted(set(pca_vals)))
                ax.grid(True)

            ax.legend(title="Dataset", fontsize='small')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            save_path = os.path.abspath(save_path)
            os.makedirs(save_path, exist_ok = True)
            plt.savefig(f"{save_path}/viz_{model_key}.png", bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    return metrics_by_dataset

def create_3d_rotation_animation(
    reduced_df,
    label_column,
    save_path,
    dataset_name,
    method_name,
    fps=20,
    frames=120,
    figsize=(8, 6),
    use_color_palette=True,
    show_legend=True,
    save_gif=True,     # 🆕 Toggle saving GIF
    save_mp4=True      # 🆕 Toggle saving MP4
):
    """
    Creates and saves an animated 3D rotation of class-separated data.

    Parameters:
    - reduced_df (DataFrame): DataFrame with Component_1, Component_2, Component_3, and class label
    - label_column (str): Column name for the class label
    - save_path (str): Directory path to save the animation
    - dataset_name (str): Name of the dataset (used in filename)
    - method_name (str): Method used (e.g., PCA, t-SNE)
    - fps (int): Frames per second of the animation
    - frames (int): Total number of frames in the animation
    - figsize (tuple): Size of the figure
    - use_color_palette (bool): Use seaborn color palette
    - show_legend (bool): Show class legend
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    classes = reduced_df[label_column].unique()
    palette = sns.color_palette("Set2") if use_color_palette else None
    color_map = dict(zip(classes, palette if palette else plt.cm.tab10.colors[:len(classes)]))

    scatters = []

    for cls in classes:
        subset = reduced_df[reduced_df[label_column] == cls]
        sc = ax.scatter(
            subset["Component_1"],
            subset["Component_2"],
            subset["Component_3"],
            label=cls,
            color=color_map[cls],
            alpha=0.8
        )
        scatters.append(sc)

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title(f"3D Class Rotation — {dataset_name} ({method_name.upper()})")
    if show_legend:
        ax.legend(title="Class")

    def update(frame):
        ax.view_init(elev=20., azim=frame)
        return scatters

    ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 360, frames), interval=1000/fps, blit=False)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        if save_mp4:
            filename = os.path.join(save_path, f"{dataset_name}_{method_name}_3d_rotation.mp4")
            try:
                writer = FFMpegWriter(fps=fps, extra_args=["-vcodec", "libx264"])
                ani.save(filename, writer=writer)
                print(f"[MP4] MP4 animation saved to: {filename}")
            except Exception as e:
                print(f"[ERR] MP4 export failed: {e}")

        if save_gif:
            filename = os.path.join(save_path, f"{dataset_name}_{method_name}_3d_rotation.gif")
            try:
                writer = PillowWriter(fps=fps)
                ani.save(filename, writer=writer)
                print(f"[GIF] GIF animation saved to: {filename}")
            except Exception as e:
                print(f"[ERR] GIF export failed: {e}")

        plt.close()
        print(f"[MP4] 3D animation saved to: {filename}")
    else:
        print("[WARN] No save_path provided. Skipping 3D animation export.")
        plt.close()

def visualize_class_separability(
    dataset_paths,
    label_column="label",
    method="pca",           # 'pca', 'tsne', or 'umap'
    save_path=None,         # where to save plots
    figsize=(8, 6),
    random_state=42,
    plot_3d=False,          # toggle 2D/3D
    density_overlay=False,  # add KDE density (2D only)
    show_legend=True,
    use_color_palette=True,
    animate_3d=False,
    animated_plot_path=None,
    fps=20,
    frames=120,
    save_mp4=True,   # enable both
    save_gif=True,
    title="Class Separability Visualization"
):
    """
    Visualizes class separability using PCA, t-SNE, or UMAP for one or more datasets.

    Parameters:
    - dataset_paths (list): List of dataset file paths (CSV format).
    - label_column (str): Name of the class label column.
    - method (str): One of 'pca', 'tsne', or 'umap'.
    - save_path (str): Directory path to save the plots. If None, shows the plots.
    - figsize (tuple): Figure size for plots.
    - random_state (int): Random state for reproducibility.
    - plot_3d (bool): If True, plots in 3D using 3 components.
    - density_overlay (bool): If True and in 2D, overlays KDE contours.
    - show_legend (bool): Toggle display of legend.
    - use_color_palette (bool): Toggle seaborn color palette usage.
    - title (str): Plot title prefix.
    """

    if method not in {"pca", "tsne", "umap"}:
        raise ValueError("Method must be 'pca', 'tsne', or 'umap'")
    if method == "umap" and not HAS_UMAP:
        raise ImportError("UMAP is not installed. Please run `pip install umap-learn`.")

    n_components = 3 if plot_3d else 2

    for path in dataset_paths:
        df = pd.read_csv(path)
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in {path}")

        features = df.drop(columns=[label_column])
        labels = df[label_column]

        # Dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=random_state, init='pca', learning_rate='auto')
        else:  # umap
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)

        reduced = reducer.fit_transform(features)
        reduced_df = pd.DataFrame(reduced, columns=[f"Component_{i+1}" for i in range(n_components)])
        reduced_df[label_column] = labels

        # Color setup
        palette = "Set2" if use_color_palette else None

        dataset_name = os.path.basename(path).replace(".csv", "")
        plot_title = f"{title} ({method.upper()}) — {dataset_name}"

        # Plotting
        if plot_3d:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            classes = reduced_df[label_column].unique()

            for cls in classes:
                subset = reduced_df[reduced_df[label_column] == cls]
                ax.scatter(
                    subset["Component_1"],
                    subset["Component_2"],
                    subset["Component_3"],
                    label=cls,
                    alpha=0.7
                )

            ax.set_title(plot_title)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
            if show_legend:
                ax.legend(title="Class")
            
            # Animating the 3D plot
            if animate_3d:
                create_3d_rotation_animation(
                    reduced_df = reduced_df,
                    label_column = "label",
                    dataset_name = dataset_name,
                    method_name = method,
                    fps = fps,
                    frames = frames,
                    save_path = animated_plot_path,
                    use_color_palette = use_color_palette,
                    show_legend = show_legend,
                    save_mp4=save_mp4,   # enable both
                    save_gif=save_gif
            )

        else:
            plt.figure(figsize=figsize)
            ax = sns.scatterplot(
                data=reduced_df,
                x="Component_1",
                y="Component_2",
                hue=label_column,
                palette=palette,
                alpha=0.6,
                edgecolor="k"
            )
            if density_overlay:
                for cls in reduced_df[label_column].unique():
                    sns.kdeplot(
                        data=reduced_df[reduced_df[label_column] == cls],
                        x="Component_1",
                        y="Component_2",
                        fill=True,
                        alpha=0.3,
                        linewidth=0,
                        levels=10,
                        thresh=0.01
                    )
            plt.title(plot_title)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            if show_legend:
                plt.legend(title="Class")
            else:
                plt.legend([], [], frameon=False)

        plt.tight_layout()

        if save_path:
            save_dir = _ensure_dir(save_path)
            suffix = "3d" if plot_3d else "2d"
            out_path = save_dir / f"{dataset_name}_{method}_{suffix}.png"
            plt.savefig(out_path, bbox_inches="tight")
            print(f"[OK] Saved: {out_path}")
            plt.close()
        else:
            plt.show()

def build_mapper_viz(
    data,
    resampled_data_label,
    resolution: list,
    percentage_overlap: list,
    clustering_grid: dict,
    lens_methods: list = ["pca"],
    lens_params: dict = None,
    color_functions: list = ["lens", "labels"],
    color_function_name: list = ["Default Status"],
    output_dir: str = ".",
    n_jobs: int = -1  # ignored now
):

    if lens_params is None:
        lens_params = {}

    dir_output = str(_ensure_dir(output_dir))
    results_log = []

    def format_params(param_dict):
        return "_".join(f"{k}_{v}" for k, v in param_dict.items())

    def run_mapper_experiment(res, overlap, cluster_type, cluster_args, lens_name, lens_args):
        mapper = KeplerMapper(verbose=0)

        # === Lens
        if lens_name == "pca":
            lens_model = PCA(**lens_args)
        elif lens_name == "umap":
            lens_model = umap.UMAP(**lens_args)
        elif lens_name == "tsne":
            lens_model = TSNE(**lens_args)
        else:
            raise ValueError(f"Unsupported lens method: {lens_name}")
        
        lens = lens_model.fit_transform(data)

        # === Clusterer
        if cluster_type == "kmeans":
            clusterer = KMeans(**cluster_args)
        elif cluster_type == "dbscan":
            clusterer = DBSCAN(**cluster_args)
        elif cluster_type == "agglomerative":
            clusterer = AgglomerativeClustering(**cluster_args)
        else:
            raise ValueError(f"Unsupported clustering type: {cluster_type}")

        # === Mapper Graph
        graph = mapper.map(
            X=data,
            lens=lens,
            cover=km.Cover(n_cubes=res, perc_overlap=overlap),
            clusterer=clusterer
        )

        # === Output Directory
        ovl_tag = str(int(round(float(overlap) * 100)))
        n_cl = cluster_args.get("n_clusters", "")
        experiment_name = f"{lens_name}_r{res}_o{ovl_tag}_{cluster_type}{n_cl}"
        experiment_path = str(_ensure_dir(os.path.join(dir_output, experiment_name)))

        # === Save Parameters
        with open(os.path.join(experiment_path, "parameters.txt"), "w") as f:
            f.write("Mapper Graph Parameters:\n")
            f.write(f"Lens: {lens_name} {lens_args}\n")
            f.write(f"Resolution: {res}\n")
            f.write(f"Overlap: {overlap}\n")
            f.write(f"Clustering: {cluster_type} {cluster_args}\n")

        # === Visualizations
        color_values = {
            "lens": lens,
            "labels": resampled_data_label.values if hasattr(resampled_data_label, "values") else resampled_data_label
        }

        output_files = {}
        for idx, color_key in enumerate(color_functions):
            file_name = f"mapper_output_{color_key}.html"
            file_path = os.path.join(experiment_path, file_name)

            mapper.visualize(
                graph,
                path_html=file_path,
                title="Mapper Graph",
                color_values=color_values[color_key],
                color_function_name=color_function_name[idx] if idx < len(color_function_name) else color_key,
                include_searchbar=True
            )

            output_files[color_key] = file_path

        # === Metadata
        meta_data = {
            "lens": lens_name,
            "lens_params": lens_args,
            "resolution": res,
            "overlap": overlap,
            "cluster_type": cluster_type,
            "cluster_params": cluster_args,
            "output_dir": experiment_path,
            "output_files": output_files
        }

        file_metadata_name = "metadata.pkl"
        file_metadata_path = os.path.join(experiment_path, file_metadata_name)
        joblib.dump(meta_data, file_metadata_path)

        return meta_data

    # === Run Experiments SEQUENTIALLY
    print("[RUN] Running experiments sequentially (no parallel processing)...")
    for res, overlap, lens_name in product(resolution, percentage_overlap, lens_methods):
        for cluster_type, cluster_list in clustering_grid.items():
            for cluster_args in cluster_list:
                lens_args = lens_params.get(lens_name, {})
                result = run_mapper_experiment(res, overlap, cluster_type, cluster_args, lens_name, lens_args)
                results_log.append(result)

    # === Save Experiment Summary
    results_df = pd.DataFrame(results_log)
    results_df.to_csv(os.path.join(dir_output, "mapper_experiments.csv"), index=False)

    with open(os.path.join(dir_output, "mapper_experiments.json"), "w") as f:
        json.dump(results_log, f, indent=4)

    print(f"[OK] Saved {len(results_log)} experiments to CSV and JSON.")

def drop_correlated_features(features: pd.DataFrame,
                             threshold: float = 0.75,
                             feature_label: bool = False,
                             drop_columns: list = None,
                             strategy: str = 'first',
                             target: pd.Series = None,
                             decimals: int = 4):
    """
    Drops highly correlated features from the DataFrame based on a chosen strategy.

    Parameters:
    - features: pandas DataFrame.
    - feature_label: bool, if original contextual column names have been assigned to each variable.
    - drop_columns: list of columns to drop before correlation check.
    - threshold: float, correlation threshold above which variables are considered redundant.
    - strategy: str, one of ['first', 'high_variance', 'low_missing', 'target_corr'].
    - target: Series, required if strategy == 'target_corr'.
    - decimals: int, number of decimals to round correlation values in output.

    Returns:
    - Reduced DataFrame.
    - Dictionary: keys are kept features, values are dicts of dropped features with correlation values, or None.
    """

    df = features.copy()

    # Drop specified columns
    dropped_initial = set()
    if drop_columns is not None:
        if feature_label:
            df = df.drop(df.columns[drop_columns], axis = 1)
        else:
            dropped_initial = set(drop_columns)
            df = df.drop(columns=drop_columns)

    # Compute correlation matrix
    corr_matrix = df.corr().abs()

    # Build graph of correlated features
    G = nx.Graph()
    G.add_nodes_from(corr_matrix.columns)

    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and corr_matrix.loc[i, j] >= threshold:
                G.add_edge(i, j)

    drop_map = {}
    to_drop = set()

    for group in nx.connected_components(G):
        group = list(group)
        print(group)
        if len(group) == 1:
            feature = group[0]
            drop_map[feature] = None
            continue

        # Select feature to keep
        if strategy == 'first':
            keep = group[0]
        elif strategy == 'high_variance':
            keep = df[group].var().idxmax()
        elif strategy == 'low_missing':
            keep = df[group].isnull().sum().idxmin()
        elif strategy == 'target_corr':
            if target is None:
                raise ValueError("You must provide a target variable for strategy='target_corr'")
            correlations = {}
            for col in group:
                try:
                    corr = abs(df[col].corr(target))
                    correlations[col] = corr if not np.isnan(corr) else 0
                except Exception:
                    correlations[col] = 0
            keep = max(correlations, key=correlations.get)
        else:
            raise ValueError("Invalid strategy.")

        group.remove(keep)
        drop_map[keep] = {
            col: round(corr_matrix.loc[keep, col], decimals)
            for col in group if col in corr_matrix.columns
        }
        to_drop.update(group)
    print("\n\n")

    # Handle any isolated uncorrelated features not in the graph
    unprocessed = set(corr_matrix.columns) - set(drop_map.keys()) - to_drop
    for col in unprocessed:
        drop_map[col] = None

    # Drop correlated features
    df_reduced = df.drop(columns=to_drop)
    
    if target is not None:
        df_reduced = df_reduced.copy()
        df_reduced["label"] = np.asarray(target)


    return df_reduced, drop_map

def visualize_correlation_drop_maps(
    drop_maps: list,
    corr_matrices: list,
    dataset_labels: list,
    save_path: str = None,
    save_png: bool = False,
    save_html: bool = False,
    use_pyvis: bool = True,
    figsize=(8, 6)
):
    """
    Visualize multiple correlation drop maps as subplots or interactive graphs.

    Parameters:
    - drop_maps: list of drop_map dicts from drop_correlated_features
    - corr_matrices: list of correlation matrices (same length as drop_maps)
    - dataset_labels: list of dataset names for titles
    - save_path: folder where plots will be saved
    - save_png: whether to save static PNG files
    - save_html: whether to save interactive HTML files
    - use_pyvis: if True, use pyvis; else use plotly
    - figsize: tuple, size of each individual figure
    """
    # Adjust save path
    save_path = os.path.abspath(save_path)
    
    for idx, (drop_map, corr_matrix, label) in enumerate(zip(drop_maps, corr_matrices, dataset_labels)):
        # Build NetworkX graph
        G = nx.Graph()
        for kept, dropped in drop_map.items():
            G.add_node(kept, label='kept')
            if dropped:
                for col, corr in dropped.items():
                    G.add_node(col, label='dropped')
                    G.add_edge(kept, col, weight=corr)

        title = f"Correlation Drop Map: {label}"
        file_label = label.lower().replace(" ", "_")

        # ---- STATIC PNG ----
        if save_png:
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(G, seed=42)

            kept_nodes = [n for n, d in G.nodes(data=True) if d['label'] == 'kept']
            dropped_nodes = [n for n, d in G.nodes(data=True) if d['label'] == 'dropped']
            nx.draw_networkx_nodes(G, pos, nodelist=kept_nodes, node_color='lightgreen', node_size=800, label='Kept')
            nx.draw_networkx_nodes(G, pos, nodelist=dropped_nodes, node_color='salmon', node_size=600, label='Dropped')

            edges = G.edges(data=True)
            weights = [d['weight'] for _, _, d in edges]
            nx.draw_networkx_edges(G, pos, width=[w * 3 for w in weights])
            nx.draw_networkx_labels(G, pos, font_size=9)

            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray')

            plt.title(title)
            plt.legend()
            plt.axis('off')

            if save_path:
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f"{file_label}.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # ---- INTERACTIVE HTML ----
        if save_html:
            os.makedirs(save_path, exist_ok=True)
            html_file = os.path.join(save_path, f"{file_label}.html")

            if use_pyvis:
                net = Network(height="750px", width="100%", notebook=False, directed=False)
                for node, d in G.nodes(data=True):
                    color = 'lightgreen' if d['label'] == 'kept' else 'salmon'
                    net.add_node(node, label=node, color=color)

                for u, v, d in G.edges(data=True):
                    net.add_edge(u, v, value=d['weight'], title=f"Corr: {d['weight']:.2f}")

                net.write_html(html_file, open_browser = True)

            else:
                edge_x = []
                edge_y = []
                weights = []
                pos = nx.spring_layout(G, seed=42)

                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                    weights.append(edge[2]['weight'])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='gray'),
                    hoverinfo='none',
                    mode='lines'
                )

                node_x = []
                node_y = []
                node_text = []
                node_color = []

                for node, attr in G.nodes(data=True):
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    node_color.append('lightgreen' if attr['label'] == 'kept' else 'salmon')

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition="bottom center",
                    hoverinfo='text',
                    marker=dict(
                        color=node_color,
                        size=12,
                        line_width=2
                    )
                )

                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title=title,
                                    titlefont_size=16,
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    xaxis=dict(showgrid=False, zeroline=False),
                                    yaxis=dict(showgrid=False, zeroline=False)
                                ))
                fig.write_html(html_file)

def get_distance_view(data, 
                      target_col='label',
                      metric='euclidean',
                      return_class_view=False, 
                      centroid_method='mean',
                      random_state=None):
    """
    Parameters:
        data (DataFrame): Data with features and target
        target_col (str): Name of class column
        return_class_view (bool): True = return DataFrame with class comparison
        centroid_method (str): 'mean', 'farthest', 'random'
        random_state (int): Seed for reproducibility (if needed)

    Returns:
        pd.DataFrame
    """
    
    def get_centroids(features_scaled, 
                      labels=None, 
                      method='mean', 
                      random_state=None):
        """
        Selects centroids based on the chosen method.
        
        Parameters:
            features_scaled: np.ndarray of shape (n_samples, n_features)
            labels: Series or array of labels (optional for unsupervised methods)
            method: 'mean', 'farthest', 'random'
            random_state: int or None
            
        Returns:
            centroids: np.ndarray of shape (2, n_features)
        """
        rng = check_random_state(random_state)

        if method == 'mean':
            if labels is None:
                raise ValueError("Labels required for method='mean'")
            class_labels = sorted(np.unique(labels))
            return np.vstack([
                features_scaled[labels == class_labels[0]].mean(axis=0),
                features_scaled[labels == class_labels[1]].mean(axis=0)
            ])

        elif method == 'farthest':
            # Compute full distance matrix and find max distance pair
            dist = squareform(pdist(features_scaled, metric=metric))
            i, j = np.unravel_index(np.argmax(dist), dist.shape)
            return np.vstack([features_scaled[i], features_scaled[j]])

        elif method == 'random':
            idxs = rng.choice(features_scaled.shape[0], size=2, replace=False)
            return features_scaled[idxs]

        else:
            raise ValueError(f"Unknown centroid method: {method}")
    
    features = data.drop(columns=[target_col])
    labels = data[target_col].reset_index(drop=True)

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    if not return_class_view:
        dist_matrix = squareform(pdist(features_scaled, metric=metric))
        return pd.DataFrame(dist_matrix)

    # Get centroids based on method
    centroids = get_centroids(features_scaled, labels if centroid_method == 'mean' else None,
                              method=centroid_method, random_state=random_state)

    # Compute distances to the centroids
    distances_to_centroids = cdist(features_scaled, centroids, metric=metric)
    closest_class = distances_to_centroids.argmin(axis=1)

    result_df = pd.DataFrame({
        'row_index': np.arange(len(data)),
        'distance_to_centroid_0': distances_to_centroids[:, 0],
        'distance_to_centroid_1': distances_to_centroids[:, 1],
        'closest_class': closest_class,
        'actual_class': labels
    })

    return result_df


# =============================================================================
# Publication-quality visualization (shared by all experiment visualizers)
# =============================================================================
VISUALIZATIONS_DIRNAME = "Visualizations"
_VIZ_DPI = 160
_OKABE_ITO = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
)
_MODEL_DISPLAY = {
    "knn": "kNN",
    "logistic": "Logistic",
    "logit": "Logistic",
    "logreg": "Logistic",
    "random_forest": "Random Forest",
    "rf": "Random Forest",
    "svm": "SVM",
    "xgb": "XGBoost",
    "xgboost": "XGBoost",
    "lgbm": "LightGBM",
    "lightgbm": "LightGBM",
    "linear": "Linear",
    "linear_regression": "Linear regression",
}
_MODEL_COLOR = {
    "knn": "#0072B2",
    "logistic": "#E69F00",
    "logit": "#E69F00",
    "logreg": "#E69F00",
    "random_forest": "#009E73",
    "rf": "#009E73",
    "svm": "#CC79A7",
    "xgb": "#D55E00",
    "xgboost": "#D55E00",
    "lgbm": "#56B4E9",
    "lightgbm": "#56B4E9",
    "linear": "#000000",
    "linear_regression": "#000000",
}
_METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 score",
    "f1": "F1 score",
    "balanced_accuracy": "Balanced accuracy",
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
}
_DATASET_SHORT = {
    "Statlog_German_Credit_Data": "Statlog German",
    "Default_Of_Credit_Card_Client_Data": "Credit card default",
    "PKDD_Czech_Financial": "PKDD Czech",
    "Polish_Bankruptcy_3Year": "Polish bankruptcy",
    "Taiwan_Bankruptcy": "Taiwan bankruptcy",
    "South_German_Credit": "South German",
}
_DATASET_FILE_SLUG = {
    "Statlog_German_Credit_Data": "Statlog_German",
    "Default_Of_Credit_Card_Client_Data": "Credit_Card_Default",
    "PKDD_Czech_Financial": "PKDD_Czech",
    "Polish_Bankruptcy_3Year": "Polish_Bankruptcy",
    "Taiwan_Bankruptcy": "Taiwan_Bankruptcy",
    "South_German_Credit": "South_German",
}
_DATASET_DISPLAY_SHORT = {
    "Statlog German Credit": "Statlog German",
    "Default of Credit Card Client": "Credit card default",
    "PKDD'99 Czech Financial": "PKDD Czech",
    "Polish Companies Bankruptcy (3 year)": "Polish bankruptcy",
    "Taiwanese Bankruptcy Prediction": "Taiwan bankruptcy",
    "South German Credit (updated-German sensitivity)": "South German",
}
_RATE_METRIC_KEYS = {
    "accuracy", "precision", "recall", "f1", "f1_score",
    "balanced_accuracy", "roc_auc", "average_precision",
}
_LANDMARK_TOKEN_RE = re.compile(r"L_?(\d+(?:\.\d+)?)", re.IGNORECASE)
_VIZ_STYLE_APPLIED = False


def _public_fs_path(path: Union[str, Path]) -> Path:
    text = os.fspath(path)
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):
        text = text[4:]
    return Path(text)


def apply_publication_viz_style() -> None:
    """Idempotent matplotlib/seaborn style for paper-readable figures."""
    global _VIZ_STYLE_APPLIED
    if _VIZ_STYLE_APPLIED:
        return
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.dpi": _VIZ_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.28,
    })
    _VIZ_STYLE_APPLIED = True


def pretty_model_label(raw: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
    if key in _MODEL_DISPLAY:
        return _MODEL_DISPLAY[key]
    return str(raw).replace("_", " ").title()


def pretty_metric_label(raw: str) -> str:
    key = str(raw).strip().lower()
    if key in _METRIC_DISPLAY:
        return _METRIC_DISPLAY[key]
    return str(raw).replace("_", " ").capitalize()


def pretty_setting_label(raw: str) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "default", "main"}:
        return ""
    if "|" in text:
        text = text.split("|")[-1].strip()
    if "=" in text:
        key, _, val = text.partition("=")
        key, val = key.strip().lower(), val.strip()
        if "hist" in val.lower():
            return "Historical protocol"
        if "clean" in val.lower():
            return "Clean protocol"
        if key in {"protocol", "variant", "setting", "feature_space", "run_key"}:
            return val.replace("_", " ").title()
        text = val
    name = Path(text.replace("\\", "/")).name
    name = re.sub(r"\.(csv|pkl|json)$", "", name, flags=re.IGNORECASE)
    lower = name.lower()
    if "protocol_historical" in lower or lower in {"historical", "protocol historical"}:
        return "Historical protocol"
    if "protocol_clean" in lower or lower in {"clean", "protocol clean"}:
        return "Clean protocol"
    match = _LANDMARK_TOKEN_RE.search(name)
    if match:
        return f"{match.group(1)}% of class"
    return name.replace("_", " ")


def _registry_folder_from_token(token: str) -> Optional[str]:
    token = str(token).strip()
    if not token:
        return None
    try:
        return get_dataset_config(token).folder_name
    except (ValueError, KeyError):
        return None


def parse_group_key(raw: str) -> Tuple[str, str]:
    """Split a model-results group key into (dataset_folder, setting)."""
    text = str(raw).strip()
    dataset, setting = text, ""
    if ":" in text:
        dataset, setting = text.split(":", 1)
    if "|" in dataset and not _registry_folder_from_token(dataset):
        left, right = dataset.split("|", 1)
        if _registry_folder_from_token(left.strip()):
            extra = right.strip()
            setting = f"{extra} | {setting}".strip(" |") if setting else extra
            dataset = left.strip()
    if _registry_folder_from_token(dataset):
        return _registry_folder_from_token(dataset), setting
    if re.search(r"data_L", dataset, re.IGNORECASE) or dataset.lower().endswith((".csv", ".pkl")):
        return "", dataset
    return dataset, setting


def pretty_dataset_label(raw: str, short: bool = False) -> str:
    if raw is None:
        return "Unknown dataset"
    text = str(raw).strip()
    if not text:
        return "Unknown dataset"
    folder = _registry_folder_from_token(text)
    if folder is None:
        for part in re.split(r"[\\/,:|]+", text):
            folder = _registry_folder_from_token(part)
            if folder:
                break
    if folder:
        if short:
            return _DATASET_SHORT.get(folder, folder.replace("_", " "))
        try:
            display = get_dataset_config(folder).display_name
        except (ValueError, KeyError):
            display = folder.replace("_", " ")
        return _DATASET_DISPLAY_SHORT.get(display, display) if short else display
    cleaned = text.replace("_", " ")
    return _DATASET_DISPLAY_SHORT.get(cleaned, cleaned)


def dataset_slug_for_filename(raw: str) -> str:
    folder = _registry_folder_from_token(raw)
    if folder:
        return _DATASET_FILE_SLUG.get(folder, folder)
    for part in re.split(r"[\\/,:]+", str(raw)):
        folder = _registry_folder_from_token(part)
        if folder:
            return _DATASET_FILE_SLUG.get(folder, folder)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(raw)).strip("_")
    return slug or "dataset"


def _is_rate_metric(name: str) -> bool:
    key = str(name).strip().lower()
    return key in _RATE_METRIC_KEYS


def _model_palette(models: List[str]) -> Dict[str, str]:
    palette: Dict[str, str] = {}
    extra_i = 0
    for model in models:
        key = re.sub(r"[^a-z0-9]+", "_", str(model).strip().lower()).strip("_")
        if key in _MODEL_COLOR:
            palette[model] = _MODEL_COLOR[key]
        else:
            palette[model] = _OKABE_ITO[extra_i % len(_OKABE_ITO)]
            extra_i += 1
    return palette


def _categorical_palette(levels: List[str]) -> Dict[str, str]:
    return {level: _OKABE_ITO[i % len(_OKABE_ITO)] for i, level in enumerate(levels)}


def _save_figure(fig: plt.Figure, path: Union[str, Path]) -> Path:
    path = win_long_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(os.fspath(path), dpi=_VIZ_DPI, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return _public_fs_path(path)


def _write_csv(path: Union[str, Path], frame: pd.DataFrame) -> Path:
    path = win_long_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(os.fspath(path), index=False)
    return _public_fs_path(path)


def _barplot(ax, data: pd.DataFrame, x: str, y: str, hue: Optional[str], palette, order=None, hue_order=None):
    plot_hue = hue if hue is not None else x
    plot_hue_order = hue_order if hue is not None else order
    kwargs = dict(
        data=data, x=x, y=y, hue=plot_hue, ax=ax,
        palette=palette, order=order, hue_order=plot_hue_order,
    )
    if hue is None:
        kwargs["legend"] = False
    try:
        return sns.barplot(errorbar=None, **kwargs)
    except TypeError:
        kwargs.pop("legend", None)
        try:
            return sns.barplot(ci=None, **kwargs)
        except TypeError:
            return sns.barplot(**kwargs)


def _style_rate_axis(ax, ylabel: str, is_rate: bool) -> None:
    ax.set_ylabel(ylabel, fontsize=12)
    if is_rate:
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.linspace(0, 1, 6))


def _set_wrapped_ticks(ax, labels: List[str], width: int = 16) -> None:
    wrapped = ["\n".join(textwrap.wrap(str(lab), width=width)) if lab else "" for lab in labels]
    ax.set_xticks(range(len(wrapped)))
    rotate = 40 if len(labels) > 5 else 0
    ax.set_xticklabels(
        wrapped,
        fontsize=10,
        rotation=rotate,
        ha="right" if rotate else "center",
    )


def _place_legend_outside(fig: plt.Figure, handles, labels, title: Optional[str] = None, ncol: int = 4) -> None:
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(ncol, max(1, len(labels))),
        frameon=False,
        fontsize=10,
    )


def _hide_unused_axes(axes, used: int) -> None:
    for ax in axes[used:]:
        ax.set_visible(False)


def protocol_context_sentence(protocol_bucket: str) -> str:
    """One-sentence protocol reminder for figure footnotes (English names, not compact t/l symbols)."""
    if not protocol_bucket or protocol_bucket == "Default_Parameters":
        return "These scores use the original tabular features, not barcode statistics."
    if protocol_bucket == "Statistics":
        return "Estimates are computed on the processed table, before any TDA snapshots."
    key = resolve_protocol_bucket(protocol_bucket)
    spec = TDA_PROTOCOL_SPECS.get(key) or {}
    parts = []
    display = spec.get("display_name")
    if display:
        parts.append(f"Process: {display}.")
    if spec.get("split_timing") == "early":
        parts.append("Customers are split into train and test before PCA.")
    elif spec.get("split_timing") == "late":
        parts.append("PCA is fit on the full table; the train/test split is applied to barcode rows afterwards.")
    if spec.get("undersample") is True:
        parts.append("The majority class is undersampled to the minority class count.")
    elif spec.get("undersample") is False:
        parts.append("No undersampling: both class pools keep their original sizes.")
    if spec.get("homology") == "H0":
        parts.append("Barcode tables keep homology-0 statistics only.")
    elif spec.get("homology") == "H0_and_H1":
        parts.append("Barcode tables keep both homology-0 and homology-1 statistics.")
    return " ".join(parts)


def _finish_and_save(
    fig: plt.Figure,
    save_path: Path,
    *,
    note: str = "",
    handles=None,
    labels=None,
    legend_title: Optional[str] = None,
    left: float = 0.02,
) -> Path:
    """Legend above a wrapped methodology footnote, then save at 160 dpi."""
    keep = []
    for handle, label in zip(handles or [], labels or []):
        if label:
            keep.append((handle, str(label)))
    if len(keep) <= 1:
        keep = []
    note = " ".join(str(note or "").split())
    wrapped = textwrap.wrap(note, width=108) if note else []
    n_lines = len(wrapped)
    legend_h = 0.09 if keep else 0.0
    note_h = (0.028 * n_lines + 0.045) if wrapped else 0.0
    bottom = min(0.48, legend_h + note_h)
    fig.tight_layout(rect=[left, bottom, 0.98, 0.95])
    y = 0.012
    if wrapped:
        fig.text(
            0.5, y,
            "\n".join(wrapped),
            ha="center", va="bottom", fontsize=9, color="#222222",
            linespacing=1.3,
        )
        y += max(0.04, note_h - 0.02)
    if keep:
        fig.legend(
            [h for h, _ in keep],
            [lab for _, lab in keep],
            title=legend_title,
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            ncol=min(6, max(1, len(keep))),
            frameon=False,
            fontsize=10,
        )
    return _save_figure(fig, save_path)


def plot_faceted_bars(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    facet: str,
    title: str,
    ylabel: str,
    save_path: Path,
    hue: Optional[str] = None,
    hline: Optional[float] = None,
    hline_label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    yscale: Optional[str] = None,
    x_order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    palette: Optional[Dict[str, str]] = None,
    annotate: bool = False,
    wrap_width: int = 14,
    note: str = "",
    xlabel: Optional[str] = None,
    share_x: bool = True,
) -> Path:
    """One primary question, small multiples: one panel per facet level."""
    apply_publication_viz_style()
    data = frame.dropna(subset=[x, y, facet]).copy()
    if data.empty:
        raise ValueError(f"No rows to plot for {Path(save_path).name}")
    facets = list(dict.fromkeys(data[facet].tolist()))
    x_levels = x_order or list(dict.fromkeys(data[x].tolist()))
    hue_levels = None
    if hue:
        hue_levels = hue_order or list(dict.fromkeys(data[hue].tolist()))
        if palette is None:
            palette = _categorical_palette(hue_levels)
    elif palette is None:
        palette = _categorical_palette(x_levels)

    n = len(facets)
    ncols = 3 if n > 4 else (2 if n > 1 else 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.7 * nrows + 0.35), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    is_rate = ylim == (0, 1.05) or _is_rate_metric(y) or _is_rate_metric(ylabel)

    handles, labels = [], []
    for i, facet_val in enumerate(facets):
        ax = axes[i]
        sub = data[data[facet] == facet_val]
        local_hues = list(dict.fromkeys(sub[hue].tolist())) if hue else None
        local_x = list(x_levels)
        if not share_x:
            local_x = list(dict.fromkeys(sub[x].tolist()))
            try:
                local_x = sorted(local_x, key=lambda v: float(v))
            except (TypeError, ValueError):
                pass
        _barplot(
            ax, sub, x, y, hue,
            palette=(
                palette if hue
                else [palette.get(v, _OKABE_ITO[j % len(_OKABE_ITO)]) for j, v in enumerate(local_x)]
                if isinstance(palette, dict)
                else palette
            ),
            order=local_x,
            hue_order=local_hues,
        )
        ax.set_title(str(facet_val), fontsize=12)
        ax.set_xlabel(xlabel if xlabel and i >= n - ncols else "")
        if is_rate and yscale is None:
            _style_rate_axis(ax, "", True)
        else:
            if ylim is not None:
                ax.set_ylim(*ylim)
        ax.set_ylabel("")
        if yscale:
            ax.set_yscale(yscale)
        if hline is not None:
            ax.axhline(hline, color="#222222", linestyle="--", linewidth=1.0)
        legend = ax.get_legend()
        if legend is not None:
            new_handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
            new_labels = [t.get_text() for t in legend.get_texts()]
            for handle, label in zip(new_handles, new_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            legend.remove()
        _set_wrapped_ticks(ax, [str(v) for v in local_x], width=wrap_width)
        if annotate and sub[x].nunique() <= 6 and (hue is None or sub[hue].nunique() <= 2):
            for patch in ax.patches:
                height = patch.get_height()
                if np.isfinite(height) and height > 0:
                    ax.annotate(
                        f"{height:.2f}",
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                        textcoords="offset points",
                    )

    _hide_unused_axes(axes, n)
    fig.suptitle(title, fontsize=14, y=1.03)
    if ylabel:
        fig.supylabel(ylabel, fontsize=12)
    if hline is not None and hline_label:
        labels = list(labels) + [hline_label]
        from matplotlib.lines import Line2D
        handles = list(handles) + [Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.0)]
    return _finish_and_save(fig, save_path, note=note, handles=handles, labels=labels, left=0.11)


def plot_grouped_bars(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    save_path: Path,
    hue: Optional[str] = None,
    hline: Optional[float] = None,
    hline_label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    yscale: Optional[str] = None,
    x_order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    palette: Optional[Dict[str, str]] = None,
    annotate: bool = False,
    wrap_width: int = 16,
    note: str = "",
) -> Path:
    """Single-panel bar chart with legend outside the axes."""
    apply_publication_viz_style()
    data = frame.dropna(subset=[x, y]).copy()
    if data.empty:
        raise ValueError(f"No rows to plot for {Path(save_path).name}")
    x_levels = x_order or list(dict.fromkeys(data[x].tolist()))
    hue_levels = hue_order or (list(dict.fromkeys(data[hue].tolist())) if hue else None)
    if palette is None:
        palette = _categorical_palette(hue_levels if hue else x_levels)
    width = max(7.5, 0.7 * len(x_levels) + 2.8)
    fig, ax = plt.subplots(figsize=(width, 5.2))
    _barplot(ax, data, x, y, hue, palette=palette, order=x_levels, hue_order=hue_levels)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("")
    is_rate = ylim == (0, 1.05) or _is_rate_metric(y) or _is_rate_metric(ylabel)
    if is_rate and yscale is None:
        _style_rate_axis(ax, ylabel, True)
    else:
        ax.set_ylabel(ylabel, fontsize=12)
        if ylim is not None:
            ax.set_ylim(*ylim)
    if yscale:
        ax.set_yscale(yscale)
    if hline is not None:
        ax.axhline(hline, color="#222222", linestyle="--", linewidth=1.0, label=hline_label or None)
    legend = ax.get_legend()
    handles, labels = [], []
    if legend is not None:
        handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
        labels = [t.get_text() for t in legend.get_texts()]
        legend.remove()
    if hline is not None and hline_label and hline_label not in labels:
        from matplotlib.lines import Line2D
        handles = list(handles) + [Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.0)]
        labels = list(labels) + [hline_label]
    _set_wrapped_ticks(ax, [str(v) for v in x_levels], width=wrap_width)
    if annotate and len(x_levels) <= 8 and (hue is None or (hue_levels is not None and len(hue_levels) <= 2)):
        for patch in ax.patches:
            height = patch.get_height()
            if np.isfinite(height) and height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2),
                    textcoords="offset points",
                )
    if handles:
        return _finish_and_save(fig, save_path, note=note, handles=handles, labels=labels)
    return _finish_and_save(fig, save_path, note=note)


def _model_results_long_frame(model_results: dict, default_dataset: str = "") -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    rows = []
    for group_key, models in (model_results or {}).items():
        if not isinstance(models, dict):
            continue
        dataset_folder, setting = parse_group_key(group_key)
        if not _registry_folder_from_token(dataset_folder):
            if default_dataset:
                setting = setting or dataset_folder or str(group_key)
                dataset_folder = default_dataset
            elif _registry_folder_from_token(default_dataset):
                dataset_folder = default_dataset
        dataset_folder = _registry_folder_from_token(dataset_folder) or dataset_folder
        setting_label = pretty_setting_label(setting)
        dataset_label = pretty_dataset_label(dataset_folder or group_key, short=True)
        dataset_full = pretty_dataset_label(dataset_folder or group_key, short=False)
        slug = dataset_slug_for_filename(dataset_folder or default_dataset or group_key)
        for model_name, stats in models.items():
            if not isinstance(stats, dict):
                continue
            stats = _normalize_model_stats(stats) if "f1" in stats or "f1_score" in stats else stats
            for metric in metrics:
                value = stats.get(metric)
                if value is None and metric == "f1_score":
                    value = stats.get("f1")
                if value is None or not np.isfinite(float(value)):
                    continue
                rows.append({
                    "group_key": group_key,
                    "dataset_folder": dataset_folder or slug,
                    "dataset_label": dataset_label,
                    "dataset_full": dataset_full,
                    "setting": setting_label or "Default pipeline",
                    "model": str(model_name),
                    "model_label": pretty_model_label(model_name),
                    "metric": metric,
                    "metric_label": pretty_metric_label(metric),
                    "value": float(value),
                })
    return pd.DataFrame(rows)


def _write_metric_csvs(frame: pd.DataFrame, save_dir: Path, prefix: str = "") -> List[Path]:
    written: List[Path] = []
    stem = f"{prefix}_" if prefix else ""
    save_dir = win_long_path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    export = frame.copy()
    if not export.empty:
        written.append(_write_csv(save_dir / f"{stem}test_metrics_summary.csv", export))
        for metric, sub in export.groupby("metric"):
            written.append(_write_csv(save_dir / f"{stem}{metric}_summary.csv", sub))
    return written


def _plot_dataset_metric_dashboard(frame: pd.DataFrame, save_path: Path, dataset_title: str, note: str = "", protocol_bucket: str = "") -> Path:
    apply_publication_viz_style()
    metrics = [m for m in ("accuracy", "precision", "recall", "f1_score") if m in set(frame["metric"])]
    models = list(dict.fromkeys(frame["model"].tolist()))
    settings = list(dict.fromkeys(frame["setting"].tolist()))
    use_hue = len(settings) > 1
    palette = _model_palette(models) if not use_hue else _categorical_palette(settings)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), sharey=True)
    axes = axes.ravel()
    handles, labels = [], []
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = frame[frame["metric"] == metric]
        _barplot(
            ax, sub, "model_label", "value",
            hue="setting" if use_hue else None,
            palette=palette if use_hue else [palette.get(m, _OKABE_ITO[0]) for m in models],
            order=[pretty_model_label(m) for m in models],
            hue_order=settings if use_hue else None,
        )
        ax.set_title(pretty_metric_label(metric), fontsize=13)
        ax.set_xlabel("")
        _style_rate_axis(ax, "Score" if i % 2 == 0 else "", True)
        legend = ax.get_legend()
        if legend is not None:
            if not handles:
                handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
                labels = [t.get_text() for t in legend.get_texts()]
            legend.remove()
        ax.tick_params(axis="x", labelsize=10)
        if not use_hue and sub["model"].nunique() <= 6:
            for patch in ax.patches:
                height = patch.get_height()
                if np.isfinite(height) and height > 0:
                    ax.annotate(
                        f"{height:.2f}",
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 1),
                        textcoords="offset points",
                    )
    for j in range(len(metrics), 4):
        axes[j].set_visible(False)
    fig.suptitle(process_figure_title(protocol_bucket, f"Held-out test metrics — {dataset_title}"), fontsize=14, y=1.02)
    legend_title = "Snapshot size" if use_hue else None
    return _finish_and_save(
        fig, save_path, note=note, handles=handles, labels=labels, legend_title=legend_title
    )


def improved_visualize_model_results(
    model_results: dict,
    save_dir: str = "results/visualizations",
    export_metrics: bool = True,
    plot_precision_recall: bool = False,
    hide_axis_labels: bool = False,
    compare_datasets: bool = False,
    colormap: str = "tab10",
    filename_prefix: str = "",
    protocol_bucket: str = "",
    figure_note: str = "",
):
    """Publication-quality test-metric figures. One question per figure.

    Cross-dataset calls write faceted small-multiples (one metric per file).
    Single-dataset calls write a 2x2 metric dashboard with consistent model colors.
    CV results are never mixed into these figures.
    """
    del plot_precision_recall, hide_axis_labels, colormap  # kept for call-site compatibility
    apply_publication_viz_style()
    save_path = win_long_path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    frame = _model_results_long_frame(
        model_results,
        default_dataset=filename_prefix if filename_prefix not in {"", "cross"} else "",
    )
    if frame.empty:
        return []

    written: List[Path] = []
    prefix = filename_prefix.strip("_")
    if prefix and prefix != "cross":
        prefix = dataset_slug_for_filename(prefix)
    if export_metrics:
        written.extend(_write_metric_csvs(frame, save_path, prefix))

    datasets = list(dict.fromkeys(frame["dataset_folder"].tolist()))
    settings = list(dict.fromkeys(frame["setting"].tolist()))
    models = list(dict.fromkeys(frame["model"].tolist()))
    model_labels = [pretty_model_label(m) for m in models]
    use_setting_hue = len(settings) > 1
    palette = _model_palette(models)
    context = figure_note or protocol_context_sentence(protocol_bucket)

    if compare_datasets and len(datasets) >= 2:
        metric_file = {
            "accuracy": "accuracy_by_model_faceted.png",
            "precision": "precision_by_model_faceted.png",
            "recall": "recall_by_model_faceted.png",
            "f1_score": "f1_by_model_faceted.png",
        }
        for metric, filename in metric_file.items():
            sub = frame[frame["metric"] == metric].copy()
            if sub.empty:
                continue
            written.append(plot_faceted_bars(
                sub,
                x="model_label",
                y="value",
                facet="dataset_label",
                hue="setting" if use_setting_hue else None,
                title=process_figure_title(protocol_bucket, f"Held-out {pretty_metric_label(metric)} by model"),
                ylabel=pretty_metric_label(metric),
                save_path=_public_fs_path(save_path) / filename,
                ylim=(0, 1.05),
                x_order=model_labels,
                hue_order=settings if use_setting_hue else None,
                palette=_categorical_palette(settings) if use_setting_hue else {pretty_model_label(m): palette[m] for m in models},
                annotate=False,
                wrap_width=12,
                note=(
                    f"Each bar is held-out test-set {pretty_metric_label(metric)} for one classifier. "
                    "Datasets are separate panels so models are comparable within a table. "
                    f"{context} "
                    "When two colours appear they mark different snapshot sizes "
                    "(fraction of the class used as points per snapshot) or table-cleaning protocols, not different metrics."
                ),
            ))
        return written

    for dataset_folder, sub in frame.groupby("dataset_folder", sort=False):
        slug = dataset_slug_for_filename(dataset_folder)
        title = sub["dataset_full"].iloc[0]
        written.append(_plot_dataset_metric_dashboard(
            sub,
            save_path=_public_fs_path(save_path) / f"{slug}_test_metrics_by_model.png",
            dataset_title=title,
            protocol_bucket=protocol_bucket,
            note=(
                f"Each panel is one held-out test-set metric on a 0-1 scale. "
                f"Bars are classifiers on {title}. {context} "
                "Cross-validation accuracy is plotted in separate figures and is not mixed in here."
            ),
        ))
    return written


def visualize_cross_validation_detailed(
    cross_val_results: Dict[str, Dict[str, Dict[str, Any]]],
    save_dir: str = "results/visualizations",
    colormap: str = "tab10",
    compare_models: bool = False,
    protocol_bucket: str = "",
    figure_note: str = "",
):
    """CV-only figures: fold accuracy (faceted by model) and mean±std by model."""
    del colormap, compare_models
    apply_publication_viz_style()
    save_path = win_long_path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_dir = _public_fs_path(save_path)
    paths: List[Path] = []

    for dataset_name, models_data in (cross_val_results or {}).items():
        if not isinstance(models_data, dict):
            continue
        slug = dataset_slug_for_filename(dataset_name)
        dataset_title = pretty_dataset_label(dataset_name, short=False)
        fold_rows = []
        summary_rows = []
        for model_name, stats in models_data.items():
            if not isinstance(stats, dict):
                continue
            raw_scores = stats.get("cross_val_scores")
            if raw_scores is None:
                scores = []
            else:
                scores = [float(v) for v in np.asarray(raw_scores).ravel().tolist()]
            mean_score = stats.get("mean_accuracy", stats.get("mean_accracy"))
            std_score = stats.get("std_accuracy", 0.0)
            if mean_score is None and scores:
                mean_score = float(np.mean(scores))
            if not scores and mean_score is None:
                continue
            for i, score in enumerate(scores, start=1):
                fold_rows.append({
                    "model": str(model_name),
                    "model_label": pretty_model_label(model_name),
                    "fold": f"Fold {i}",
                    "fold_num": i,
                    "accuracy": float(score),
                })
            summary_rows.append({
                "model": str(model_name),
                "model_label": pretty_model_label(model_name),
                "mean_accuracy": float(mean_score or 0.0),
                "std_accuracy": float(std_score or 0.0),
            })
        if not summary_rows:
            continue
        summary = pd.DataFrame(summary_rows)
        models = list(summary["model"])
        palette = _model_palette(models)

        if fold_rows:
            folds = pd.DataFrame(fold_rows)
            n_models = folds["model_label"].nunique()
            ncols = 3 if n_models > 4 else (2 if n_models > 1 else 1)
            nrows = int(np.ceil(n_models / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), sharey=True)
            axes = np.atleast_1d(axes).ravel()
            for i, model_label in enumerate(dict.fromkeys(folds["model_label"].tolist())):
                ax = axes[i]
                sub = folds[folds["model_label"] == model_label]
                model_key = sub["model"].iloc[0]
                ax.bar(sub["fold"], sub["accuracy"], color=palette.get(model_key, _OKABE_ITO[0]), width=0.72)
                mean_val = float(summary.loc[summary["model_label"] == model_label, "mean_accuracy"].iloc[0])
                std_val = float(summary.loc[summary["model_label"] == model_label, "std_accuracy"].iloc[0])
                ax.axhline(mean_val, color="#222222", linestyle="--", linewidth=1.0)
                ax.set_title(f"{model_label}  (mean {mean_val:.3f} ± {std_val:.3f})", fontsize=12)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Accuracy" if i % ncols == 0 else "")
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelrotation=0, labelsize=9)
            _hide_unused_axes(axes, n_models)
            fig.suptitle(process_figure_title(protocol_bucket, f"Cross-validation fold accuracy — {dataset_title}"), fontsize=14, y=1.03)
            paths.append(_finish_and_save(
                fig,
                out_dir / f"cv_{slug}_accuracy_by_fold.png",
                note=(
                    "Each bar is classification accuracy on one cross-validation fold. "
                    "The dashed line is the mean across folds. "
                    "This is training-set resampling, not the held-out test score. "
                    f"{figure_note or protocol_context_sentence(protocol_bucket)}"
                ),
            ))

        fig, ax = plt.subplots(figsize=(max(7.5, 1.15 * len(summary)), 5.0))
        x_pos = np.arange(len(summary))
        colors = [palette.get(m, _OKABE_ITO[0]) for m in summary["model"]]
        ax.bar(
            x_pos,
            summary["mean_accuracy"],
            yerr=summary["std_accuracy"],
            color=colors,
            capsize=5,
            width=0.66,
            ecolor="#333333",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary["model_label"], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean CV accuracy")
        ax.set_title(process_figure_title(protocol_bucket, f"Mean cross-validation accuracy — {dataset_title}"), fontsize=14)
        paths.append(_finish_and_save(
            fig,
            out_dir / f"cv_{slug}_mean_accuracy_by_model.png",
            note=(
                "Each bar is mean cross-validation accuracy; whiskers are one standard deviation across folds. "
                "Compare this with the held-out test dashboards — they answer different questions. "
                f"{figure_note or protocol_context_sentence(protocol_bucket)}"
            ),
        ))
    return paths

def build_results_dataframe_v3(
    full_experiment_dict: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    
    def extract_sampling_from_filename(filename: str) -> str:
        """
        Extracts the full sampling token from filenames like:
        - data_L5.csv
        - data_L_1.87.csv
        - anyprefix_Lxx.csv
        - anyprefix_L_xx.xx_something.csv
    
        Returns:
        - Full sampling token (e.g., "L5", "L_1.87"), or np.nan if not found
        """
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]
    
        match = re.search(r"(L_?\d+(?:\.\d+)?)", name_without_ext, re.IGNORECASE)
        return match.group(1) if match else np.nan
    
    rows = []
    data = []

    for experiment, exp_content in full_experiment_dict.items():
        description = exp_content.get("DESCRIPTION", "No Description")
        dataset_results = exp_content.get("RESULT", {})

        for dataset_name, sampling_dict in dataset_results.items():
            for sampling_key, models in sampling_dict.items():
                sampling = extract_sampling_from_filename(sampling_key)

                for model_name, metrics in models.items():
                    filtered_metrics = {
                        metric.capitalize(): round(value, 3) if isinstance(value, (int, float)) else value
                        for metric, value in metrics.items()
                        if metric.lower() in {"accuracy", "precision", "recall", "f1_score"}
                    }
                    
                    for metric in {"Accuracy", "Precision", "Recall", "F1_score"}:
                        filtered_metrics.setdefault(metric, np.nan)

                    row_key = (
                        experiment,
                        description,
                        dataset_name,
                        sampling,
                        model_name.upper(),
                    )
                    rows.append(row_key)
                    data.append(filtered_metrics)

    index = pd.MultiIndex.from_tuples(
        rows, names=["Exp.", "Desc.", "Dataset", "Sampling", "Model"]
    )
    df = pd.DataFrame(data, index=index)

    return df


# =============================================================================
# Early train/test split helpers (Experiment 23+)
# =============================================================================
def stratified_early_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified 80/20 (or custom) split on tabular features BEFORE PCA / landmarks.
    Returns X_train, X_test, y_train, y_test.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def fit_scaler_pca_on_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int,
    random_state: int = 42,
):
    """
    Fit MinMaxScaler + PCA on TRAIN only; transform train and test.
    Avoids leakage from fitting PCA on the full dataset.
    """
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    pca = PCA(n_components=n_components, random_state=random_state)
    pca_cols = [f"PCA_{i}" for i in range(1, n_components + 1)]
    X_train_pca = pd.DataFrame(
        pca.fit_transform(X_train_scaled),
        columns=pca_cols,
        index=X_train.index,
    )
    X_test_pca = pd.DataFrame(
        pca.transform(X_test_scaled),
        columns=pca_cols,
        index=X_test.index,
    )
    variance_ratio = float(pca.explained_variance_ratio_.sum())
    return X_train_pca, X_test_pca, scaler, pca, variance_ratio


def balance_binary_by_undersampling(
    features: pd.DataFrame,
    labels: pd.Series,
    positive_label: int = 1,
    random_state: int = 42,
):
    """
    Undersample the majority class to match the minority count.
    Returns a feature DataFrame with a 'Class' column.
    """
    data = features.copy()
    data["Class"] = labels.values
    pos = data[data["Class"] == positive_label].reset_index(drop=True)
    neg = data[data["Class"] != positive_label].reset_index(drop=True)
    n = min(len(pos), len(neg))
    pos_b = pos.sample(n=n, random_state=random_state).reset_index(drop=True)
    neg_b = neg.sample(n=n, random_state=random_state).reset_index(drop=True)
    return pd.concat([pos_b, neg_b], ignore_index=True)


def train_dataset_tda_presplit(
    train_data: Union[str, pd.DataFrame],
    test_data: Union[str, pd.DataFrame],
    y_col_name: str = "label",
    scale_features: bool = True,
    random_state: int = 42,
    **kwargs,
):
    """
    Train on a pre-built train barcode matrix; evaluate on a pre-built test matrix.
    Does NOT re-split (for early-split Experiment 23 protocol B).
    """
    if isinstance(train_data, str):
        train_df = pd.read_csv(os.path.abspath(train_data))
    else:
        train_df = train_data.copy()
    if isinstance(test_data, str):
        test_df = pd.read_csv(os.path.abspath(test_data))
    else:
        test_df = test_data.copy()

    X_train = train_df.drop(columns=[y_col_name])
    y_train = train_df[y_col_name]
    X_test = test_df.drop(columns=[y_col_name])
    y_test = test_df[y_col_name]

    if scale_features:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values

    models = {
        "svm": SVC(**kwargs.get("svm", {})),
        "knn": KNeighborsClassifier(**kwargs.get("knn", {})),
        "xgb": XGBClassifier(**kwargs.get("xgb", {})),
        "logistic": LogisticRegression(**kwargs.get("logistic", {})),
        "random_forest": RandomForestClassifier(**kwargs.get("random_forest", {})),
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        print(f"[OK] Trained {model_name} (presplit)")
    return results


def train_multiple_dataset_tda_presplit(
    train_test_pairs: Dict[str, Dict[str, str]],
    y_col_name: str = "label",
    scale_features: bool = True,
    random_state: int = 42,
    **kwargs,
):
    """
    train_test_pairs: {
        "data_L5": {"train": path_train_csv, "test": path_test_csv},
        ...
    }
    """
    model_results = {}
    overall_start = time.time()
    for name, paths in train_test_pairs.items():
        print(f"\n\n[RUN] Presplit training on: {name}")
        start = time.time()
        model_results[name] = train_dataset_tda_presplit(
            train_data=paths["train"],
            test_data=paths["test"],
            y_col_name=y_col_name,
            scale_features=scale_features,
            random_state=random_state,
            **kwargs,
        )
        elapsed = int(time.time() - start)
        print(f"[TIME] Finished {name} in {elapsed}s")
    total = int(time.time() - overall_start)
    print(f"\n[OK] All presplit datasets completed in {total}s")
    return model_results


def train_models_on_presplit_dataset(
    train_path: str,
    test_path: str,
    model_configs: dict,
    target_column: str = "label",
    scoring_metric: str = "f1",
    scale_features: bool = True,
    random_state: int = 42,
    n_splits_kfold: int = 5,
):
    """
    GridSearchCV on TRAIN barcodes only; evaluate best models on TEST barcodes.
    """
    train_df = pd.read_csv(os.path.abspath(train_path))
    test_df = pd.read_csv(os.path.abspath(test_path))

    X_train = train_df.drop(columns=[target_column]).values
    y_train = train_df[target_column].values
    X_test = test_df.drop(columns=[target_column]).values
    y_test = test_df[target_column].values

    if scale_features:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    results = {}
    cv = StratifiedKFold(
        n_splits=n_splits_kfold, shuffle=True, random_state=random_state
    )
    for model_name, config in model_configs.items():
        base_model = config["model"]
        param_grid = config["params"]
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring_metric,
            cv=cv,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        results[model_name] = {
            "model": best_model,
            "best_params": grid_search.best_params_,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        print(f"[OK] Tuned (presplit): {model_name}")
    return results


def train_models_on_multiple_presplit_datasets(
    train_test_pairs: Dict[str, Dict[str, str]],
    model_configs: dict,
    target_column: str = "label",
    scoring_metric: str = "f1",
    scale_features: bool = True,
    random_state: int = 42,
    n_splits_kfold: int = 5,
):
    model_results = {}
    for name, paths in train_test_pairs.items():
        print(f"\n\n[RUN] Presplit tuned training on: {name}")
        model_results[name] = train_models_on_presplit_dataset(
            train_path=paths["train"],
            test_path=paths["test"],
            model_configs=model_configs,
            target_column=target_column,
            scoring_metric=scoring_metric,
            scale_features=scale_features,
            random_state=random_state,
            n_splits_kfold=n_splits_kfold,
        )
    return model_results


# =============================================================================
# Statistical audit helpers (Experiments 24–27)
# =============================================================================
def compute_sampling_ratio_audit(
    n1: int,
    n2: int,
    t: int,
    l: int,
    landmark_percent: float = None,
) -> Dict[str, Any]:
    """
    Audit sampling ratios from the statistical checklist.

    Internal arguments keep the compact symbols (n1, n2, t, l) so existing
    callers do not break. The returned dict also has English keys:
    minority_class_count, majority_class_count, points_per_snapshot,
    n_snapshots, snapshot_size_percent_of_class, reuse_ratio.
    See docs/Notation.md.
    """
    n = n1 + n2
    ratios = {
        "n": n,
        "n1": n1,
        "n2": n2,
        "t": t,
        "l": l,
        "minority_class_count": n1,
        "majority_class_count": n2,
        "points_per_snapshot": t,
        "n_snapshots": l,
        "snapshot_size_percent_of_class": landmark_percent,
        "reuse_ratio": (t * l) / n1 if n1 else np.nan,
        "landmark_percent": landmark_percent,
        "t_over_n": t / n if n else np.nan,
        "t_over_n1": t / n1 if n1 else np.nan,
        "t_over_n2": t / n2 if n2 else np.nan,
        "max_t_over_class": max(
            (t / n1) if n1 else 0,
            (t / n2) if n2 else 0,
            (t / n) if n else 0,
        ),
        "naive_2tl_over_n": (t * 2 * l) / n if n else np.nan,
        "naive_tl_over_n1": (t * l) / n1 if n1 else np.nan,
        "naive_tl_over_n2": (t * l) / n2 if n2 else np.nan,
        "suggested_t_over_class_lt_0_20": None,
        "suggested_naive_near_or_below_1": None,
    }
    ratios["suggested_t_over_class_lt_0_20"] = ratios["max_t_over_class"] < 0.20
    ratios["suggested_naive_near_or_below_1"] = (
        ratios["naive_tl_over_n1"] <= 1.0 and ratios["naive_tl_over_n2"] <= 1.0
    )
    return ratios


def summarize_snapshot_statistics(
    barcode_csv: str,
    label_col: str = "label",
) -> Dict[str, Any]:
    """
    Record mean and variance of each barcode-statistic column across snapshots.
    Also returns the empirical mean vector (proxy for landscape average \\bar\\lambda
    when working in barcode-statistic feature space rather than full landscapes).
    """
    df = pd.read_csv(os.path.abspath(barcode_csv))
    feature_cols = [c for c in df.columns if c != label_col]
    means = df[feature_cols].mean()
    variances = df[feature_cols].var(ddof=1)
    per_label = {}
    if label_col in df.columns:
        for lab, group in df.groupby(label_col):
            per_label[str(lab)] = {
                "n_snapshots": len(group),
                "mean": group[feature_cols].mean().to_dict(),
                "variance": group[feature_cols].var(ddof=1).to_dict(),
            }
    return {
        "n_snapshots": len(df),
        "feature_columns": feature_cols,
        "global_mean": means.to_dict(),
        "global_variance": variances.to_dict(),
        "lambda_bar_proxy": means.to_dict(),
        "per_label": per_label,
        "source": os.path.abspath(barcode_csv),
    }


def flatten_snapshot_mean_variance(summary: Dict[str, Any], source_key: str) -> List[Dict[str, Any]]:
    """Turn one summarize_snapshot_statistics() result into CSV rows."""
    rows = []
    for feat, mean_v in summary["global_mean"].items():
        rows.append({
            "source": source_key,
            "feature": feat,
            "mean": mean_v,
            "variance": summary["global_variance"][feat],
            "n_snapshots": summary["n_snapshots"],
        })
    return rows


def estimate_intrinsic_dimension_two_nn(
    X: np.ndarray,
    n_samples: int = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Two-NN intrinsic dimension estimator (Facco et al.).
    Uses ratio of distances to 1st and 2nd nearest neighbours.
    """
    rng = check_random_state(random_state)
    X = np.asarray(X, dtype=float)
    if n_samples is not None and n_samples < len(X):
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X = X[idx]

    dists = cdist(X, X)
    np.fill_diagonal(dists, np.inf)
    nn = np.sort(dists, axis=1)[:, :2]
    r1 = nn[:, 0]
    r2 = nn[:, 1]
    valid = (r1 > 0) & np.isfinite(r1) & np.isfinite(r2)
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1]
    if len(mu) < 2:
        return {"intrinsic_dim_two_nn": np.nan, "n_points_used": float(len(X))}

    # MLE: d = 1 / mean(log(mu))
    d_hat = 1.0 / np.mean(np.log(mu))
    return {
        "intrinsic_dim_two_nn": float(d_hat),
        "n_points_used": float(valid.sum()),
        "mean_mu": float(np.mean(mu)),
    }


def estimate_intrinsic_dimension_levina_bickel(
    X: np.ndarray,
    k: int = 10,
    n_samples: int = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Levina–Bickel MLE intrinsic dimension (average over local neighbourhoods).
    """
    rng = check_random_state(random_state)
    X = np.asarray(X, dtype=float)
    if n_samples is not None and n_samples < len(X):
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X = X[idx]

    dists = cdist(X, X)
    np.fill_diagonal(dists, np.inf)
    nn = np.sort(dists, axis=1)[:, :k]
    dims = []
    for row in nn:
        rk = row[-1]
        if rk <= 0 or not np.isfinite(rk):
            continue
        # Avoid zero neighbour distances (duplicates / numerical ties)
        rj = row[:-1]
        if np.any(rj <= 0) or not np.all(np.isfinite(rj)):
            continue
        # d_i = (1/(k-1)) * sum_{j=1}^{k-1} log(r_k / r_j)
        logs = np.log(rk / rj)
        if not np.all(np.isfinite(logs)):
            continue
        dims.append((1.0 / (k - 1)) * np.sum(logs))
    if not dims:
        return {"intrinsic_dim_levina_bickel": np.nan, "k": float(k), "n_points_used": 0.0}
    d_hat = float(np.mean(dims))
    if not np.isfinite(d_hat):
        d_hat = float("nan")
    return {
        "intrinsic_dim_levina_bickel": d_hat,
        "k": float(k),
        "n_points_used": float(len(dims)),
        "std_local_dims": float(np.nanstd(dims)),
    }


def n_components_for_target_variance(
    X: np.ndarray,
    target: float = 0.90,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Smallest PCA rank that keeps at least ``target`` of total variance."""
    X = np.asarray(X, dtype=float)
    max_comp = min(X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=max_comp, random_state=random_state).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cum, target) + 1)
    n = min(max(n, 1), max_comp)
    return {
        "n_components": n,
        "variance_at_n": float(cum[n - 1]),
        "target": float(target),
        "max_components": int(max_comp),
    }


def estimate_intrinsic_dimension_skdim(
    X: np.ndarray,
    n_samples: int = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """scikit-dimension estimators (Bac et al., arXiv:2109.02596).

    TwoNN, MLE (Levina–Bickel), MiND_ML, and lPCA. DANCo is skipped here
    (slow); Experiment 28 already runs it on modest samples.
    """
    rng = check_random_state(random_state)
    X = np.asarray(X, dtype=float)
    if n_samples is not None and n_samples < len(X):
        idx = rng.choice(len(X), size=n_samples, replace=False)
        X = X[idx]
    out: Dict[str, Any] = {
        "package": "scikit-dimension",
        "n_points_used": int(len(X)),
        "estimators": {},
    }
    try:
        import skdim
    except ImportError:
        out["package"] = "skdim_not_installed"
        return out

    def _fit(name, ctor):
        try:
            est = ctor().fit(X)
            dim = getattr(est, "dimension_", None)
            if dim is None:
                out["estimators"][name] = "error:no_dimension_"
                return
            dim = np.asarray(dim, dtype=float).ravel()
            finite = dim[np.isfinite(dim)]
            if finite.size == 0:
                out["estimators"][name] = "error:non_finite_dimension"
                return
            out["estimators"][name] = float(np.mean(finite))
        except Exception as exc:
            out["estimators"][name] = f"error:{type(exc).__name__}:{exc}"

    _fit("TwoNN", skdim.id.TwoNN)
    _fit("MLE_LevinaBickel", lambda: skdim.id.MLE(K=20))
    if hasattr(skdim.id, "MiND_ML"):
        _fit("MiND_ML", skdim.id.MiND_ML)
    _fit("lPCA", skdim.id.lPCA)
    return out


def estimate_intrinsic_dimension_suite(
    X_scaled: np.ndarray,
    pca_components: int,
    n_samples: int = 5000,
    random_state: int = 42,
    variance_target: float = 0.90,
) -> Dict[str, Any]:
    """Intrinsic dimension **before** PCA and **after** the Exp 3 PCA rank.

    Also reports how many components would be needed to keep
    ``variance_target`` (the ~90% design rule for the new tables).
    """
    X_scaled = np.asarray(X_scaled, dtype=float)
    var_info = n_components_for_target_variance(
        X_scaled, target=variance_target, random_state=random_state
    )
    n_comp = min(int(pca_components), X_scaled.shape[1], max(1, X_scaled.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Headline Two-NN / LB can use the full cap. skdim (especially MiND_ML)
    # is O(n²) per estimator — keep a tighter cap so six datasets stay runnable.
    skdim_n = n_samples if n_samples is None else min(int(n_samples), 2000)
    before = {
        "handcoded_two_nn": estimate_intrinsic_dimension_two_nn(
            X_scaled, n_samples=n_samples, random_state=random_state
        ),
        "handcoded_levina_bickel": estimate_intrinsic_dimension_levina_bickel(
            X_scaled, k=10, n_samples=n_samples, random_state=random_state
        ),
        "skdim": estimate_intrinsic_dimension_skdim(
            X_scaled, n_samples=skdim_n, random_state=random_state
        ),
    }
    after = {
        "handcoded_two_nn": estimate_intrinsic_dimension_two_nn(
            X_pca, n_samples=n_samples, random_state=random_state
        ),
        "handcoded_levina_bickel": estimate_intrinsic_dimension_levina_bickel(
            X_pca, k=10, n_samples=n_samples, random_state=random_state
        ),
        "skdim": estimate_intrinsic_dimension_skdim(
            X_pca, n_samples=skdim_n, random_state=random_state
        ),
    }
    return {
        "n_features": int(X_scaled.shape[1]),
        "pca_components_used_in_TDA": int(pca.n_components_),
        "variance_retained_pca": float(pca.explained_variance_ratio_.sum()),
        "n_components_for_target_variance": var_info,
        "before_pca": before,
        "after_pca": after,
    }


def joint_loss_fpq_feature_vectors(
    group1: np.ndarray,
    group2: np.ndarray,
    p: float = 2.0,
    q: float = 2.0,
) -> float:
    """
    F_{p,q} on vector summaries (barcode statistics), using L_p distances.
    Proxy for Robinson & Turner (arXiv:1310.7467) when full diagram distances
    are too costly; document as approximation when publishing.
    """
    def _within(G):
        n = len(G)
        if n < 2:
            return 0.0
        D = cdist(G, G, metric="minkowski", p=p)
        # exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        return float(np.sum(D[mask] ** q) / (2 * n * (n - 1)))

    return _within(group1) + _within(group2)


def permutation_test_algorithm2(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permutations: int = 200,
    p: float = 2.0,
    q: float = 2.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Algorithm 2 (Robinson & Turner, arXiv:1310.7467): permutation p-value for
    whether two samples of persistence summaries arise from the same process.
    Uses F_{p,q} on barcode-statistic vectors (see joint_loss_fpq_feature_vectors).
    """
    rng = check_random_state(random_state)
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    observed = joint_loss_fpq_feature_vectors(g1, g2, p=p, q=q)

    combined = np.vstack([g1, g2])
    z = 1  # count observed as extreme
    null_losses = []
    for _ in range(n_permutations - 1):
        perm = rng.permutation(n1 + n2)
        g1_p = combined[perm[:n1]]
        g2_p = combined[perm[n1:]]
        loss = joint_loss_fpq_feature_vectors(g1_p, g2_p, p=p, q=q)
        null_losses.append(loss)
        if loss <= observed:
            z += 1
    p_value = z / n_permutations
    return {
        "observed_F_pq": float(observed),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "p": p,
        "q": q,
        "n1": n1,
        "n2": n2,
        "null_mean": float(np.mean(null_losses)) if null_losses else np.nan,
        "null_std": float(np.std(null_losses)) if null_losses else np.nan,
        "reference": "Robinson & Turner, arXiv:1310.7467 (Algorithm 2; vector-summary proxy)",
    }


# =============================================================================
# Revised snapshot protocol helpers (Experiment 28+)
# Fixed absolute t, no undersampling, formula vs reuse as separate concerns.
# Full implementation lives in:
#   5_Experiments/Early_Split_TDA_And_No_Undersampling/9_Revised_Snapshot_Protocol/protocol_lib.py
# =============================================================================
def formula_l_from_t_b(t: int, b: float, log_base: str = "e") -> float:
    """Email rule: l ≈ (t / log t)^{2/b}. See Experiment 28 protocol_lib for details."""
    import math

    t = int(t)
    if t < 3:
        raise ValueError("t must be >= 3")
    if b is None or not np.isfinite(b) or b <= 0:
        raise ValueError(f"b must be positive finite, got {b}")
    log_t = math.log(t) if log_base == "e" else math.log10(t)
    return float((t / log_t) ** (2.0 / float(b)))


def reuse_ratio_tl_over_n(t: int, l: int, n_class: int) -> float:
    """R = (t * l) / n_class — expected appearances of a typical point across snapshots."""
    if n_class <= 0:
        return float("nan")
    return float(t * l) / float(n_class)


def select_landmarks_fixed_t(
    data: pd.DataFrame,
    t: int,
    n_files: int,
    dataset_to_use: str,
    save_label_dir: str,
    experiment_name: str,
    add_optional_path: str = None,
    random_state: int = 42,
    verbose: bool = False,
):
    """
    Fixed absolute landmark count t (not a percentage of the class).
    Does not undersample; caller must pass the full class pool.
    """
    rng = check_random_state(random_state)
    if t > len(data):
        raise ValueError(f"Requested t={t} landmarks from only {len(data)} rows")
    dataset_string = get_dataset_folder(dataset_to_use)
    if add_optional_path is None:
        output_dir = str(
            REPO_ROOT / "1_Data" / "Landmark_Sets" / dataset_string / experiment_name / save_label_dir
        )
    else:
        output_dir = str(
            REPO_ROOT / "1_Data" / "Landmark_Sets" / dataset_string / experiment_name / add_optional_path / save_label_dir
        )
    absolute_output_dir = os.path.abspath(output_dir)
    os.makedirs(absolute_output_dir, exist_ok=True)
    for i in range(n_files):
        local = check_random_state(rng.randint(0, 2**31 - 1))
        landmarks = data.sample(n=t, random_state=int(local.randint(0, 2**31 - 1)))
        file_path = os.path.join(absolute_output_dir, f"landmarks_t{t}_{i}.csv")
        landmarks.to_csv(file_path, index=False)
        if verbose:
            print(f"Saved: {file_path}")
    print(f"Saved {n_files} fixed-t landmark files (t={t}) to {absolute_output_dir}")
    return absolute_output_dir


# =============================================================================
# Protocol-aware TDA pipeline (four active arms)
# =============================================================================
PKDD_DUMMY_COLUMNS = (
    "frequency",
    "type",
    "sex",
    "A2",
    "A3",
    "A12",
    "A15",
    "preloan_card_type",
)
PKDD_LOG_COLUMNS = ("amount", "payments", "tx_amount_sum", "tx_amount_mean")
SOUTH_GERMAN_LOG_COLUMNS = ("hoehe", "laufzeit")

DEFAULT_TDA_TUNED_MODEL_CONFIGS = {
    "svm": {
        "model": SVC(),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    },
    "knn": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], "p": [1, 2]},
    },
    "xgb": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    },
    "logistic": {
        "model": LogisticRegression(max_iter=1000),
        "params": {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
    },
    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
        },
    },
}


def dataset_landmark_percentages(dataset_key: str) -> List[float]:
    cfg = get_dataset_config(dataset_key)
    return [float(p) for p in cfg.landmark_percentages]


def dataset_pca_rank(dataset_key: str) -> int:
    cfg = get_dataset_config(dataset_key)
    return int(cfg.notes["pca_n_components_exp3"])


def dataset_n_files(dataset_key: str) -> int:
    cfg = get_dataset_config(dataset_key)
    return int(cfg.notes.get("n_files_per_percentage", 500))


def _processed_table_path(cfg: DatasetConfig) -> Path:
    folder = REPO_ROOT / "1_Data" / "Processed_Datasets" / cfg.folder_name
    xlsx = folder / "processed_data.xlsx"
    csv = folder / "processed_data.csv"
    if xlsx.exists():
        return xlsx
    if csv.exists():
        return csv
    raise FileNotFoundError(
        f"No processed table for {cfg.folder_name} under {folder} "
        "(expected processed_data.xlsx or processed_data.csv)."
    )


def load_processed_features(dataset_key: str) -> Tuple[pd.DataFrame, pd.Series, DatasetConfig]:
    """Load the shared processed table and apply Exp-3 encoding (not PCA)."""
    cfg = get_dataset_config(dataset_key)
    path = _processed_table_path(cfg)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path)

    drop_unnamed = [c for c in data.columns if str(c).startswith("Unnamed")]
    if drop_unnamed:
        data = data.drop(columns=drop_unnamed)

    target = cfg.target_column
    if target not in data.columns:
        raise KeyError(f"{cfg.folder_name} is missing target column '{target}'")

    if cfg.key == "pkdd_czech":
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())
        for col in data.select_dtypes(include=["object"]).columns:
            data[col] = data[col].fillna("missing").astype(str)
        data = data_preprocessing_pipeline(
            data,
            log_col=list(PKDD_LOG_COLUMNS),
            dummy_col=list(PKDD_DUMMY_COLUMNS),
        )
    elif cfg.key == "polish_bankruptcy":
        for col in data.columns:
            if col != target and data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())
        data = data_preprocessing_pipeline(data)
    elif cfg.key == "taiwan_bankruptcy":
        data = data_preprocessing_pipeline(data)
    elif cfg.key == "south_german_credit":
        data = data_preprocessing_pipeline(data, log_col=list(SOUTH_GERMAN_LOG_COLUMNS))
    else:
        for col in data.select_dtypes(include=[np.number]).columns:
            if col != target and data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())
        leftover = [c for c in data.select_dtypes(include=["object"]).columns if c != target]
        if leftover:
            data = pd.get_dummies(data, columns=leftover, drop_first=True, dtype=np.int64)

    y = data[target].astype(int)
    X = data.drop(columns=[target])
    X = X.select_dtypes(include=[np.number]).copy()
    return X, y, cfg


def class_pools_from_features(
    features: pd.DataFrame,
    labels: pd.Series,
    undersample: bool,
    positive_label: int = 1,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Build default / non-default clouds. Snapshot size t is later
    ``floor(n_class * L / 100)`` on whichever pool is returned here."""
    if undersample:
        balanced = balance_binary_by_undersampling(
            features, labels, positive_label=positive_label, random_state=random_state
        )
        pos = balanced[balanced["Class"] == positive_label].drop(columns=["Class"])
        neg = balanced[balanced["Class"] != positive_label].drop(columns=["Class"])
    else:
        frame = features.copy()
        frame["Class"] = labels.values
        pos = frame[frame["Class"] == positive_label].drop(columns=["Class"]).reset_index(drop=True)
        neg = frame[frame["Class"] != positive_label].drop(columns=["Class"]).reset_index(drop=True)
    return {"default": pos, "non-default": neg}


def protocol_tda_matrices_exist(
    protocol_bucket: str,
    dataset_folder: str,
    percentages: List[float],
    split_timing: str,
    experiment_name: str = "1_PH_Default_Parameters",
) -> bool:
    if split_timing == "early":
        needed = [
            tda_artefact_dir(
                "TDA_Datasets", protocol_bucket, experiment_name, dataset_folder, split, f"data_L{int(p) if float(p).is_integer() else p}.csv"
            )
            for split in ("train", "test")
            for p in percentages
        ]
    else:
        needed = [
            tda_artefact_dir(
                "TDA_Datasets",
                protocol_bucket,
                experiment_name,
                dataset_folder,
                f"data_L{int(p) if float(p).is_integer() else p}.csv",
            )
            for p in percentages
        ]
    return all(path.exists() for path in needed)


def late_split_barcode_paths(
    protocol_bucket: str,
    dataset_folder: str,
    percentages: List[float],
    experiment_name: str = "1_PH_Default_Parameters",
) -> List[str]:
    return [
        str(
            tda_artefact_dir(
                "TDA_Datasets",
                protocol_bucket,
                experiment_name,
                dataset_folder,
                f"data_L{_percent_token(p)}.csv",
            )
        )
        for p in percentages
    ]


def early_split_barcode_pairs(
    protocol_bucket: str,
    dataset_folder: str,
    percentages: List[float],
    experiment_name: str = "1_PH_Default_Parameters",
) -> Dict[str, Dict[str, str]]:
    pairs = {}
    for p in percentages:
        token = _percent_token(p)
        pairs[f"data_L{token}"] = {
            "train": str(
                tda_artefact_dir(
                    "TDA_Datasets",
                    protocol_bucket,
                    experiment_name,
                    dataset_folder,
                    "train",
                    f"data_L{token}.csv",
                )
            ),
            "test": str(
                tda_artefact_dir(
                    "TDA_Datasets",
                    protocol_bucket,
                    experiment_name,
                    dataset_folder,
                    "test",
                    f"data_L{token}.csv",
                )
            ),
        }
    return pairs


def _write_h0_slice(src: Path, dest: Path) -> Path:
    df = pd.read_csv(src)
    keep = [c for c in df.columns if c == "label" or c.endswith("_0") or "(Dim 0)" in c]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(dest, index=False)
    return dest


def generate_protocol_barcodes(
    dataset_key: str,
    protocol_bucket: str,
    n_files: Optional[int] = None,
    skip_existing: bool = True,
    homology_dim: int = 2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Build landmarks + Ripser barcodes for one protocol arm's experiment 1.

    Does not train models. Reuses existing ``data_L*.csv`` when ``skip_existing``.
    """
    protocol = get_tda_protocol(protocol_bucket)
    protocol_bucket = protocol["bucket"]
    if protocol.get("homology") == "H0":
        raise ValueError(
            f"{protocol_bucket} uses just H0. Slice barcode tables from "
            f"{protocol['barcode_source_bucket']} instead of running Ripser."
        )
    X, y, cfg = load_processed_features(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    pca_n = dataset_pca_rank(dataset_key)
    n_files = int(n_files or dataset_n_files(dataset_key))
    folder = cfg.folder_name
    exp1 = "1_PH_Default_Parameters"
    label_map = {1: "default", 0: "non-default"}

    meta: Dict[str, Any] = {
        "dataset": folder,
        "protocol_bucket": protocol_bucket,
        "split_timing": protocol["split_timing"],
        "undersample": protocol["undersample"],
        "pca_n_components": pca_n,
        "landmark_percentages": percentages,
        "n_files_per_percentage": n_files,
        "homology_dim": homology_dim,
        "t_rule": "t = floor(n_class * L / 100) on the pool after the protocol split (and after undersample when that knob is on)",
        "skipped_existing": False,
    }

    if skip_existing and protocol_tda_matrices_exist(
        protocol_bucket, folder, percentages, protocol["split_timing"], exp1
    ):
        meta["skipped_existing"] = True
        print(f"[SKIP] {protocol_bucket}/{folder}: data_L*.csv already exist.")
        return meta

    def _emit(pools: Dict[str, pd.DataFrame], optional_path: Optional[str] = None) -> None:
        class_sizes = {name: int(len(frame)) for name, frame in pools.items()}
        print(f"  class pools{'' if optional_path is None else ' [' + optional_path + ']'}: {class_sizes}")
        for name, frame in pools.items():
            for pct in percentages:
                t = max(2, int(len(frame) * pct / 100))
                print(f"    {name} L{pct:g}: n={len(frame)} t={t} l={n_files}")
        generate_landmark_sets(
            class_label_and_data=pools,
            landmark_percentages=percentages,
            dataset_to_use=cfg.key,
            experiment_name=exp1,
            add_optional_path=optional_path,
            n_files_per_percentage=n_files,
            protocol_bucket=protocol_bucket,
        )
        extra = [optional_path] if optional_path else []
        landmark_dir = tda_artefact_dir("Landmark_Sets", protocol_bucket, exp1, folder, *extra)
        barcode_dir = tda_artefact_dir("Barcode_Statistics", protocol_bucket, exp1, folder, *extra)
        tda_dir = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp1, folder, *extra)
        compute_barcodes_from_multiple_landmarks(
            landmark_percentages=percentages,
            landmark_dir=str(landmark_dir),
            barcode_output_dir=str(barcode_dir),
            dim=homology_dim,
            label=label_map,
        )
        build_final_barcode_statistics_data(
            landmark_percentages=percentages,
            barcode_dir=str(barcode_dir),
            output_dir=str(tda_dir),
            label=label_map,
        )

    if protocol["split_timing"] == "late":
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        pca = PCA(n_components=pca_n, random_state=random_state)
        pca_cols = [f"PCA_{i}" for i in range(1, pca_n + 1)]
        X_pca = pd.DataFrame(pca.fit_transform(X_scaled), columns=pca_cols, index=X.index)
        meta["variance_retained"] = float(pca.explained_variance_ratio_.sum())
        print(f"Variance retained (full-table PCA): {meta['variance_retained']:.2%}")
        pools = class_pools_from_features(
            X_pca, y, undersample=protocol["undersample"], random_state=random_state
        )
        meta["class_pool_sizes"] = {k: int(len(v)) for k, v in pools.items()}
        _emit(pools)
    else:
        X_train, X_test, y_train, y_test = stratified_early_split(
            X, y, test_size=0.2, random_state=random_state
        )
        X_train_pca, X_test_pca, _scaler, _pca, var_ratio = fit_scaler_pca_on_train(
            X_train, X_test, n_components=pca_n, random_state=random_state
        )
        meta["variance_retained"] = float(var_ratio)
        meta["n_train"] = int(len(X_train))
        meta["n_test"] = int(len(X_test))
        print(f"Variance retained (train-fit PCA): {var_ratio:.2%}")
        train_pools = class_pools_from_features(
            X_train_pca, y_train, undersample=protocol["undersample"], random_state=random_state
        )
        test_pools = class_pools_from_features(
            X_test_pca, y_test, undersample=protocol["undersample"], random_state=random_state
        )
        meta["train_class_pool_sizes"] = {k: int(len(v)) for k, v in train_pools.items()}
        meta["test_class_pool_sizes"] = {k: int(len(v)) for k, v in test_pools.items()}
        _emit(train_pools, "train")
        _emit(test_pools, "test")

    results_dir = tda_results_dir(protocol_bucket, exp1, folder)
    store_data_as_csv_or_json(
        path=str(results_dir),
        csv=False,
        save_as=["protocol_metadata"],
        data_object=[meta],
    )
    return meta


def train_protocol_default_models(
    dataset_key: str,
    protocol_bucket: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    protocol = get_tda_protocol(protocol_bucket)
    cfg = get_dataset_config(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    folder = cfg.folder_name
    exp1 = "1_PH_Default_Parameters"
    save_path = str(tda_results_dir(protocol_bucket, exp1, folder))
    if protocol["split_timing"] == "early":
        pairs = early_split_barcode_pairs(protocol_bucket, folder, percentages, exp1)
        results = train_multiple_dataset_tda_presplit(
            train_test_pairs=pairs,
            y_col_name="label",
            random_state=random_state,
            xgb={"eval_metric": "logloss"},
        )
    else:
        paths = late_split_barcode_paths(protocol_bucket, folder, percentages, exp1)
        results = train_multiple_dataset_tda(
            path_datasets=paths,
            y_col_name="label",
            test_size=0.2,
            random_state=random_state,
            xgb={"eval_metric": "logloss"},
        )
    store_results(path=save_path, save_name="model_results", result_object=results)
    return results


def train_protocol_tuned_models(
    dataset_key: str,
    protocol_bucket: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Consumer: GridSearchCV on experiment-1 barcodes. No Ripser."""
    protocol = get_tda_protocol(protocol_bucket)
    cfg = get_dataset_config(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    folder = cfg.folder_name
    exp1 = "1_PH_Default_Parameters"
    exp2 = "2_PH_Tuned_Parameters"
    save_path = str(tda_results_dir(protocol_bucket, exp2, folder))
    if protocol["split_timing"] == "early":
        pairs = early_split_barcode_pairs(protocol_bucket, folder, percentages, exp1)
        results = train_models_on_multiple_presplit_datasets(
            train_test_pairs=pairs,
            model_configs=DEFAULT_TDA_TUNED_MODEL_CONFIGS,
            target_column="label",
            scoring_metric="f1",
            scale_features=True,
            random_state=random_state,
            n_splits_kfold=5,
        )
    else:
        paths = late_split_barcode_paths(protocol_bucket, folder, percentages, exp1)
        results = train_models_on_multiple_datasets(
            data_paths=paths,
            model_configs=DEFAULT_TDA_TUNED_MODEL_CONFIGS,
            target_column="label",
            test_size=0.2,
            scoring_metric="f1",
            scale_features=True,
            random_state=random_state,
            n_splits_kfold=5,
        )
    store_results(path=save_path, save_name="model_results", result_object=results)
    return results


def train_protocol_h0_only_models(
    dataset_key: str,
    protocol_bucket: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Consumer: keep H0 columns from the matching H0-and-H1 experiment-1 matrices. No Ripser."""
    protocol = get_tda_protocol(protocol_bucket)
    cfg = get_dataset_config(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    folder = cfg.folder_name
    exp1 = "1_PH_Default_Parameters"
    source_bucket = barcode_source_bucket(protocol_bucket)
    dest_experiment = exp1 if protocol.get("homology") == "H0" else "3_H0_Only"
    dest_root = tda_artefact_dir("TDA_Datasets", protocol_bucket, dest_experiment, folder)
    if protocol["split_timing"] == "early":
        pairs = {}
        for p in percentages:
            token = _percent_token(p)
            src_train = tda_artefact_dir("TDA_Datasets", source_bucket, exp1, folder, "train", f"data_L{token}.csv")
            src_test = tda_artefact_dir("TDA_Datasets", source_bucket, exp1, folder, "test", f"data_L{token}.csv")
            dest_train = dest_root / "train" / f"data_L{token}.csv"
            dest_test = dest_root / "test" / f"data_L{token}.csv"
            _write_h0_slice(src_train, dest_train)
            _write_h0_slice(src_test, dest_test)
            pairs[f"data_L{token}"] = {"train": str(dest_train), "test": str(dest_test)}
        results = train_multiple_dataset_tda_presplit(
            train_test_pairs=pairs,
            y_col_name="label",
            random_state=random_state,
            xgb={"eval_metric": "logloss"},
        )
    else:
        paths = []
        for p in percentages:
            token = _percent_token(p)
            src = tda_artefact_dir("TDA_Datasets", source_bucket, exp1, folder, f"data_L{token}.csv")
            dest = dest_root / f"data_L{token}.csv"
            _write_h0_slice(src, dest)
            paths.append(str(dest))
        results = train_multiple_dataset_tda(
            path_datasets=paths,
            y_col_name="label",
            test_size=0.2,
            random_state=random_state,
            xgb={"eval_metric": "logloss"},
        )
    store_results(
        path=str(tda_results_dir(protocol_bucket, dest_experiment, folder)),
        save_name="model_results",
        result_object=results,
    )
    return results


def train_protocol_drop_correlated(
    dataset_key: str,
    protocol_bucket: str,
    threshold: float = 0.80,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Consumer: drop correlated barcode columns from experiment-1 matrices."""
    protocol = get_tda_protocol(protocol_bucket)
    cfg = get_dataset_config(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    folder = cfg.folder_name
    exp1 = "1_PH_Default_Parameters"
    exp4 = "4_Dropping_Correlated_Barcode_Statistics_Columns"
    save_path = str(tda_results_dir(protocol_bucket, exp4, folder))
    var_dir = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp4, folder, "Using_High_Variance_For_Correlation")
    target_dir = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp4, folder, "Using_Target_Variable_For_Correlation")
    var_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    data_objects = {}
    dropped_payload = []
    dropped_names = []
    if protocol["split_timing"] == "early":
        pairs = early_split_barcode_pairs(protocol_bucket, folder, percentages, exp1)
        for name, paths in pairs.items():
            train_df = rename_barcode_statistics_columns(pd.read_csv(paths["train"]))
            test_df = rename_barcode_statistics_columns(pd.read_csv(paths["test"]))
            X_train = train_df.drop(columns=["label"])
            y_train = train_df["label"]
            X_test = test_df.drop(columns=["label"])
            y_test = test_df["label"]
            kept_target, dropped_target = drop_correlated_features(
                X_train,
                threshold=threshold,
                feature_label=True,
                strategy="target_corr",
                target=y_train,
            )
            keep_cols = [c for c in kept_target.columns if c != "label"]
            data_objects[name] = {
                "data": kept_target,
                "X_test": X_test[keep_cols],
                "y_test": y_test,
            }
            dropped_payload.append(dropped_target)
            dropped_names.append(f"{name}_target_drop")
            kept_target.to_csv(target_dir / f"{name}_target.csv", index=False)
    else:
        for p in percentages:
            token = _percent_token(p)
            src = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp1, folder, f"data_L{token}.csv")
            df = rename_barcode_statistics_columns(pd.read_csv(src)).sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
            X = df.drop(columns=["label"])
            y = df["label"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )
            kept_var, dropped_var = drop_correlated_features(
                X_train,
                threshold=threshold,
                feature_label=True,
                strategy="high_variance",
                target=y_train,
            )
            kept_target, dropped_target = drop_correlated_features(
                X_train,
                threshold=threshold,
                feature_label=True,
                strategy="target_corr",
                target=y_train,
            )
            keep_cols = [c for c in kept_target.columns if c != "label"]
            name = f"data_L{token}"
            data_objects[name] = {
                "data": kept_target,
                "X_test": X_test[keep_cols],
                "y_test": y_test,
            }
            var_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            kept_var.to_csv(var_dir / f"{name}_var.csv", index=False)
            kept_target.to_csv(target_dir / f"{name}_target.csv", index=False)
            dropped_payload.extend([dropped_var, dropped_target])
            dropped_names.extend([f"{name}_var_drop", f"{name}_target_drop"])

    store_data_as_csv_or_json(
        path=save_path,
        csv=False,
        save_as=dropped_names,
        data_object=dropped_payload,
    )
    results = train_multiple_dataset_tda_drop_correlated(
        data_objects=data_objects,
        test_size=0.2,
        random_state=random_state,
        xgb={"eval_metric": "logloss"},
    )
    store_results(path=save_path, save_name="model_results", result_object=results)
    return results


def train_protocol_linear_regression(
    dataset_key: str,
    protocol_bucket: str,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Consumer: linear regression on the H0 slice of experiment-1 barcodes."""
    protocol = get_tda_protocol(protocol_bucket)
    cfg = get_dataset_config(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    folder = cfg.folder_name
    exp1 = "1_PH_Default_Parameters"
    exp5 = "5_Linear_Regression_For_Prediction"
    dest_root = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp5, folder)
    if protocol["split_timing"] == "early":
        # Concatenate train/test H0 slices so the existing late-split linear
        # trainer can 80/20 the barcode rows of this arm's already-split matrices
        # would leak the customer split. Train on train H0, evaluate on test H0
        # by writing a combined file only for bookkeeping and scoring manually.
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        results = {}
        for p in percentages:
            token = _percent_token(p)
            train_src = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp1, folder, "train", f"data_L{token}.csv")
            test_src = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp1, folder, "test", f"data_L{token}.csv")
            dest_train = dest_root / "train" / f"data_L{token}.csv"
            dest_test = dest_root / "test" / f"data_L{token}.csv"
            _write_h0_slice(train_src, dest_train)
            _write_h0_slice(test_src, dest_test)
            train_df = pd.read_csv(dest_train)
            test_df = pd.read_csv(dest_test)
            feature_cols = [c for c in train_df.columns if c != "label"]
            model = LinearRegression()
            model.fit(train_df[feature_cols], train_df["label"])
            scores = model.predict(test_df[feature_cols])
            y_pred = (scores >= 0.5).astype(int)
            y_test = test_df["label"].astype(int)
            results[f"data_L{token}"] = {
                "linear_regression": {
                    "model": model,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, zero_division=0),
                    "classification_report": classification_report(y_test, y_pred, zero_division=0),
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                }
            }
            print(f"[OK] Linear regression (presplit) data_L{token}")
    else:
        paths = []
        for p in percentages:
            token = _percent_token(p)
            src = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp1, folder, f"data_L{token}.csv")
            dest = dest_root / f"data_L{token}.csv"
            _write_h0_slice(src, dest)
            paths.append(str(dest))
        results = train_multiple_dataset_tda_linear_regression(
            path_datasets=paths,
            y_col_name="label",
            test_size=0.2,
            random_state=random_state,
        )
    store_results(
        path=str(tda_results_dir(protocol_bucket, exp5, folder)),
        save_name="model_results",
        result_object=results,
    )
    return results


def run_protocol_sampling_ratio_audit(
    dataset_key: str,
    protocol_bucket: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Audit t/l reuse from class pools after the protocol's split step. No Ripser."""
    import math

    protocol = get_tda_protocol(protocol_bucket)
    X, y, cfg = load_processed_features(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    n_files = dataset_n_files(dataset_key)
    folder = cfg.folder_name
    exp6 = "6_Sampling_Ratio_Audit"
    save_path = tda_results_dir(protocol_bucket, exp6, folder)

    def _pool_counts(labels: pd.Series, undersample: bool) -> Tuple[int, int, int, int]:
        n_pos = int((labels == cfg.positive_label).sum())
        n_neg = int((labels != cfg.positive_label).sum())
        if undersample:
            n1 = n2 = min(n_pos, n_neg)
        else:
            n1, n2 = n_pos, n_neg
        return n_pos, n_neg, n1, n2

    rows = []
    payload: Dict[str, Any] = {
        "dataset": folder,
        "protocol_bucket": protocol_bucket,
        "split_timing": protocol["split_timing"],
        "undersample": protocol["undersample"],
        "t_rule": "t = floor(n_class * L / 100) on the available pool after the protocol split",
        "l": n_files,
        "landmarks": {},
    }

    if protocol["split_timing"] == "early":
        _X_train, _X_test, y_train, y_test = stratified_early_split(
            X, y, test_size=0.2, random_state=random_state
        )
        splits = {"train": y_train, "test": y_test}
    else:
        splits = {"full": y}

    for split_name, labels in splits.items():
        raw_pos, raw_neg, n1, n2 = _pool_counts(labels, protocol["undersample"])
        payload[f"{split_name}_raw_n_pos"] = raw_pos
        payload[f"{split_name}_raw_n_neg"] = raw_neg
        payload[f"{split_name}_n1"] = n1
        payload[f"{split_name}_n2"] = n2
        for pct in percentages:
            t1 = max(2, int(n1 * pct / 100))
            t2 = max(2, int(n2 * pct / 100))
            # Audit uses the class-specific snapshot size. When balanced,
            # t1 == t2. When not, report both and score reuse per class.
            for class_name, n_class, t in (("class1", n1, t1), ("class2", n2, t2)):
                revised_l = max(2, int(math.ceil(n_class / t))) if t else 1
                for rule, l_value in (("historical_l500", n_files), ("revised_ceil_n_over_t", revised_l)):
                    audit = compute_sampling_ratio_audit(
                        n1=n1, n2=n2, t=t, l=l_value, landmark_percent=pct
                    )
                    audit.update(
                        {
                            "dataset": folder,
                            "protocol_bucket": protocol_bucket,
                            "split": split_name,
                            "class": class_name,
                            "n_class": n_class,
                            "t_class": t,
                            "l_rule": rule,
                            "undersample": protocol["undersample"],
                        }
                    )
                    rows.append(audit)
                    payload["landmarks"][f"{split_name}_L{pct:g}_{class_name}_{rule}"] = audit

    save_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(save_path / "sampling_ratio_audit.csv", index=False)
    store_results(path=str(save_path), save_name="sampling_ratio_audit", result_object=payload)
    print(f"Saved {save_path / 'sampling_ratio_audit.csv'}")
    return frame


def run_protocol_snapshot_mean_variance(
    dataset_key: str,
    protocol_bucket: str,
) -> Dict[str, Any]:
    """Consumer: mean/variance of experiment-1 barcode columns."""
    protocol = get_tda_protocol(protocol_bucket)
    cfg = get_dataset_config(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    folder = cfg.folder_name
    exp1 = "1_PH_Default_Parameters"
    exp7 = "7_Snapshot_Mean_Variance"
    save_path = tda_results_dir(protocol_bucket, exp7, folder)
    sources: List[Path] = []
    if protocol["split_timing"] == "early":
        for split in ("train", "test"):
            for p in percentages:
                sources.append(
                    tda_artefact_dir(
                        "TDA_Datasets",
                        protocol_bucket,
                        exp1,
                        folder,
                        split,
                        f"data_L{_percent_token(p)}.csv",
                    )
                )
    else:
        for p in percentages:
            sources.append(
                tda_artefact_dir(
                    "TDA_Datasets",
                    protocol_bucket,
                    exp1,
                    folder,
                    f"data_L{_percent_token(p)}.csv",
                )
            )

    all_summaries = {}
    flat_rows = []
    missing = []
    for path in sources:
        if not path.exists():
            missing.append(str(path))
            print(f"Missing (run this arm's experiment 1 first): {path}")
            continue
        summary = summarize_snapshot_statistics(str(path))
        key = f"{protocol_bucket}/{exp1}/{folder}/{path.name}"
        if "train" in path.parts or "test" in path.parts:
            key = f"{protocol_bucket}/{exp1}/{folder}/{path.parent.name}/{path.name}"
        all_summaries[key] = summary
        for feat, mean_v in summary["global_mean"].items():
            flat_rows.append(
                {
                    "source": key,
                    "feature": feat,
                    "mean": mean_v,
                    "variance": summary["global_variance"][feat],
                    "n_snapshots": summary["n_snapshots"],
                }
            )
        print(f"OK {path.name}: n={summary['n_snapshots']}")

    if not flat_rows:
        raise FileNotFoundError(
            f"No experiment-1 barcode files for {protocol_bucket}/{folder}. Missing: {missing}"
        )
    save_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat_rows).to_csv(save_path / "snapshot_mean_variance.csv", index=False)
    store_results(
        path=str(save_path),
        save_name="snapshot_mean_variance_full",
        result_object=all_summaries,
    )
    return all_summaries


def run_protocol_algorithm2(
    dataset_key: str,
    protocol_bucket: str,
    max_per_group: int = 100,
    n_perm: int = 200,
) -> pd.DataFrame:
    """Consumer: Robinson–Turner Algorithm 2 on experiment-1 barcode matrices."""
    protocol = get_tda_protocol(protocol_bucket)
    cfg = get_dataset_config(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    folder = cfg.folder_name
    exp1 = "1_PH_Default_Parameters"
    exp8 = "8_Null_Hypothesis_Algorithm2"
    save_path = tda_results_dir(protocol_bucket, exp8, folder)
    sources: List[Path] = []
    if protocol["split_timing"] == "early":
        for split in ("train", "test"):
            for p in percentages:
                sources.append(
                    tda_artefact_dir(
                        "TDA_Datasets",
                        protocol_bucket,
                        exp1,
                        folder,
                        split,
                        f"data_L{_percent_token(p)}.csv",
                    )
                )
    else:
        for p in percentages:
            sources.append(
                tda_artefact_dir(
                    "TDA_Datasets",
                    protocol_bucket,
                    exp1,
                    folder,
                    f"data_L{_percent_token(p)}.csv",
                )
            )

    rows = []
    payload = {}
    rng = np.random.default_rng(42)
    for path in sources:
        if not path.exists():
            print(f"Missing (run this arm's experiment 1 first): {path}")
            continue
        df = pd.read_csv(path)
        feats = [c for c in df.columns if c != "label"]
        g1 = df[df["label"] == cfg.positive_label][feats].to_numpy()
        g2 = df[df["label"] != cfg.positive_label][feats].to_numpy()
        if len(g1) > max_per_group:
            g1 = g1[rng.choice(len(g1), max_per_group, replace=False)]
        if len(g2) > max_per_group:
            g2 = g2[rng.choice(len(g2), max_per_group, replace=False)]
        rel = f"{path.parent.name}/{path.name}" if path.parent.name in {"train", "test"} else path.name
        key = f"{folder}/{rel}"
        payload[key] = {"barcode_vector_proxy": True, "tests": {}}
        for p, q in ((2, 2), (1, 1), (2, 1)):
            result = permutation_test_algorithm2(
                g1, g2, n_permutations=n_perm, p=p, q=q, random_state=42
            )
            payload[key]["tests"][f"F_{p}_{q}"] = result
            rows.append(
                {
                    "source": key,
                    "protocol_bucket": protocol_bucket,
                    "p": p,
                    "q": q,
                    "observed_F_pq": result["observed_F_pq"],
                    "p_value": result["p_value"],
                    "n1": result["n1"],
                    "n2": result["n2"],
                    "null_mean": result["null_mean"],
                    "barcode_vector_proxy": True,
                }
            )
            print(f"{rel} F_{p},{q}: observed={result['observed_F_pq']:.4f}, p={result['p_value']:.4f}")

    if not rows:
        raise FileNotFoundError(
            f"No experiment-1 barcode files for Algorithm 2 on {protocol_bucket}/{folder}."
        )
    save_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(save_path / "algorithm2_permutation_results.csv", index=False)
    store_results(
        path=str(save_path),
        save_name="algorithm2_permutation_results",
        result_object=payload,
    )
    return frame


def run_protocol_experiment(
    dataset_key: str,
    protocol_bucket: str,
    experiment: str,
    skip_existing_barcodes: bool = True,
    n_files: Optional[int] = None,
) -> Any:
    """Dispatch one numbered experiment inside a protocol bucket."""
    if experiment == "1_PH_Default_Parameters":
        protocol = get_tda_protocol(protocol_bucket)
        if protocol.get("homology") == "H0":
            return train_protocol_h0_only_models(dataset_key, protocol_bucket)
        meta = generate_protocol_barcodes(
            dataset_key,
            protocol_bucket,
            n_files=n_files,
            skip_existing=skip_existing_barcodes,
        )
        models = train_protocol_default_models(dataset_key, protocol_bucket)
        return {"metadata": meta, "model_results": models}
    if experiment == "2_PH_Tuned_Parameters":
        return train_protocol_tuned_models(dataset_key, protocol_bucket)
    if experiment == "3_H0_Only":
        return train_protocol_h0_only_models(dataset_key, protocol_bucket)
    if experiment == "4_Dropping_Correlated_Barcode_Statistics_Columns":
        return train_protocol_drop_correlated(dataset_key, protocol_bucket)
    if experiment == "5_Linear_Regression_For_Prediction":
        return train_protocol_linear_regression(dataset_key, protocol_bucket)
    if experiment == "6_Sampling_Ratio_Audit":
        return run_protocol_sampling_ratio_audit(dataset_key, protocol_bucket)
    if experiment == "7_Snapshot_Mean_Variance":
        return run_protocol_snapshot_mean_variance(dataset_key, protocol_bucket)
    if experiment == "8_Null_Hypothesis_Algorithm2":
        return run_protocol_algorithm2(dataset_key, protocol_bucket)
    raise ValueError(f"Unknown active TDA experiment: {experiment}")


# =============================================================================
# Experiment visualization (6_Results/{Bucket}/{Experiment}/)
# =============================================================================

_MODEL_METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_score",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
}
_CANONICAL_DATASET_FOLDERS = (
    "Default_Of_Credit_Card_Client_Data",
    "Statlog_German_Credit_Data",
    "PKDD_Czech_Financial",
    "Polish_Bankruptcy_3Year",
    "Taiwan_Bankruptcy",
    "South_German_Credit",
)
_METRIC_CSV_CANDIDATES = (
    "model_results.pkl",
    "CV_results.pkl",
    "baseline_results.csv",
    "extended_results.csv",
    "tda_results.csv",
    "metrics_table.csv",
)
_SAMPLING_ARTEFACTS = ("sampling_ratio_audit.csv", "sampling_ratio_audit.pkl")
_SNAPSHOT_ARTEFACTS = ("snapshot_mean_variance.csv", "snapshot_mean_variance_full.pkl")
_ALG2_ARTEFACTS = ("algorithm2_permutation_results.csv", "algorithm2_permutation_results.pkl")
_EXP9_ARTEFACTS = ("ml_results.csv", "all_ml_results.csv", "design.json")
_ID_ARTEFACTS = ("intrinsic_dimension_estimates.csv", "intrinsic_dimension_estimates.pkl")


class ResultsNotGeneratedError(FileNotFoundError):
    """Raised when a visualizer cannot find the experiment's result artefacts."""


def _results_not_generated(expected_paths: List[Union[str, Path]]) -> None:
    rendered = "\n".join(f"  - {Path(p)}" for p in expected_paths)
    raise ResultsNotGeneratedError(
        "results not generated yet. Expected artefacts at one of:\n" + rendered
    )


def experiment_results_root(protocol_bucket: str, experiment: str) -> Path:
    return win_long_path(REPO_ROOT / "6_Results" / resolve_protocol_bucket(protocol_bucket) / experiment)


def experiment_visualizations_dir(protocol_bucket: str, experiment: str) -> Path:
    """Canonical plot destination: 6_Results/{Bucket}/{Experiment}/Visualizations/."""
    return experiment_results_root(protocol_bucket, experiment) / VISUALIZATIONS_DIRNAME


_VIZ_OUTPUT_DIR_NAMES = {
    VISUALIZATIONS_DIRNAME,
    "model_viz",
    "cv_viz",
    "cross_dataset_viz",
    "plots",
}


def _registered_dataset_folders() -> List[str]:
    folders = [cfg.folder_name for cfg in DATASET_REGISTRY.values()]
    return folders or list(_CANONICAL_DATASET_FOLDERS)


def _dataset_result_dirs(results_root: Path) -> List[Path]:
    found = []
    for folder in _registered_dataset_folders():
        path = results_root / folder
        if path.is_dir():
            found.append(path)
    if found:
        return found
    if results_root.is_dir():
        return sorted(
            p for p in results_root.iterdir()
            if p.is_dir() and not p.name.startswith(("_", "."))
            and p.name not in _VIZ_OUTPUT_DIR_NAMES
        )
    return []


def _is_metric_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(_MODEL_METRIC_KEYS & set(obj.keys()))


def _normalize_model_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(stats)
    if "f1" in out and "f1_score" not in out:
        out["f1_score"] = out["f1"]
    return out


def normalize_model_results(obj: Any, fallback_label: str) -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """Coerce pickle/CSV model blobs into {group: {model: metrics}}."""
    if not isinstance(obj, dict) or not obj:
        return None
    sample = next(iter(obj.values()))
    if isinstance(sample, dict) and sample:
        inner_sample = next(iter(sample.values()))
        if _is_metric_dict(inner_sample):
            normalized: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for group, models in obj.items():
                if not isinstance(models, dict):
                    continue
                normalized[str(group)] = {
                    str(model): _normalize_model_stats(stats)
                    for model, stats in models.items()
                    if isinstance(stats, dict)
                }
            return normalized or None
    if any(_is_metric_dict(v) for v in obj.values()):
        return {
            fallback_label: {
                str(model): _normalize_model_stats(stats)
                for model, stats in obj.items()
                if isinstance(stats, dict) and _is_metric_dict(stats)
            }
        }
    flattened: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, value in obj.items():
        nested = normalize_model_results(value, str(key)) if isinstance(value, dict) else None
        if not nested:
            continue
        for group, models in nested.items():
            label = f"{key}/{group}" if group != str(key) else str(key)
            flattened[label] = models
    return flattened or None


def _model_results_from_metrics_frame(
    frame: pd.DataFrame,
    fallback_label: str,
) -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    if frame.empty or "model" not in frame.columns:
        return None
    metric_cols = {
        col: ("f1_score" if col == "f1" else col)
        for col in frame.columns
        if col in _MODEL_METRIC_KEYS
    }
    if not metric_cols:
        return None
    group_col = next(
        (
            col for col in (
                "landmark_percent", "setting", "protocol", "variant",
                "feature_space", "run_key", "t",
            )
            if col in frame.columns and frame[col].nunique(dropna=True) > 1
        ),
        None,
    )
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for _, row in frame.iterrows():
        group = fallback_label
        if group_col is not None and pd.notna(row[group_col]):
            group = f"{fallback_label} | {group_col}={row[group_col]}"
        model = str(row["model"])
        stats = {
            metric_cols[col]: float(row[col])
            for col in metric_cols
            if pd.notna(row[col])
        }
        if stats:
            grouped.setdefault(group, {})[model] = _normalize_model_stats(stats)
    return grouped or None


def _openable(path: Union[str, Path]) -> Path:
    return win_long_path(path)


def _save_current_figure(path: Path, tight: bool = True) -> Path:
    """Compatibility wrapper; new writers should call `_save_figure`."""
    fig = plt.gcf()
    del tight
    return _save_figure(fig, path)


def _bar_from_frame(
    frame: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str],
    title: str,
    ylabel: str,
    save_path: Path,
    hline: Optional[float] = None,
    rotate_xticks: int = 35,
    hline_label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    yscale: Optional[str] = None,
    note: str = "",
) -> Path:
    """Publication-quality bar chart; facets automatically when many datasets."""
    del rotate_xticks
    data = frame.copy()
    n_x = int(data[x].nunique(dropna=True)) if x in data.columns else 0
    n_hue = int(data[hue].nunique(dropna=True)) if hue and hue in data.columns else 0
    looks_like_dataset = x in {"dataset", "dataset_label", "dataset_folder"}
    if looks_like_dataset and n_x >= 3 and n_hue >= 1:
        facet_frame = data.copy()
        if "dataset_label" not in facet_frame.columns:
            facet_frame["dataset_label"] = facet_frame[x].map(lambda v: pretty_dataset_label(v, short=True))
        return plot_faceted_bars(
            facet_frame,
            x=hue,
            y=y,
            facet="dataset_label",
            title=title,
            ylabel=ylabel,
            save_path=save_path,
            hline=hline,
            hline_label=hline_label,
            ylim=ylim,
            yscale=yscale,
            note=note,
        )
    return plot_grouped_bars(
        data,
        x=x,
        y=y,
        hue=hue,
        title=title,
        ylabel=ylabel,
        save_path=save_path,
        hline=hline,
        hline_label=hline_label,
        ylim=ylim,
        yscale=yscale,
        note=note,
    )


def _attach_dataset_labels(frame: pd.DataFrame, source_col: Optional[str] = None, dataset_col: str = "dataset") -> pd.DataFrame:
    out = frame.copy()
    folders = []
    for _, row in out.iterrows():
        folder = ""
        if source_col and source_col in out.columns:
            folder = ""
            for part in re.split(r"[\\/:]+", str(row[source_col])):
                resolved = _registry_folder_from_token(part)
                if resolved:
                    folder = resolved
                    break
        if not folder and dataset_col in out.columns:
            folder = _registry_folder_from_token(row[dataset_col]) or str(row[dataset_col])
        folders.append(folder)
    out["dataset_folder"] = folders
    out["dataset_label"] = [pretty_dataset_label(f, short=True) if f else "Unknown" for f in folders]
    return out


def _points_per_snapshot_label(value: Any) -> str:
    num = float(value)
    shown = int(num) if num.is_integer() else num
    return f"{shown:g} points per snapshot"


def _points_count_tick(value: Any) -> str:
    num = float(value)
    shown = int(num) if num.is_integer() else num
    return f"{shown:g}"


def _pretty_l_rule(value: Any) -> str:
    text = str(value)
    mapping = {
        "historical_l500": "Historical (500 snapshots)",
        "revised_ceil_n_over_t": "Revised snapshot count",
    }
    for key, label in mapping.items():
        if key in text:
            return label
    return text.replace("_", " ")


def visualize_model_pickle_experiment(
    protocol_bucket: str,
    experiment: str,
    hide_axis_labels: bool = False,
) -> List[Path]:
    """Plot model_results.pkl / metric CSVs for every dataset folder that has them."""
    results_root = experiment_results_root(protocol_bucket, experiment)
    expected = [
        results_root / folder / name
        for folder in _registered_dataset_folders()
        for name in _METRIC_CSV_CANDIDATES
    ]
    written: List[Path] = []
    missing: List[Path] = []
    cross_groups: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for dataset_dir in _dataset_result_dirs(results_root):
        pickle_path = dataset_dir / "model_results.pkl"
        csv_candidates = [
            dataset_dir / name
            for name in ("baseline_results.csv", "extended_results.csv", "tda_results.csv", "metrics_table.csv")
            if (dataset_dir / name).is_file()
        ]
        model_results = None
        if pickle_path.is_file():
            model_results = normalize_model_results(joblib.load(_openable(pickle_path)), dataset_dir.name)
        elif csv_candidates:
            model_results = _model_results_from_metrics_frame(
                pd.read_csv(csv_candidates[0]), dataset_dir.name
            )
        else:
            missing.append(pickle_path)
            continue
        if not model_results:
            missing.append(pickle_path)
            continue

        viz_dir = experiment_visualizations_dir(protocol_bucket, experiment)
        plot_path = improved_visualize_model_results(
            model_results=model_results,
            save_dir=str(viz_dir),
            compare_datasets=False,
            export_metrics=True,
            plot_precision_recall=False,
            colormap="viridis",
            hide_axis_labels=hide_axis_labels,
            filename_prefix=dataset_dir.name,
            protocol_bucket=protocol_bucket,
        )
        if isinstance(plot_path, list):
            written.extend(Path(p) for p in plot_path)
        elif plot_path:
            written.append(Path(plot_path))

        cv_path = dataset_dir / "CV_results.pkl"
        if cv_path.is_file():
            cv_results = joblib.load(_openable(cv_path))
            if isinstance(cv_results, dict) and cv_results and not isinstance(next(iter(cv_results.values())), dict):
                cv_results = {dataset_dir.name: cv_results}
            elif isinstance(cv_results, dict) and cv_results:
                sample = next(iter(cv_results.values()))
                if isinstance(sample, dict) and sample and not isinstance(next(iter(sample.values()), None), dict):
                    cv_results = {dataset_dir.name: cv_results}
            cv_out = visualize_cross_validation_detailed(
                cross_val_results=cv_results,
                save_dir=str(viz_dir),
                colormap="viridis",
                compare_models=True,
                protocol_bucket=protocol_bucket,
            )
            written.extend(Path(p) for p in (cv_out or []))

        for group, models in model_results.items():
            group_key = str(group)
            if dataset_dir.name in group_key:
                cross_groups[group_key] = models
            else:
                cross_groups[f"{dataset_dir.name}:{group_key}"] = models

    if not written and not cross_groups:
        _results_not_generated(expected)

    if len(cross_groups) >= 2:
        viz_dir = experiment_visualizations_dir(protocol_bucket, experiment)
        plot_path = improved_visualize_model_results(
            model_results=cross_groups,
            save_dir=str(viz_dir),
            compare_datasets=True,
            export_metrics=True,
            plot_precision_recall=False,
            colormap="viridis",
            hide_axis_labels=False,
            filename_prefix="cross",
            protocol_bucket=protocol_bucket,
        )
        if isinstance(plot_path, list):
            written.extend(Path(p) for p in plot_path)
        elif plot_path:
            written.append(Path(plot_path))

    for path in missing:
        print(f"[skip] results not generated yet: {path}")
    return written


def visualize_sampling_ratio_audit_experiment(protocol_bucket: str, experiment: str) -> List[Path]:
    results_root = experiment_results_root(protocol_bucket, experiment)
    viz_dir = experiment_visualizations_dir(protocol_bucket, experiment)
    per_dataset = [results_root / folder / "sampling_ratio_audit.csv" for folder in _registered_dataset_folders()]
    expected = per_dataset + [results_root / "sampling_ratio_audit.csv"]
    frames = []
    for csv_path in per_dataset:
        if csv_path.is_file():
            frame = pd.read_csv(csv_path)
            if "dataset" not in frame.columns:
                frame["dataset"] = csv_path.parent.name
            frames.append(frame)
    if not frames and (results_root / "sampling_ratio_audit.csv").is_file():
        frames.append(pd.read_csv(results_root / "sampling_ratio_audit.csv"))
    if not frames:
        _results_not_generated(expected)
    data = _attach_dataset_labels(pd.concat(frames, ignore_index=True))
    viz_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    reuse_col = next(
        (c for c in ("naive_tl_over_n1", "max_t_over_class", "naive_2tl_over_n") if c in data.columns),
        None,
    )
    if reuse_col is not None:
        plot_df = data.copy()
        if "l_rule" in plot_df.columns:
            plot_df = plot_df[plot_df["l_rule"].astype(str).str.contains("historical|revised", case=False, na=False)]
            plot_df["l_rule_label"] = plot_df["l_rule"].map(_pretty_l_rule)
        if "class" in plot_df.columns:
            plot_df = plot_df.drop_duplicates(
                subset=[c for c in ("dataset_folder", "landmark_percent", "l_rule", "split", "class") if c in plot_df.columns]
            )
        if "landmark_percent" in plot_df.columns:
            plot_df["landmark_label"] = plot_df["landmark_percent"].map(
                lambda v: f"{v:g}% of class"
            )
        hue = "landmark_label" if "landmark_label" in plot_df.columns else None
        x_col = "l_rule_label" if "l_rule_label" in plot_df.columns else "dataset_label"
        ctx = protocol_context_sentence(protocol_bucket)
        written.append(plot_faceted_bars(
            plot_df,
            x=x_col,
            y=reuse_col,
            facet="dataset_label",
            hue=hue if x_col != "dataset_label" else ("l_rule_label" if "l_rule_label" in plot_df.columns else hue),
            title=process_figure_title(protocol_bucket, "Expected sampling reuse by snapshot-count rule"),
            ylabel="Expected reuse relative to minority class count",
            save_path=viz_dir / "sampling_reuse_by_rule_faceted.png",
            hline=1.0,
            hline_label="Each minority-class row used about once",
            yscale="log",
            wrap_width=22,
            note=(
                "Each bar is expected sampling reuse: (points per snapshot times number of snapshots) "
                "divided by the minority class count. A value of 1 means each minority-class customer is used about once. "
                "The historical rule always takes 500 snapshots; the revised rule chooses the number of snapshots from "
                "class size and points per snapshot. The vertical axis is logarithmic so both rules remain visible. "
                f"{ctx}"
            ),
        ))
        if "l_rule" in plot_df.columns:
            revised = plot_df[plot_df["l_rule"].astype(str).str.contains("revised", case=False, na=False)]
            if not revised.empty:
                written.append(plot_faceted_bars(
                    revised,
                    x="landmark_label" if "landmark_label" in revised.columns else "dataset_label",
                    y=reuse_col,
                    facet="dataset_label",
                    hue=None,
                    title=process_figure_title(protocol_bucket, "Revised-rule sampling reuse (linear scale)"),
                    ylabel="Expected reuse relative to minority class count",
                    save_path=viz_dir / "sampling_reuse_revised_rule_faceted.png",
                    hline=1.0,
                    hline_label="Each minority-class row used about once",
                    wrap_width=18,
                    note=(
                        "Same reuse definition as the companion figure, restricted to the revised snapshot-count rule, "
                        "shown on a linear scale. The dashed line is reuse = 1. "
                        f"{ctx}"
                    ),
                ))
    written.append(_write_csv(viz_dir / "sampling_ratio_audit_combined.csv", data))
    return written


def visualize_snapshot_mean_variance_experiment(protocol_bucket: str, experiment: str) -> List[Path]:
    results_root = experiment_results_root(protocol_bucket, experiment)
    viz_dir = experiment_visualizations_dir(protocol_bucket, experiment)
    per_dataset = [results_root / folder / "snapshot_mean_variance.csv" for folder in _registered_dataset_folders()]
    expected = per_dataset + [results_root / "snapshot_mean_variance.csv"]
    frames = [pd.read_csv(p) for p in per_dataset if p.is_file()]
    if not frames and (results_root / "snapshot_mean_variance.csv").is_file():
        frames.append(pd.read_csv(results_root / "snapshot_mean_variance.csv"))
    if not frames:
        _results_not_generated(expected)
    data = _attach_dataset_labels(pd.concat(frames, ignore_index=True), source_col="source")
    viz_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    source = data["source"].astype(str) if "source" in data.columns else pd.Series([""] * len(data))
    data["landmark_label"] = source.map(pretty_setting_label).replace("", "Unknown snapshot size")
    if "feature" in data.columns:
        data["feature_label"] = data["feature"].map(lambda f: COLUMN_DESCRIPTIONS.get(str(f), str(f)))
    headline = data[data["feature"].isin(["g2_0", "g3_1"])].copy() if "feature" in data.columns else data
    ctx = protocol_context_sentence(protocol_bucket)
    if not headline.empty and "mean" in headline.columns:
        written.append(plot_faceted_bars(
            headline,
            x="landmark_label",
            y="mean",
            facet="dataset_label",
            hue="feature_label",
            title="Headline barcode means (mean death in homology 0, mean persistence in homology 1)",
            ylabel="Mean statistic",
            save_path=viz_dir / "barcode_mean_g2_0_g3_1_faceted.png",
            wrap_width=16,
            note=(
                "Each bar is the mean of one barcode statistic across snapshots. "
                "Mean death in homology 0 and mean persistence in homology 1 are shown. "
                "Panels are datasets; colours are the two statistics. These are descriptive summaries of the snapshot cloud, not classifier scores. "
                f"{ctx}"
            ),
        ))
    if not headline.empty and "variance" in headline.columns:
        written.append(plot_faceted_bars(
            headline,
            x="landmark_label",
            y="variance",
            facet="dataset_label",
            hue="feature_label",
            title="Headline barcode variances (mean death in homology 0, mean persistence in homology 1)",
            ylabel="Variance",
            save_path=viz_dir / "barcode_variance_g2_0_g3_1_faceted.png",
            wrap_width=16,
            note=(
                "Each bar is the variance of one barcode statistic across snapshots. "
                "Large whiskers mean the statistic is unstable from snapshot to snapshot. "
                f"{ctx}"
            ),
        ))
    if "feature" in data.columns and "mean" in data.columns:
        apply_publication_viz_style()
        pivot = data.pivot_table(index="feature_label", columns="dataset_label", values="mean", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(11.5, max(6.5, 0.32 * len(pivot) + 1.5)))
        sns.heatmap(pivot, ax=ax, cmap="cividis", linewidths=0.3)
        ax.set_title("Mean barcode statistics by dataset", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        written.append(_finish_and_save(
            fig,
            viz_dir / "barcode_mean_heatmap.png",
            note=(
                "Each cell is the mean of one barcode statistic, averaged over snapshots for that dataset. "
                "Rows are statistics; columns are datasets. Darker or lighter colour is a relative scale, not a classifier score. "
                f"{ctx}"
            ),
        ))
    written.append(_write_csv(viz_dir / "snapshot_mean_variance_combined.csv", data))
    return written


def visualize_algorithm2_experiment(protocol_bucket: str, experiment: str) -> List[Path]:
    results_root = experiment_results_root(protocol_bucket, experiment)
    viz_dir = experiment_visualizations_dir(protocol_bucket, experiment)
    per_dataset = [
        results_root / folder / "algorithm2_permutation_results.csv"
        for folder in _registered_dataset_folders()
    ]
    expected = per_dataset + [results_root / "algorithm2_permutation_results.csv"]
    frames = [pd.read_csv(p) for p in per_dataset if p.is_file()]
    if not frames and (results_root / "algorithm2_permutation_results.csv").is_file():
        frames.append(pd.read_csv(results_root / "algorithm2_permutation_results.csv"))
    if not frames:
        _results_not_generated(expected)
    data = _attach_dataset_labels(pd.concat(frames, ignore_index=True), source_col="source")
    viz_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    source = data["source"].astype(str) if "source" in data.columns else pd.Series([""] * len(data))
    data["landmark_label"] = source.map(pretty_setting_label).replace("", "Unknown snapshot size")
    if "p" in data.columns and "q" in data.columns:
        data["F_label"] = [f"Contrast ({int(p)}, {int(q)})" for p, q in zip(data["p"], data["q"])]
    else:
        data["F_label"] = "Contrast"
    ctx = protocol_context_sentence(protocol_bucket)
    written.append(plot_faceted_bars(
        data,
        x="F_label",
        y="p_value",
        facet="dataset_label",
        hue="landmark_label" if data["landmark_label"].nunique() > 1 else None,
        title=process_figure_title(protocol_bucket, "Algorithm 2 permutation p-values"),
        ylabel="p-value",
        save_path=viz_dir / "algorithm2_pvalues_faceted.png",
        hline=0.05,
        hline_label="Significance threshold 0.05",
        ylim=(0, 1.05),
        wrap_width=14,
        note=(
            "Each bar is a permutation p-value for one barcode-vector contrast. "
            "The dashed line is 0.05. Small values mean the observed contrast is unusual under a random label shuffle. "
            "Panels are datasets; colours are snapshot sizes (fraction of the class used as points per snapshot). "
            f"{ctx}"
        ),
    ))
    if "observed_F_pq" in data.columns:
        written.append(plot_faceted_bars(
            data,
            x="F_label",
            y="observed_F_pq",
            facet="dataset_label",
            hue="landmark_label" if data["landmark_label"].nunique() > 1 else None,
            title=process_figure_title(protocol_bucket, "Algorithm 2 observed contrast statistic"),
            ylabel="Observed contrast statistic",
            save_path=viz_dir / "algorithm2_observed_F_faceted.png",
            wrap_width=14,
            note=(
                "Each bar is the observed contrast statistic before permutation. "
                "Read it together with the p-value figure: a large statistic with a small p-value is evidence against the shuffle null. "
                f"{ctx}"
            ),
        ))
    written.append(_write_csv(viz_dir / "algorithm2_permutation_combined.csv", data))
    return written


def visualize_revised_snapshot_protocol_experiment(protocol_bucket: str, experiment: str) -> List[Path]:
    results_root = experiment_results_root(protocol_bucket, experiment)
    viz_dir = experiment_visualizations_dir(protocol_bucket, experiment)
    per_dataset = [results_root / folder / "ml_results.csv" for folder in _registered_dataset_folders()]
    expected = per_dataset + [results_root / "all_ml_results.csv"]
    frames = []
    for csv_path in per_dataset:
        if csv_path.is_file():
            frame = pd.read_csv(csv_path)
            if "dataset" not in frame.columns:
                frame["dataset"] = csv_path.parent.name
            frames.append(frame)
    if not frames and (results_root / "all_ml_results.csv").is_file():
        frames.append(pd.read_csv(results_root / "all_ml_results.csv"))
    if not frames:
        _results_not_generated(expected)
    data = _attach_dataset_labels(pd.concat(frames, ignore_index=True))
    viz_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    if "model" in data.columns:
        data["model_label"] = data["model"].map(pretty_model_label)
    if "t" in data.columns:
        data["snapshot_points_label"] = data["t"].map(_points_per_snapshot_label)
    models = list(dict.fromkeys(data["model"].tolist())) if "model" in data.columns else []
    model_palette = {pretty_model_label(m): _model_palette(models)[m] for m in models} if models else None
    ctx = protocol_context_sentence(protocol_bucket)

    for metric in ("balanced_accuracy", "f1", "accuracy"):
        if metric not in data.columns:
            continue
        hue = "snapshot_points_label" if "snapshot_points_label" in data.columns and data["snapshot_points_label"].nunique() > 1 else None
        written.append(plot_faceted_bars(
            data,
            x="model_label" if "model_label" in data.columns else "snapshot_points_label",
            y=metric,
            facet="dataset_label",
            hue=hue if "model_label" in data.columns else None,
            title=f"Revised snapshot protocol — {pretty_metric_label(metric)}",
            ylabel=pretty_metric_label(metric),
            save_path=viz_dir / f"{'f1' if metric == 'f1' else metric}_by_model_faceted.png",
            ylim=(0, 1.05),
            palette=model_palette if hue is None and model_palette else None,
            wrap_width=16,
            note=(
                f"Each bar is held-out {pretty_metric_label(metric).lower()} for one classifier under the revised snapshot protocol. "
                "Colours are points per snapshot (the size of one point cloud). Panels are datasets so models are compared within a table. "
                f"{ctx}"
            ),
        ))
        break
    if "f1" in data.columns and "balanced_accuracy" in data.columns:
        written.append(plot_faceted_bars(
            data,
            x="model_label",
            y="f1",
            facet="dataset_label",
            hue="snapshot_points_label" if "snapshot_points_label" in data.columns and data["snapshot_points_label"].nunique() > 1 else None,
            title=process_figure_title(protocol_bucket, "Revised snapshot protocol — F1 score"),
            ylabel="F1 score",
            save_path=viz_dir / "f1_by_model_faceted.png",
            ylim=(0, 1.05),
            wrap_width=16,
            note=(
                "Each bar is held-out F1 for one classifier. Colours are points per snapshot. "
                "Read this next to the balanced-accuracy figure; they are different questions. "
                f"{ctx}"
            ),
        ))

    concern_frames = []
    for folder in _registered_dataset_folders():
        path = results_root / folder / "concern_A_formula_rows.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["dataset"] = folder
            concern_frames.append(frame)
    if concern_frames:
        concern = _attach_dataset_labels(pd.concat(concern_frames, ignore_index=True))
        y = "l_formula" if "l_formula" in concern.columns else concern.columns[-1]
        x = "t" if "t" in concern.columns else concern.columns[0]
        if x == "t":
            concern["points_tick"] = concern[x].map(_points_count_tick)
            x_order = [_points_count_tick(v) for v in sorted(concern[x].astype(float).unique())]
        else:
            concern["points_tick"] = concern[x].astype(str)
            x_order = None
        written.append(plot_faceted_bars(
            concern,
            x="points_tick",
            y=y,
            facet="dataset_label",
            x_order=x_order,
            xlabel="Points per snapshot",
            share_x=False,
            title="Concern A: formula number of snapshots versus points per snapshot",
            ylabel="Number of snapshots from the formula",
            save_path=viz_dir / "concern_A_snapshot_count_faceted.png",
            note=(
                "Each bar is the number of snapshots implied by the design formula, given the chosen points per snapshot. "
                "This is a planning quantity, not a model score. The horizontal axis is points per snapshot. "
                f"{ctx}"
            ),
        ))
    reuse_frames = []
    for folder in _registered_dataset_folders():
        path = results_root / folder / "concern_B_reuse_rows.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["dataset"] = folder
            reuse_frames.append(frame)
    if reuse_frames:
        reuse = _attach_dataset_labels(pd.concat(reuse_frames, ignore_index=True))
        reuse_y = next((c for c in ("reuse_pos", "reuse_neg", "R", "reuse") if c in reuse.columns), None)
        reuse_ylabel = {
            "reuse_pos": "Reuse on the default class",
            "reuse_neg": "Reuse on the non-default class",
            "R": "Reuse ratio",
            "reuse": "Reuse ratio",
        }.get(str(reuse_y), "Reuse relative to class count")
        if reuse_y is None:
            numeric = [c for c in reuse.columns if pd.api.types.is_numeric_dtype(reuse[c]) and c not in {"t", "l", "b"}]
            reuse_y = numeric[0] if numeric else None
        if reuse_y is not None:
            x = "t" if "t" in reuse.columns else "dataset_label"
            x_order = None
            xlabel = None
            if x == "t":
                reuse["points_tick"] = reuse["t"].map(_points_count_tick)
                x_order = [_points_count_tick(v) for v in sorted(reuse["t"].astype(float).unique())]
                x = "points_tick"
                xlabel = "Points per snapshot"
            written.append(plot_faceted_bars(
                reuse,
                x=x,
                y=reuse_y,
                facet="dataset_label",
                x_order=x_order,
                xlabel=xlabel,
                share_x=False,
                title="Concern B: sampling reuse",
                ylabel=reuse_ylabel,
                save_path=viz_dir / "concern_B_reuse_faceted.png",
                hline=1.0,
                hline_label="Each class row used about once",
                note=(
                    "Each bar is sampling reuse on one class pool. "
                    "Values near 1 mean each customer is used about once across snapshots; "
                    "large values mean the same rows reappear in many snapshots. "
                    "The horizontal axis is points per snapshot. "
                    f"{ctx}"
                ),
            ))
    overlap_rows = []
    for folder in _registered_dataset_folders():
        for json_path in (results_root / folder).glob("overlap_*.json"):
            payload = json.loads(_openable(json_path).read_text(encoding="utf-8"))
            for class_name, block in payload.items():
                if not isinstance(block, dict):
                    continue
                summary = block.get("summary") or {}
                overlap_rows.append({
                    "dataset": folder,
                    "file": json_path.name,
                    "class": class_name,
                    "mean_overlap_frac": summary.get("mean_overlap_frac"),
                    "expected_overlap_frac_indep": summary.get("expected_overlap_frac_indep"),
                    "reuse_ratio_tl_over_n": summary.get("reuse_ratio_tl_over_n"),
                })
    if overlap_rows:
        overlap = _attach_dataset_labels(pd.DataFrame(overlap_rows)).dropna(subset=["mean_overlap_frac"])
        if not overlap.empty:
            overlap["class_label"] = overlap["class"].map(lambda c: str(c).replace("_", " ").title())
            written.append(plot_faceted_bars(
                overlap,
                x="class_label",
                y="mean_overlap_frac",
                facet="dataset_label",
                title="Mean pairwise snapshot overlap",
                ylabel="Mean overlap fraction",
                save_path=viz_dir / "overlap_mean_fraction_faceted.png",
                ylim=(0, 1.05),
                note=(
                    "Each bar is the mean fraction of points shared by a pair of snapshots of the same class. "
                    "Independent sampling would share about (points per snapshot / class count). Panels are datasets. "
                    f"{ctx}"
                ),
            ))
            written.append(_write_csv(viz_dir / "overlap_summary.csv", overlap))
    written.append(_write_csv(viz_dir / "ml_results_combined.csv", data))
    return written


def visualize_intrinsic_dimension_experiment(
    protocol_bucket: str = "Statistics",
    experiment: str = "1_Intrinsic_Dimension_Estimation",
) -> List[Path]:
    results_root = experiment_results_root(protocol_bucket, experiment)
    viz_dir = experiment_visualizations_dir(protocol_bucket, experiment)
    expected = [results_root / folder / "intrinsic_dimension_estimates.csv" for folder in _registered_dataset_folders()]
    expected.append(results_root / "intrinsic_dimension_estimates.csv")
    rich_frames = []
    for folder in _registered_dataset_folders():
        path = results_root / folder / "intrinsic_dimension_estimates.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            if "dataset" not in frame.columns:
                frame["dataset"] = folder
            rich_frames.append(frame)
    if not rich_frames and (results_root / "intrinsic_dimension_estimates.csv").is_file():
        rich_frames.append(pd.read_csv(results_root / "intrinsic_dimension_estimates.csv"))
    if not rich_frames:
        _results_not_generated(expected)
    data = _attach_dataset_labels(pd.concat(rich_frames, ignore_index=True))
    viz_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    rename = {
        "two_nn_raw": "two_nn_before_pca",
        "two_nn_pca": "two_nn_after_pca",
        "levina_bickel_raw": "levina_bickel_before_pca",
        "levina_bickel_pca": "levina_bickel_after_pca",
        "pca_components": "pca_components_exp3",
        "variance_retained_pca": "variance_retained_exp3_pca",
    }
    data = data.rename(columns={k: v for k, v in rename.items() if k in data.columns})

    id_map = {
        "two_nn_before_pca": ("Two-NN (in-house)", "Before PCA"),
        "two_nn_after_pca": ("Two-NN (in-house)", "After PCA"),
        "skdim_TwoNN_before_pca": ("Two-NN (skdim)", "Before PCA"),
        "skdim_TwoNN_after_pca": ("Two-NN (skdim)", "After PCA"),
        "levina_bickel_before_pca": ("Levina-Bickel", "Before PCA"),
        "levina_bickel_after_pca": ("Levina-Bickel", "After PCA"),
        "skdim_MLE_before_pca": ("MLE (skdim)", "Before PCA"),
        "skdim_MLE_after_pca": ("MLE (skdim)", "After PCA"),
        "skdim_MiND_ML_before_pca": ("MiND ML", "Before PCA"),
        "skdim_MiND_ML_after_pca": ("MiND ML", "After PCA"),
        "skdim_lPCA_before_pca": ("local PCA", "Before PCA"),
        "skdim_lPCA_after_pca": ("local PCA", "After PCA"),
    }
    long_rows = []
    for col, (family, stage) in id_map.items():
        if col not in data.columns:
            continue
        for _, row in data.iterrows():
            value = row[col]
            if pd.isna(value):
                continue
            long_rows.append({
                "dataset_label": row["dataset_label"],
                "family": family,
                "stage": stage,
                "estimate": float(value),
            })
    melted = pd.DataFrame(long_rows)
    ctx = protocol_context_sentence(protocol_bucket)
    if not melted.empty:
        two_nn = melted[melted["family"].str.contains("Two-NN", na=False)]
        if not two_nn.empty:
            written.append(plot_faceted_bars(
                two_nn,
                x="family",
                y="estimate",
                facet="dataset_label",
                hue="stage",
                title="Two-NN intrinsic dimension before vs after PCA",
                ylabel="Estimated intrinsic dimension",
                save_path=viz_dir / "two_nn_before_after_pca.png",
                wrap_width=16,
                note=(
                    "Each bar is an estimated intrinsic dimension. "
                    "'Before PCA' uses the processed table; 'After PCA' uses the Experiment 3 projection. "
                    "In-house Two-NN and the skdim Two-NN estimator are shown side by side. Panels are datasets. "
                    f"{ctx}"
                ),
            ))
        suite = melted[~melted["family"].str.contains("Two-NN", na=False)]
        if not suite.empty:
            written.append(plot_faceted_bars(
                suite,
                x="family",
                y="estimate",
                facet="dataset_label",
                hue="stage",
                title="Other intrinsic-dimension estimators before vs after PCA",
                ylabel="Estimated intrinsic dimension",
                save_path=viz_dir / "id_estimator_suite_faceted.png",
                wrap_width=12,
                note=(
                    "Each bar is an estimated intrinsic dimension from one estimator. "
                    "Colours mark before versus after the Experiment 3 PCA projection. Panels are datasets. "
                    f"{ctx}"
                ),
            ))
    rank_cols = [c for c in ("pca_components_exp3", "n_components_for_90pct") if c in data.columns]
    if rank_cols:
        rank = data.melt(id_vars=["dataset_label"], value_vars=rank_cols, var_name="rank_kind", value_name="n_components")
        rank["rank_label"] = rank["rank_kind"].map({
            "pca_components_exp3": "Exp 3 PCA rank",
            "n_components_for_90pct": "Components for 90% variance",
        })
        written.append(plot_faceted_bars(
            rank,
            x="rank_label",
            y="n_components",
            facet="dataset_label",
            title="PCA rank used in Exp 3 vs components needed for 90% variance",
            ylabel="Number of components",
            save_path=viz_dir / "pca_rank_vs_90pct.png",
            wrap_width=18,
            note=(
                "Bars compare the number of principal components actually used in Experiment 3 "
                "with the number needed to retain 90% of variance. Panels are datasets. "
                f"{ctx}"
            ),
        ))
    if "variance_retained_exp3_pca" in data.columns:
        written.append(plot_grouped_bars(
            data,
            x="dataset_label",
            y="variance_retained_exp3_pca",
            title="Variance retained by Exp 3 PCA rank",
            ylabel="Variance retained",
            save_path=viz_dir / "pca_variance_retained.png",
            hline=0.90,
            hline_label="90% target",
            ylim=(0, 1.05),
            wrap_width=14,
            note=(
                "Each bar is the fraction of variance kept by the Experiment 3 PCA rank. "
                "The dashed line is the 90% target. "
                f"{ctx}"
            ),
        ))
    written.append(_write_csv(viz_dir / "intrinsic_dimension_estimates_combined.csv", data))
    return written


def visualize_experiment_folder(protocol_bucket: str, experiment: str) -> List[Path]:
    """Dispatch visualization for one active experiment folder.

    Run from `5_Experiments/{Bucket}/{Experiment}/visualize_results.py`.
    Writes figures only into `6_Results/{Bucket}/{Experiment}/Visualizations/`.
    Raises ResultsNotGeneratedError with the expected artefact paths when
    nothing has been produced yet.
    """
    apply_publication_viz_style()
    viz_dir = experiment_visualizations_dir(protocol_bucket, experiment)
    print(f"Visualizing {process_display_name(protocol_bucket)} / {experiment}")
    print(f"Figures -> {viz_dir}")
    hide_axes = experiment == "4_Dropping_Correlated_Barcode_Statistics_Columns"
    if experiment == "6_Sampling_Ratio_Audit":
        written = visualize_sampling_ratio_audit_experiment(protocol_bucket, experiment)
    elif experiment == "7_Snapshot_Mean_Variance":
        written = visualize_snapshot_mean_variance_experiment(protocol_bucket, experiment)
    elif experiment == "8_Null_Hypothesis_Algorithm2":
        written = visualize_algorithm2_experiment(protocol_bucket, experiment)
    elif experiment == "9_Revised_Snapshot_Protocol":
        written = visualize_revised_snapshot_protocol_experiment(protocol_bucket, experiment)
    elif protocol_bucket == "Statistics" or experiment == "1_Intrinsic_Dimension_Estimation":
        written = visualize_intrinsic_dimension_experiment(protocol_bucket, experiment)
    else:
        written = visualize_model_pickle_experiment(
            protocol_bucket, experiment, hide_axis_labels=hide_axes
        )
    print(f"Wrote {len(written)} artefact(s) -> {viz_dir}")
    return written


# =============================================================================
# English snapshot notation (symbol mapping: docs/Notation.md)
# Additive helpers — names do not collide with existing t/l internals.
# =============================================================================
SNAPSHOT_NOTATION_ENGLISH = {
    "t": "points per snapshot",
    "l": "number of snapshots",
    "L": "snapshot size as a percent of the class",
    "n1": "minority class count",
    "n2": "majority class count",
    "R": "reuse ratio = (points per snapshot × number of snapshots) / minority class count",
}


def reuse_ratio_from_counts(
    points_per_snapshot: int,
    n_snapshots: int,
    minority_count: int,
) -> float:
    """Reuse ratio = (points per snapshot × number of snapshots) / minority class count."""
    if minority_count <= 0:
        return float("nan")
    return float(points_per_snapshot * n_snapshots) / float(minority_count)

# =============================================================================
# SNAPSHOT SAMPLE SIZE HELPERS (Ripser / IO / figures)
# These are called from 5_Experiments/Snapshot_Sample_Size dataset scripts.
# The dataset script itself shows load, scale, PCA, split, and undersample.
# =============================================================================

# -----------------------------------------------------------------------------
# Locked design
# -----------------------------------------------------------------------------
BUCKET = "Snapshot_Sample_Size"
SHARED_EXPERIMENT = "0_Shared_Pools"
CANDIDATE_POINTS_PER_SNAPSHOT: Tuple[int, ...] = (15, 30, 45, 60)
N_SNAPSHOTS_GRID: Tuple[int, ...] = (15, 30, 45, 60)
N_TRAIN_POOL = 60
N_TEST_SNAPSHOTS = 15
N_REPEATS = 10
CUSTOMER_SPLIT_SEED = 0
PCA_RANDOM_STATE = 42  # matches Exp 3 / DatasetConfig geometry
MODEL_RANDOM_STATE = 0
HOMOLOGY_DIM = 2
Z_95 = 1.96

PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "Historical_Late_Split_Balanced_TDA": {
        "split_timing": "late",
        "undersample": True,
        "display": "Late split and undersample (the original historical run)",
    },
    "Early_Split_TDA": {
        "split_timing": "early",
        "undersample": True,
        "display": "Early split and undersample",
    },
    "No_Undersampling": {
        "split_timing": "late",
        "undersample": False,
        "display": "Late split, no undersample",
    },
    "Early_Split_TDA_And_No_Undersampling": {
        "split_timing": "early",
        "undersample": False,
        "display": "Early split, no undersample",
    },
}

DATASET_RUN_ORDER: Tuple[str, ...] = (
    "pkdd_czech",
    "south_german_credit",
    "statlog_german",
    "taiwan_bankruptcy",
    "polish_bankruptcy",
    "credit_card_default",
)

ITEM_FOLDERS = {
    "1": "1_Snapshot_Count_Sweep",
    "2": "2_Points_Per_Snapshot_Sweep",
    "4": "3_Snapshot_Count_Across_Cloud_Sizes",
}

CLASSIFIER_ORDER = ("svm", "logistic", "knn", "xgb", "random_forest")
CLASSIFIER_DISPLAY = {
    "svm": "SVM",
    "logistic": "Logistic Regression",
    "knn": "KNN",
    "xgb": "XGBoost",
    "random_forest": "Random Forest",
}
HIGHLIGHT_MODELS = {"svm", "logistic"}
# Okabe–Ito (Wong 2011), colourblind-safe. Yellow is replaced by dark gold so every
# hue holds on white. This mapping is locked for every Snapshot Sample Size figure.
# SVM / Logistic take the two strongest hues; the other three are full saturation.
MODEL_COLORS = {
    "svm": "#0072B2",  # blue
    "logistic": "#D55E00",  # vermillion
    "knn": "#009E73",  # bluish green
    "xgb": "#C78D00",  # dark gold
    "random_forest": "#CC79A7",  # reddish purple
}
MODEL_LINEWIDTH = {
    "svm": 2.4,
    "logistic": 2.4,
    "knn": 1.8,
    "xgb": 1.8,
    "random_forest": 1.8,
}
MODEL_MARKERS = {
    "svm": "o",
    "logistic": "o",
    "knn": "s",
    "xgb": "^",
    "random_forest": "D",
}
MODEL_MARKERSIZE = {
    "svm": 8.0,
    "logistic": 8.0,
    "knn": 6.6,
    "xgb": 7.0,
    "random_forest": 6.6,
}
MODEL_LINE_ALPHA = 1.0  # never fade the lines themselves (no alpha < 0.9)
CI_RIBBON_ALPHA = 0.28
# Families of cloud size (item 4) — same Okabe–Ito family, assigned by rank when a
# protocol drops a candidate or uses a documented clip (for example 14 on PKDD).
CLOUD_SIZE_COLOR_CYCLE = ("#0072B2", "#009E73", "#C78D00", "#CC79A7", "#D55E00")
CLOUD_SIZE_MARKER_CYCLE = ("o", "s", "^", "D", "v")

METRIC_DISPLAY = {"f1": "F1", "accuracy": "Accuracy"}
BARCODE_COLUMNS = [f"g{i}_{j}" for j in range(HOMOLOGY_DIM) for i in range(1, 13)]




def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: Any) -> Path:
    path = win_long_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2)
    return path


def load_json(path: Path) -> Any:
    with win_long_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reuse_ratio(
    points_per_snapshot: int,
    n_snapshots: int,
    minority_count: int,
) -> float:
    """(points per snapshot × number of snapshots) / minority class count."""
    if minority_count <= 0:
        return float("nan")
    return float(points_per_snapshot * n_snapshots) / float(minority_count)


def snapshot_size_percent_of_class(points_per_snapshot: int, class_count: int) -> float:
    if class_count <= 0:
        return float("nan")
    return 100.0 * float(points_per_snapshot) / float(class_count)


# =============================================================================
# Protocol clouds (honest to each arm; customer split seed is locked at 0)
# =============================================================================
def undersample_xy(
    X: pd.DataFrame,
    y: pd.Series,
    positive_label: int = 1,
    random_state: int = CUSTOMER_SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.Series]:
    data = X.copy()
    data["__y__"] = pd.Series(y).values
    pos = data[data["__y__"] == positive_label]
    neg = data[data["__y__"] != positive_label]
    n = min(len(pos), len(neg))
    if n < 2:
        raise ValueError(f"Cannot undersample: pos={len(pos)} neg={len(neg)}")
    pos = pos.sample(n=n, random_state=random_state)
    neg = neg.sample(n=n, random_state=random_state)
    out = pd.concat([pos, neg], ignore_index=True)
    return out.drop(columns=["__y__"]), out["__y__"].astype(int)


def split_classes(
    X: pd.DataFrame,
    y: pd.Series,
    positive_label: int = 1,
) -> Dict[str, pd.DataFrame]:
    data = X.copy()
    data["__y__"] = pd.Series(y).values
    pos = data[data["__y__"] == positive_label].drop(columns=["__y__"]).reset_index(drop=True)
    neg = data[data["__y__"] != positive_label].drop(columns=["__y__"]).reset_index(drop=True)
    return {"default": pos, "non-default": neg}


def split_customers(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = CUSTOMER_SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 stratified customer split. Early-split arms only."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def class_pool_sizes(classes: Dict[str, pd.DataFrame]) -> Tuple[int, int, int, int]:
    """Return (n_default, n_non_default, minority, majority)."""
    n_pos = int(len(classes["default"]))
    n_neg = int(len(classes["non-default"]))
    return n_pos, n_neg, int(min(n_pos, n_neg)), int(max(n_pos, n_neg))


def binding_class_count(
    train_classes: Dict[str, pd.DataFrame],
    test_classes: Dict[str, pd.DataFrame],
    split_timing: str,
) -> int:
    """Largest cloud that still fits a without-replacement draw from every used pool."""
    _tr_pos, _tr_neg, train_min, train_maj = class_pool_sizes(train_classes)
    _te_pos, _te_neg, test_min, _test_maj = class_pool_sizes(test_classes)
    if split_timing == "early":
        return int(min(train_min, test_min))
    return int(min(train_min, train_maj))


def _add_missing_indicators(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    miss_tr = X_train.isna().astype(float)
    miss_te = X_test.isna().astype(float)
    miss_tr.columns = [f"miss_{c}" for c in X_train.columns]
    miss_te.columns = [f"miss_{c}" for c in X_test.columns]
    keep = [c for c in miss_tr.columns if miss_tr[c].sum() > 0]
    if not keep:
        return X_train, X_test
    return (
        pd.concat([X_train, miss_tr[keep]], axis=1),
        pd.concat([X_test, miss_te[keep]], axis=1),
    )


def fit_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int,
    random_state: int = PCA_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """MinMaxScaler + PCA fitted on train only; test is transformed with that fit."""
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    Xte_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    Xtr_imp, Xte_imp = _add_missing_indicators(Xtr_imp, Xte_imp)
    scaler = MinMaxScaler()
    Xtr_s = scaler.fit_transform(Xtr_imp)
    Xte_s = scaler.transform(Xte_imp)
    n_comp = min(n_components, max(1, Xtr_s.shape[0] - 1), Xtr_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    cols = [f"PCA_{i}" for i in range(1, n_comp + 1)]
    Xtr_p = pd.DataFrame(pca.fit_transform(Xtr_s), columns=cols, index=X_train.index)
    Xte_p = pd.DataFrame(pca.transform(Xte_s), columns=cols, index=X_test.index)
    return Xtr_p, Xte_p, float(pca.explained_variance_ratio_.sum())


def early_split_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    test_size: float = 0.2,
    random_state: int = CUSTOMER_SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    X_train, X_test, y_train, y_test = split_customers(
        X, y, test_size=test_size, random_state=random_state
    )
    Xtr_p, Xte_p, var = fit_pca(X_train, X_test, n_components=n_components)
    return Xtr_p, Xte_p, y_train, y_test, var


def late_full_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
) -> Tuple[pd.DataFrame, pd.Series, float]:
    """Full-table PCA (historical late-split arms). Snapshots are then split."""
    miss = X.isna().astype(float)
    miss.columns = [f"miss_{c}" for c in X.columns]
    keep = [c for c in miss.columns if miss[c].sum() > 0]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    if keep:
        X_imp = pd.concat([X_imp, miss[keep]], axis=1)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X_imp)
    n_comp = min(n_components, max(1, Xs.shape[0] - 1), Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=PCA_RANDOM_STATE)
    cols = [f"PCA_{i}" for i in range(1, n_comp + 1)]
    Xp = pd.DataFrame(pca.fit_transform(Xs), columns=cols, index=X.index)
    return Xp, y, float(pca.explained_variance_ratio_.sum())


fit_pca_full_table = late_full_pca


def prepare_protocol_pools(
    dataset_key: str,
    protocol_bucket: str,
    n_components: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build class pools that are honest to the arm.

    Early-split arms: split customers first (random_state=0), PCA on train only,
    optional undersample inside each split. Train snapshots come from train
    pools; test snapshots come from test pools.

    Late-split arms: optional undersample on the full table, full-table PCA,
    then snapshots are drawn from the full class pools and split at snapshot
    level (historical barcode-row split). Nested prefixes change only the
    training snapshot count; 15 test snapshots are held out.
    """
    if protocol_bucket not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol_bucket!r}")
    spec = PROTOCOLS[protocol_bucket]
    X, y, cfg = load_processed_features(dataset_key)
    pca_n = int(n_components) if n_components is not None else dataset_pca_rank(dataset_key)
    split_timing = spec["split_timing"]
    undersample = bool(spec["undersample"])
    positive = int(cfg.positive_label)

    if split_timing == "early":
        X_train, X_test, y_train, y_test, var = early_split_pca(
            X, y, n_components=pca_n, random_state=CUSTOMER_SPLIT_SEED
        )
        if undersample:
            X_train, y_train = undersample_xy(
                X_train, y_train, positive_label=positive, random_state=CUSTOMER_SPLIT_SEED
            )
            X_test, y_test = undersample_xy(
                X_test, y_test, positive_label=positive, random_state=CUSTOMER_SPLIT_SEED
            )
        train_classes = split_classes(X_train, y_train, positive_label=positive)
        test_classes = split_classes(X_test, y_test, positive_label=positive)
        pca_fit = "train_only"
        snapshot_split = "customer"
    else:
        if undersample:
            X, y = undersample_xy(
                X, y, positive_label=positive, random_state=CUSTOMER_SPLIT_SEED
            )
        Xp, y_full, var = late_full_pca(X, y, n_components=pca_n)
        full_classes = split_classes(Xp, y_full, positive_label=positive)
        train_classes = full_classes
        test_classes = full_classes
        pca_fit = "full_table"
        snapshot_split = "snapshot"
        y_train = y_full
        y_test = y_full

    train_pos = int(len(train_classes["default"]))
    train_neg = int(len(train_classes["non-default"]))
    test_pos = int(len(test_classes["default"]))
    test_neg = int(len(test_classes["non-default"]))
    train_min = int(min(train_pos, train_neg))
    train_maj = int(max(train_pos, train_neg))
    test_min = int(min(test_pos, test_neg))
    test_maj = int(max(test_pos, test_neg))
    if snapshot_split == "customer":
        binding = int(min(train_min, test_min))
        binding_note = (
            "Early split: a candidate is dropped unless it is strictly smaller "
            "than both the train and test class pools (without-replacement draw)."
        )
    else:
        binding = int(min(train_min, train_maj))
        binding_note = (
            "Late split: snapshots are drawn from the full class pools after "
            "full-table PCA (and optional undersample). The binding count is "
            "the smaller class on that pool."
        )

    return {
        "dataset_key": dataset_key,
        "display_name": cfg.display_name,
        "folder_name": cfg.folder_name,
        "protocol_bucket": protocol_bucket,
        "split_timing": split_timing,
        "undersample": undersample,
        "pca_rank": pca_n,
        "pca_fit": pca_fit,
        "pca_variance_retained": float(var),
        "snapshot_split": snapshot_split,
        "train_classes": train_classes,
        "test_classes": test_classes,
        "train_minority_count": train_min,
        "train_majority_count": train_maj,
        "test_minority_count": test_min,
        "test_majority_count": test_maj,
        "binding_class_count": binding,
        "binding_note": binding_note,
        "customer_split_random_state": CUSTOMER_SPLIT_SEED,
    }


def surviving_points_per_snapshot(
    binding_class_count: int,
    candidates: Sequence[int] = CANDIDATE_POINTS_PER_SNAPSHOT,
) -> Dict[str, Any]:
    """
    Drop any candidate with points_per_snapshot >= class count.

    No silent clipping. If every candidate is dropped, a single clipped value
    of (binding_class_count - 1) is added and flagged so captions can say so.
    """
    dropped = []
    kept: List[int] = []
    for value in candidates:
        if value >= binding_class_count:
            dropped.append(
                {
                    "points_per_snapshot": int(value),
                    "reason": (
                        f"cannot draw {value} points without replacement from a "
                        f"class pool of {binding_class_count}"
                    ),
                }
            )
        else:
            kept.append(int(value))
    clipped = False
    clip_note = None
    if not kept:
        clipped_value = max(5, int(binding_class_count) - 1)
        kept = [clipped_value]
        clipped = True
        clip_note = (
            f"Every candidate in {list(candidates)} was at least the class count "
            f"{binding_class_count}, so the grid uses a documented clipped value "
            f"{clipped_value} = class count minus one. This is not silent clipping."
        )
    return {
        "candidates": [int(v) for v in candidates],
        "surviving": kept,
        "dropped": dropped,
        "clipped": clipped,
        "clip_note": clip_note,
        "default_points_per_snapshot": max(kept),
        "default_rule": (
            "Largest surviving candidate in {15, 30, 45, 60} that is strictly "
            "smaller than the protocol's binding class pool. Historical Exp 3 "
            "cloud sizes (for example 331 on DCCCD) sit above this grid, so they "
            "are not used as the item-1 default. Item 4 already varies cloud size "
            "inside the surviving grid."
        ),
    }


def resolve_grid(dataset_key: str, protocol_bucket: str) -> Dict[str, Any]:
    pools = prepare_protocol_pools(dataset_key, protocol_bucket)
    grid = surviving_points_per_snapshot(pools["binding_class_count"])
    design = {
        "dataset_key": dataset_key,
        "display_name": pools["display_name"],
        "folder_name": pools["folder_name"],
        "protocol_bucket": protocol_bucket,
        "split_timing": pools["split_timing"],
        "undersample": pools["undersample"],
        "pca_rank": pools["pca_rank"],
        "pca_fit": pools["pca_fit"],
        "pca_variance_retained": pools["pca_variance_retained"],
        "snapshot_split": pools["snapshot_split"],
        "train_minority_count": pools["train_minority_count"],
        "train_majority_count": pools["train_majority_count"],
        "test_minority_count": pools["test_minority_count"],
        "test_majority_count": pools["test_majority_count"],
        "binding_class_count": pools["binding_class_count"],
        "binding_note": pools["binding_note"],
        "n_snapshots_grid": list(N_SNAPSHOTS_GRID),
        "n_train_pool": N_TRAIN_POOL,
        "n_test_snapshots": N_TEST_SNAPSHOTS,
        "n_repeats": N_REPEATS,
        "customer_split_random_state": CUSTOMER_SPLIT_SEED,
        **grid,
        "reuse_at_default": reuse_ratio(
            grid["default_points_per_snapshot"],
            N_TRAIN_POOL,
            pools["train_minority_count"],
        ),
        "snapshot_size_percent_of_minority_at_default": snapshot_size_percent_of_class(
            grid["default_points_per_snapshot"],
            pools["train_minority_count"],
        ),
    }
    return design, pools


# =============================================================================
# Snapshot draws + Ripser (generate 60 once; nested prefixes reuse barcodes)
# =============================================================================
def _draw_index_sets(
    n_pool: int,
    points_per_snapshot: int,
    n_snapshots: int,
    random_state: int,
) -> List[List[int]]:
    rng = check_random_state(random_state)
    if points_per_snapshot >= n_pool:
        raise ValueError(
            f"Cannot draw {points_per_snapshot} points without replacement "
            f"from a pool of {n_pool}"
        )
    sets: List[List[int]] = []
    for _ in range(n_snapshots):
        idx = rng.choice(n_pool, size=points_per_snapshot, replace=False)
        sets.append(sorted(int(i) for i in idx.tolist()))
    return sets


def _nested_prefix_order(n_pool: int, random_state: int) -> List[int]:
    rng = check_random_state(random_state)
    order = np.arange(n_pool)
    rng.shuffle(order)
    return [int(i) for i in order.tolist()]


def artefact_root(kind: str, protocol_bucket: str, dataset_folder: str) -> Path:
    return (
        REPO_ROOT
        / "1_Data"
        / kind
        / BUCKET
        / protocol_bucket
        / SHARED_EXPERIMENT
        / dataset_folder
    )


def results_shared_dir(protocol_bucket: str, dataset_folder: str) -> Path:
    return REPO_ROOT / "6_Results" / BUCKET / "shared" / protocol_bucket / dataset_folder


def experiment_results_dir(
    item_folder: str, protocol_bucket: str, dataset_folder: str
) -> Path:
    return REPO_ROOT / "6_Results" / BUCKET / item_folder / protocol_bucket / dataset_folder


def visualizations_dir(item_folder: str) -> Path:
    return REPO_ROOT / "6_Results" / BUCKET / item_folder / "Visualizations"


def _sample_size_save_figure(fig, path: Path, dpi: int = 160) -> Path:
    """Write a PNG through the Windows long-path prefix when needed."""
    target = win_long_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(os.fspath(target), dpi=dpi, bbox_inches="tight")
    return Path(os.path.abspath(os.fspath(path)))


def pool_dir(
    protocol_bucket: str,
    dataset_folder: str,
    points_per_snapshot: int,
    repeat: int,
) -> Path:
    return (
        artefact_root("Barcode_Statistics", protocol_bucket, dataset_folder)
        / f"pps_{points_per_snapshot}"
        / f"repeat_{repeat:02d}"
    )


def landmark_dir(
    protocol_bucket: str,
    dataset_folder: str,
    points_per_snapshot: int,
    repeat: int,
) -> Path:
    return (
        artefact_root("Landmark_Sets", protocol_bucket, dataset_folder)
        / f"pps_{points_per_snapshot}"
        / f"repeat_{repeat:02d}"
    )


def tda_dir(
    protocol_bucket: str,
    dataset_folder: str,
    points_per_snapshot: int,
    repeat: int,
) -> Path:
    return (
        artefact_root("TDA_Datasets", protocol_bucket, dataset_folder)
        / f"pps_{points_per_snapshot}"
        / f"repeat_{repeat:02d}"
    )


def _barcode_cache_path(cache_root: Path, split: str, class_name: str, index: int) -> Path:
    return win_long_path(cache_root / split / class_name / f"snapshot_{index:03d}.csv")


def _points_to_barcode_row(points: np.ndarray, label: int) -> List[float]:
    from ripser import ripser

    dgms = ripser(np.asarray(points, dtype=float), maxdim=HOMOLOGY_DIM - 1)["dgms"]
    row: List[float] = []
    for dim in range(HOMOLOGY_DIM):
        row.extend(compute_barcode_statistics(dgms[dim]))
    row.append(int(label))
    return row


def _write_barcode_row(path: Path, row: Sequence[float]) -> None:
    path = win_long_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([list(row)], columns=BARCODE_COLUMNS + ["label"])
    frame.to_csv(path, index=False)


def _read_barcode_row(path: Path) -> pd.Series:
    frame = pd.read_csv(win_long_path(path))
    return frame.iloc[0]


def _assemble_split_matrix(
    cache_root: Path,
    split: str,
    index_order: Sequence[int],
    n_keep: Optional[int] = None,
) -> pd.DataFrame:
    keep = list(index_order) if n_keep is None else list(index_order[:n_keep])
    frames = []
    for class_name, label in (("default", 1), ("non-default", 0)):
        rows = []
        for idx in keep:
            path = _barcode_cache_path(cache_root, split, class_name, idx)
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append(_read_barcode_row(path))
        part = pd.DataFrame(rows).reset_index(drop=True)
        part["label"] = label
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


assemble_split_matrix = _assemble_split_matrix


def snapshot_draw_seeds(points_per_snapshot: int, repeat: int) -> Dict[str, int]:
    return {
        "train_seed": 10_000 + 1_000 * repeat + int(points_per_snapshot),
        "test_seed": 80_000 + 1_000 * repeat + int(points_per_snapshot),
        "shuffle_seed": 50_000 + 1_000 * repeat + int(points_per_snapshot),
    }


def repeat_metrics_path(
    protocol_bucket: str,
    dataset_folder: str,
    points_per_snapshot: int,
    repeat: int,
) -> Path:
    return win_long_path(
        results_shared_dir(protocol_bucket, dataset_folder)
        / f"repeat_{repeat:02d}_pps_{points_per_snapshot}_metrics.csv"
    )


def shared_metrics_exist(dataset_folder: str) -> bool:
    root = REPO_ROOT / "6_Results" / BUCKET / "shared"
    return any(root.glob(f"*/{dataset_folder}/repeat_*_pps_*_metrics.csv"))




def draw_snapshot_pool(
    train_classes: Dict[str, pd.DataFrame],
    test_classes: Dict[str, pd.DataFrame],
    protocol_bucket: str,
    dataset_folder: str,
    points_per_snapshot: int,
    repeat: int,
    n_train_snapshots: int = N_TRAIN_POOL,
    n_test_snapshots: int = N_TEST_SNAPSHOTS,
    skip_existing: bool = True,
) -> Tuple[Dict[str, Any], List[int], Dict[str, int]]:
    """
    Draw 60 train + 15 test index sets. No replacement inside a snapshot.

    Nested prefixes 15 ⊂ 30 ⊂ 45 ⊂ 60 are a shuffle of that same train pool.
    skip_existing reloads index_sets.json when it is already on disk.
    """
    seeds = snapshot_draw_seeds(points_per_snapshot, repeat)
    lm_root = landmark_dir(protocol_bucket, dataset_folder, points_per_snapshot, repeat)
    cache = pool_dir(protocol_bucket, dataset_folder, points_per_snapshot, repeat)
    index_path = win_long_path(lm_root / "index_sets.json")
    order_path = win_long_path(lm_root / "nested_prefix_order.json")
    meta_path = win_long_path(cache / "pool_meta.json")

    if skip_existing and index_path.exists() and order_path.exists():
        return load_json(index_path), load_json(order_path), seeds
    if skip_existing and meta_path.exists():
        meta = load_json(meta_path)
        prefix_order = meta.get("nested_prefix_order")
        if (
            prefix_order
            and int(meta.get("n_train_complete", 0)) >= n_train_snapshots
            and int(meta.get("n_test_complete", 0)) >= n_test_snapshots
        ):
            index_sets = load_json(index_path) if index_path.exists() else {"train": {}, "test": {}}
            return index_sets, list(prefix_order), seeds

    index_sets: Dict[str, Any] = {"train": {}, "test": {}}
    for class_name, frame in train_classes.items():
        index_sets["train"][class_name] = _draw_index_sets(
            n_pool=len(frame),
            points_per_snapshot=points_per_snapshot,
            n_snapshots=n_train_snapshots,
            random_state=seeds["train_seed"] + (0 if class_name == "default" else 1),
        )
    for class_name, frame in test_classes.items():
        index_sets["test"][class_name] = _draw_index_sets(
            n_pool=len(frame),
            points_per_snapshot=points_per_snapshot,
            n_snapshots=n_test_snapshots,
            random_state=seeds["test_seed"] + (0 if class_name == "default" else 1),
        )
    prefix_order = _nested_prefix_order(n_train_snapshots, random_state=seeds["shuffle_seed"])
    save_json(index_path, index_sets)
    save_json(order_path, prefix_order)
    return index_sets, prefix_order, seeds


def compute_barcodes_for_pool(
    train_classes: Dict[str, pd.DataFrame],
    test_classes: Dict[str, pd.DataFrame],
    index_sets: Dict[str, Any],
    prefix_order: Sequence[int],
    seeds: Dict[str, int],
    dataset_key: str,
    protocol_bucket: str,
    dataset_folder: str,
    points_per_snapshot: int,
    repeat: int,
    minority_count: int,
    majority_count: int,
    skip_existing: bool = True,
    n_train_snapshots: int = N_TRAIN_POOL,
    n_test_snapshots: int = N_TEST_SNAPSHOTS,
) -> Dict[str, Any]:
    """Ripser each snapshot in the pool. skip_existing is per-snapshot."""
    cache = pool_dir(protocol_bucket, dataset_folder, points_per_snapshot, repeat)
    lm_root = landmark_dir(protocol_bucket, dataset_folder, points_per_snapshot, repeat)
    tda_root = tda_dir(protocol_bucket, dataset_folder, points_per_snapshot, repeat)
    meta_path = win_long_path(cache / "pool_meta.json")
    train_csv = win_long_path(tda_root / "train_pool.csv")
    test_csv = win_long_path(tda_root / "test_pool.csv")

    if skip_existing and meta_path.exists() and train_csv.exists() and test_csv.exists():
        meta = load_json(meta_path)
        if int(meta.get("n_train_complete", 0)) >= n_train_snapshots and int(
            meta.get("n_test_complete", 0)
        ) >= n_test_snapshots:
            print(
                f"[skip] barcodes {protocol_bucket}/{dataset_folder} "
                f"pps={points_per_snapshot} repeat={repeat}"
            )
            return meta

    label_map = {"default": 1, "non-default": 0}
    n_train_done = 0
    n_test_done = 0
    t0 = time.time()
    for split, n_need, class_frames in (
        ("train", n_train_snapshots, train_classes),
        ("test", n_test_snapshots, test_classes),
    ):
        for class_name, frame in class_frames.items():
            values = frame.to_numpy(dtype=float)
            for i in range(n_need):
                out_path = _barcode_cache_path(cache, split, class_name, i)
                if skip_existing and out_path.exists():
                    if split == "train":
                        n_train_done += 1
                    else:
                        n_test_done += 1
                    continue
                idx = index_sets[split][class_name][i]
                points = values[idx]
                row = _points_to_barcode_row(points, label=label_map[class_name])
                _write_barcode_row(out_path, row)
                lm_path = win_long_path(
                    lm_root / split / class_name / f"landmarks_{i:03d}.csv"
                )
                lm_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(points, columns=list(frame.columns)).to_csv(
                    lm_path, index=False
                )
                if split == "train":
                    n_train_done += 1
                else:
                    n_test_done += 1
                if (n_train_done + n_test_done) % 20 == 0:
                    print(
                        f"  Ripser {protocol_bucket}/{dataset_folder} pps={points_per_snapshot} "
                        f"repeat={repeat}: train={n_train_done}/{n_train_snapshots * 2} "
                        f"test={n_test_done}/{n_test_snapshots * 2}"
                    )

    train_pool = assemble_split_matrix(cache, "train", list(range(n_train_snapshots)))
    test_pool = assemble_split_matrix(cache, "test", list(range(n_test_snapshots)))
    tda_root = win_long_path(tda_root)
    tda_root.mkdir(parents=True, exist_ok=True)
    train_pool.to_csv(train_csv, index=False)
    test_pool.to_csv(test_csv, index=False)
    save_json(win_long_path(lm_root / "index_sets.json"), index_sets)
    save_json(win_long_path(lm_root / "nested_prefix_order.json"), list(prefix_order))

    meta = {
        "dataset_key": dataset_key,
        "folder_name": dataset_folder,
        "protocol_bucket": protocol_bucket,
        "points_per_snapshot": int(points_per_snapshot),
        "repeat": int(repeat),
        "n_train_pool": n_train_snapshots,
        "n_test_snapshots": n_test_snapshots,
        "nested_prefix_order": list(prefix_order),
        "n_train_complete": n_train_snapshots * 2,
        "n_test_complete": n_test_snapshots * 2,
        "train_seed": seeds["train_seed"],
        "test_seed": seeds["test_seed"],
        "shuffle_seed": seeds["shuffle_seed"],
        "elapsed_seconds": round(time.time() - t0, 3),
        "train_pool_csv": str(train_csv),
        "test_pool_csv": str(test_csv),
        "reuse_ratio_at_60": reuse_ratio(
            points_per_snapshot, n_train_snapshots, minority_count
        ),
        "minority_count": minority_count,
        "majority_count": majority_count,
        "snapshot_size_percent_of_class": snapshot_size_percent_of_class(
            points_per_snapshot, minority_count
        ),
    }
    save_json(meta_path, meta)
    print(
        f"[ok] barcodes {protocol_bucket}/{dataset_folder} pps={points_per_snapshot} "
        f"repeat={repeat} in {meta['elapsed_seconds']}s"
    )
    return meta


def ensure_barcode_pool(
    dataset_key: str,
    protocol_bucket: str,
    points_per_snapshot: int,
    repeat: int,
    pools: Optional[Dict[str, Any]] = None,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Draw 60 train + 15 test snapshots and Ripser each cloud once.

    Nested snapshot-count values reuse these barcodes. skip_existing is
    per-snapshot so a killed run resumes.
    """
    if pools is None:
        _, pools = resolve_grid(dataset_key, protocol_bucket)
    folder = pools["folder_name"]
    index_sets, prefix_order, seeds = draw_snapshot_pool(
        train_classes=pools["train_classes"],
        test_classes=pools["test_classes"],
        protocol_bucket=protocol_bucket,
        dataset_folder=folder,
        points_per_snapshot=points_per_snapshot,
        repeat=repeat,
        skip_existing=skip_existing,
    )
    return compute_barcodes_for_pool(
        train_classes=pools["train_classes"],
        test_classes=pools["test_classes"],
        index_sets=index_sets,
        prefix_order=prefix_order,
        seeds=seeds,
        dataset_key=dataset_key,
        protocol_bucket=protocol_bucket,
        dataset_folder=folder,
        points_per_snapshot=points_per_snapshot,
        repeat=repeat,
        minority_count=int(pools["train_minority_count"]),
        majority_count=int(pools["train_majority_count"]),
        skip_existing=skip_existing,
    )


def _default_models() -> Dict[str, Any]:
    """Exp 1 TDA default hyperparameters (empty kwargs), seeds fixed for CI."""
    return {
        "svm": SVC(random_state=MODEL_RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "xgb": XGBClassifier(
            eval_metric="logloss",
            random_state=MODEL_RANDOM_STATE,
            verbosity=0,
        ),
        "logistic": LogisticRegression(max_iter=1000, random_state=MODEL_RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=MODEL_RANDOM_STATE),
    }


def fit_default_classifiers(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    feature_cols = [c for c in train_df.columns if c != "label"]
    X_tr = train_df[feature_cols].to_numpy(dtype=float)
    y_tr = train_df["label"].to_numpy()
    X_te = test_df[feature_cols].to_numpy(dtype=float)
    y_te = test_df["label"].to_numpy()
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    rows = []
    for name, model in _default_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append(
            {
                "model": name,
                "model_display": CLASSIFIER_DISPLAY[name],
                "accuracy": float(accuracy_score(y_te, pred)),
                "precision": float(precision_score(y_te, pred, zero_division=0)),
                "recall": float(recall_score(y_te, pred, zero_division=0)),
                "f1": float(f1_score(y_te, pred, zero_division=0)),
            }
        )
    return rows


def train_on_prefix(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Fit the five Exp 1 TDA-default classifiers; score the 15 test snapshots."""
    return fit_default_classifiers(train_df, test_df)


def attach_design_columns(
    metrics_row: Dict[str, Any],
    *,
    dataset_key: str,
    display_name: str,
    folder_name: str,
    protocol_bucket: str,
    points_per_snapshot: int,
    n_snapshots: int,
    repeat: int,
    minority_count: int,
    majority_count: int,
    binding: int,
    default_points_per_snapshot: int,
    n_train_barcode_rows: int,
    n_test_barcode_rows: int,
) -> Dict[str, Any]:
    metrics_row.update(
        {
            "dataset_key": dataset_key,
            "dataset_display": display_name,
            "folder_name": folder_name,
            "protocol": protocol_bucket,
            "protocol_display": PROTOCOLS[protocol_bucket]["display"],
            "points_per_snapshot": int(points_per_snapshot),
            "n_snapshots": int(n_snapshots),
            "repeat": int(repeat),
            "minority_class_count": int(minority_count),
            "majority_class_count": int(majority_count),
            "binding_class_count": int(binding),
            "reuse_ratio": reuse_ratio(
                points_per_snapshot, n_snapshots, minority_count
            ),
            "snapshot_size_percent_of_class": snapshot_size_percent_of_class(
                points_per_snapshot, minority_count
            ),
            "is_default_points_per_snapshot": int(
                points_per_snapshot == default_points_per_snapshot
            ),
            "n_train_barcode_rows": int(n_train_barcode_rows),
            "n_test_barcode_rows": int(n_test_barcode_rows),
            "customer_split_random_state": CUSTOMER_SPLIT_SEED,
            "ci_source": "snapshot_sampling_not_customer_split",
        }
    )
    return metrics_row


def write_repeat_metrics(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    skip_existing: bool = True,
) -> pd.DataFrame:
    path = win_long_path(path)
    if skip_existing and path.exists():
        print(f"[skip] metrics {path.name}")
        return pd.read_csv(path)
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame






def _ci_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset_key",
        "dataset_display",
        "folder_name",
        "protocol",
        "protocol_display",
        "points_per_snapshot",
        "n_snapshots",
        "model",
        "model_display",
        "minority_class_count",
        "majority_class_count",
        "binding_class_count",
        "is_default_points_per_snapshot",
    ]
    rows = []
    grouped = frame.groupby(keys, dropna=False)
    for key, grp in grouped:
        record = dict(zip(keys, key))
        record["n_repeats"] = int(len(grp))
        record["reuse_ratio"] = float(grp["reuse_ratio"].iloc[0]) if len(grp) else float("nan")
        record["snapshot_size_percent_of_class"] = float(
            grp["snapshot_size_percent_of_class"].iloc[0]
        ) if len(grp) else float("nan")
        for metric in ("f1", "accuracy", "precision", "recall"):
            values = grp[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            se = std / math.sqrt(len(values)) if len(values) else float("nan")
            lo, hi = np.percentile(values, [2.5, 97.5]) if len(values) else (mean, mean)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_se"] = float(se)
            record[f"{metric}_ci95_low"] = mean - Z_95 * se
            record[f"{metric}_ci95_high"] = mean + Z_95 * se
            record[f"{metric}_percentile_low"] = float(lo)
            record[f"{metric}_percentile_high"] = float(hi)
        rows.append(record)
    return pd.DataFrame(rows)


def load_all_repeat_metrics() -> pd.DataFrame:
    """Load every per-repeat metrics CSV. Do not trust a partial aggregate."""
    parts = list(
        (REPO_ROOT / "6_Results" / BUCKET / "shared").glob(
            "*/*/repeat_*_pps_*_metrics.csv"
        )
    )
    if not parts:
        path = REPO_ROOT / "6_Results" / BUCKET / "shared" / "all_repeat_metrics.csv"
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if "is_default_points_per_snapshot" in frame.columns:
        frame["is_default_points_per_snapshot"] = (
            frame["is_default_points_per_snapshot"].astype(int)
        )
    return frame


def export_experiment_tables(item: str, frame: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if frame is None:
        frame = load_all_repeat_metrics()
    if frame.empty:
        return frame
    item_folder = ITEM_FOLDERS[item]
    # Three views of one grid, different x-factors:
    #   item 1 — n_snapshots moves; points_per_snapshot locked at the default
    #   item 2 — points_per_snapshot moves; n_snapshots locked at 60
    #   item 4 — full (points_per_snapshot, n_snapshots) family
    if item == "1":
        sliced = frame[frame["is_default_points_per_snapshot"].astype(int) == 1].copy()
    elif item == "2":
        sliced = frame[frame["n_snapshots"] == N_TRAIN_POOL].copy()
    else:
        sliced = frame.copy()
    summary = _ci_summary(sliced)
    for (protocol, folder), grp in sliced.groupby(["protocol", "folder_name"]):
        dest = experiment_results_dir(item_folder, protocol, folder)
        dest.mkdir(parents=True, exist_ok=True)
        grp.to_csv(dest / "repeat_metrics.csv", index=False)
        _ci_summary(grp).to_csv(dest / "summary.csv", index=False)
    root = REPO_ROOT / "6_Results" / BUCKET / item_folder
    root.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(root / "all_repeat_metrics.csv", index=False)
    summary.to_csv(root / "all_summary.csv", index=False)
    return summary


def export_all_experiment_tables(frame: Optional[pd.DataFrame] = None) -> None:
    if frame is None:
        frame = load_all_repeat_metrics()
    if frame.empty:
        return
    agg_path = REPO_ROOT / "6_Results" / BUCKET / "shared" / "all_repeat_metrics.csv"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(agg_path, index=False)
    for item in ("1", "2", "4"):
        export_experiment_tables(item, frame)


def write_master_design_table(
    datasets: Optional[Sequence[str]] = None,
    protocols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    datasets = list(datasets or DATASET_RUN_ORDER)
    protocols = list(protocols or PROTOCOLS.keys())
    rows = []
    for dataset_key in datasets:
        for protocol_bucket in protocols:
            design, _pools = resolve_grid(dataset_key, protocol_bucket)
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_display": design["display_name"],
                    "folder_name": design["folder_name"],
                    "protocol": protocol_bucket,
                    "split_timing": design["split_timing"],
                    "undersample": design["undersample"],
                    "train_minority_count": design["train_minority_count"],
                    "train_majority_count": design["train_majority_count"],
                    "test_minority_count": design["test_minority_count"],
                    "test_majority_count": design["test_majority_count"],
                    "binding_class_count": design["binding_class_count"],
                    "surviving_points_per_snapshot": ",".join(
                        str(v) for v in design["surviving"]
                    ),
                    "dropped_points_per_snapshot": ",".join(
                        str(d["points_per_snapshot"]) for d in design["dropped"]
                    ),
                    "default_points_per_snapshot": design["default_points_per_snapshot"],
                    "clipped": design["clipped"],
                    "clip_note": design["clip_note"] or "",
                    "n_snapshots_grid": "15,30,45,60",
                    "n_test_snapshots": N_TEST_SNAPSHOTS,
                    "n_repeats": N_REPEATS,
                    "reuse_ratio_at_default_and_60": design["reuse_at_default"],
                    "snapshot_size_percent_of_class_at_default": design[
                        "snapshot_size_percent_of_minority_at_default"
                    ],
                }
            )
    frame = pd.DataFrame(rows)
    dest = REPO_ROOT / "6_Results" / BUCKET / "shared" / "dataset_aware_grid.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    save_json(dest.with_suffix(".json"), frame.to_dict(orient="records"))
    return frame


# =============================================================================
# Figures — English labels, methodology notes under every graph
# =============================================================================
ITEM_NOTES = {
    "1": (
        "Item 1 — number of snapshots on the x-axis; each cloud has the dataset-aware "
        "default number of points (largest surviving value in 15/30/45/60 that still fits "
        "this protocol's class pool; 60 on DCCCD). Points per snapshot is held fixed. "
        "This is not item 2, which instead holds 60 snapshots and moves points per snapshot. "
        "Item 3 is this sample-size study (items 1, 2, and 4 together), not a third "
        "independent grid.\n"
        "Headline is F1 because several tables are imbalanced, especially with no "
        "undersampling. Accuracy is plotted as the secondary metric.\n"
        "Nested prefixes: 15 ⊂ 30 ⊂ 45 ⊂ 60 from a shuffled pool of 60 training snapshots. "
        "Ten repeats redraw that pool. The customer train/test split is fixed "
        "(random_state=0). The combined overlay is the mean trend across those 10 repeats "
        "(five models, no error bars). 95% intervals (mean ± 1.96×SE) live on the companion "
        "per-model CI panels as ribbons. A 2.5–97.5 percentile interval is stored in the "
        "summary CSV. This study does not also run five customer splits on the full grid."
    ),
    "2": (
        "Item 2 — points per snapshot on the x-axis; always 60 snapshots. Number of "
        "snapshots is held fixed. This is not item 1, which instead holds points per "
        "snapshot at the dataset-aware default and moves the number of snapshots. "
        "A universal 15/30/45/60 cloud-size grid is not used: PKDD's class pool cannot "
        "host the larger steps that DCCCD can. Candidates with points per snapshot "
        "≥ class count are dropped (no silent clipping). Item 3 is this sample-size "
        "study (items 1, 2, and 4 together), not a third independent grid.\n"
        "Headline is F1; accuracy is always shown. Nested prefixes and 10 snapshot-draw "
        "repeats match item 1. Customer split is fixed (random_state=0). The combined "
        "overlay is the mean trend across 10 repeats (five models, no error bars). "
        "95% intervals live on the companion per-model CI panels as ribbons."
    ),
    "4": (
        "Item 4 — number of snapshots on the x-axis; one curve per surviving "
        "points-per-snapshot value (families of cloud size). SVM and Logistic Regression "
        "are the focus overlay; KNN, XGBoost, and Random Forest are the same colour and "
        "marker language, not washed out. Item 3 is this sample-size study (items 1, 2, "
        "and 4 together), not a third independent grid.\n"
        "Nested prefixes: 15 ⊂ 30 ⊂ 45 ⊂ 60 from one pool of 60 training snapshots per "
        "repeat. Ten repeats. Customer split fixed (random_state=0). Combined overlays "
        "are mean trends only (no error bars). Companion CI panels use a single ribbon "
        "per (model, points-per-snapshot) cell (mean ± 1.96×SE across snapshot draws). "
        "This CI is snapshot-sampling uncertainty, not customer-split uncertainty."
    ),
}


def _wrap_note(text: str, width: int = 148) -> str:
    """Pre-wrap methodology notes. matplotlib wrap=True is extremely slow with tight bbox."""
    return "\n".join(textwrap.fill(part, width=width) for part in text.split("\n"))


def _footnote(ax, text: str) -> None:
    ax.figure.text(
        0.02,
        0.01,
        _wrap_note(text),
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#2d3748",
    )


def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.06)


def _model_legend_handle(model: str) -> Line2D:
    """Colour + marker only for a stable legend."""
    return Line2D(
        [0],
        [0],
        color=MODEL_COLORS[model],
        linewidth=MODEL_LINEWIDTH[model],
        linestyle="-",
        marker=MODEL_MARKERS[model],
        markersize=MODEL_MARKERSIZE[model],
        markerfacecolor=MODEL_COLORS[model],
        markeredgecolor=MODEL_COLORS[model],
        markeredgewidth=0.6,
        alpha=MODEL_LINE_ALPHA,
        label=CLASSIFIER_DISPLAY[model],
    )


def _legend_below(ax, ncol: int = 5, bbox_y: float = -0.18) -> None:
    handles = [_model_legend_handle(model) for model in CLASSIFIER_ORDER]
    ax.legend(
        handles=handles,
        labels=[CLASSIFIER_DISPLAY[m] for m in CLASSIFIER_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, bbox_y),
        ncol=ncol,
        frameon=False,
        fontsize=9,
        handlelength=2.6,
        columnspacing=1.6,
        borderaxespad=0.0,
    )


def _cloud_size_style(pps_values: Sequence[float]) -> Dict[int, Dict[str, Any]]:
    ordered = [int(v) for v in pps_values]
    styles: Dict[int, Dict[str, Any]] = {}
    for i, pps in enumerate(ordered):
        styles[pps] = {
            "color": CLOUD_SIZE_COLOR_CYCLE[i % len(CLOUD_SIZE_COLOR_CYCLE)],
            "marker": CLOUD_SIZE_MARKER_CYCLE[i % len(CLOUD_SIZE_MARKER_CYCLE)],
        }
    return styles


def _cloud_size_legend_handles(pps_values: Sequence[float]) -> List[Line2D]:
    styles = _cloud_size_style(pps_values)
    handles = []
    for pps in pps_values:
        spec = styles[int(pps)]
        handles.append(
            Line2D(
                [0],
                [0],
                color=spec["color"],
                linewidth=2.0,
                linestyle="-",
                marker=spec["marker"],
                markersize=7.0,
                markerfacecolor=spec["color"],
                markeredgecolor=spec["color"],
                label=f"{int(pps)} points per snapshot",
            )
        )
    return handles


def _plot_mean_trend(
    ax,
    summary: pd.DataFrame,
    x_col: str,
    metric: str,
    model: str,
    *,
    label: Optional[str] = None,
) -> None:
    """Mean trend line only. Combined overlays never draw error bars."""
    sub = summary[summary["model"] == model].sort_values(x_col)
    if sub.empty:
        return
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[f"{metric}_mean"].to_numpy(dtype=float)
    highlight = model in HIGHLIGHT_MODELS
    ax.plot(
        x,
        y,
        color=MODEL_COLORS[model],
        linewidth=MODEL_LINEWIDTH[model],
        linestyle="-",
        marker=MODEL_MARKERS[model],
        markersize=MODEL_MARKERSIZE[model],
        markerfacecolor=MODEL_COLORS[model],
        markeredgecolor=MODEL_COLORS[model],
        markeredgewidth=0.6,
        alpha=MODEL_LINE_ALPHA,
        label=label or CLASSIFIER_DISPLAY[model],
        zorder=3 if highlight else 2,
    )


def _plot_mean_ribbon(
    ax,
    summary: pd.DataFrame,
    x_col: str,
    metric: str,
    model: str,
    *,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    lw: Optional[float] = None,
    label: Optional[str] = None,
) -> None:
    """Single-series ribbon. Use only when one CI is drawn on the axes."""
    sub = summary[summary["model"] == model].sort_values(x_col)
    if sub.empty:
        return
    x = sub[x_col].to_numpy(dtype=float)
    y = sub[f"{metric}_mean"].to_numpy(dtype=float)
    lo = sub[f"{metric}_ci95_low"].to_numpy(dtype=float)
    hi = sub[f"{metric}_ci95_high"].to_numpy(dtype=float)
    color = color or MODEL_COLORS[model]
    marker = marker or MODEL_MARKERS[model]
    lw = MODEL_LINEWIDTH[model] if lw is None else float(lw)
    ax.fill_between(x, lo, hi, color=color, alpha=CI_RIBBON_ALPHA, linewidth=0, zorder=1)
    ax.plot(
        x,
        y,
        color=color,
        lw=lw,
        alpha=MODEL_LINE_ALPHA,
        marker=marker,
        markersize=MODEL_MARKERSIZE[model],
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.6,
        label=label or CLASSIFIER_DISPLAY[model],
        zorder=3,
    )


def _plot_pps_trend(
    ax,
    frame: pd.DataFrame,
    metric: str,
    model: str,
    pps: int,
    *,
    color: str,
    marker: str,
    label: Optional[str] = None,
) -> None:
    """Mean trend for one cloud size. Combined overlays never draw error bars."""
    sub = frame[
        (frame["model"] == model) & (frame["points_per_snapshot"] == pps)
    ].sort_values("n_snapshots")
    if sub.empty:
        return
    x = sub["n_snapshots"].to_numpy(dtype=float)
    y = sub[f"{metric}_mean"].to_numpy(dtype=float)
    highlight = model in HIGHLIGHT_MODELS
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.4 if highlight else 1.8,
        linestyle="-",
        marker=marker,
        markersize=8.0 if highlight else 6.6,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.6,
        alpha=MODEL_LINE_ALPHA,
        label=label,
        zorder=3 if highlight else 2,
    )


def _as_axes_grid(axes, n_rows: int, n_cols: int) -> np.ndarray:
    grid = np.array(axes, dtype=object)
    if n_rows == 1 and n_cols == 1:
        return np.array([[grid]], dtype=object)
    if n_rows == 1:
        return grid.reshape(1, n_cols)
    if n_cols == 1:
        return grid.reshape(n_rows, 1)
    return grid.reshape(n_rows, n_cols)


def _plot_ci_panels(
    summary: pd.DataFrame,
    viz: Path,
    *,
    x_col: str,
    metric: str,
    path: Path,
    xlabel: str,
    title: str,
    note: str,
    xticks: Optional[Sequence[float]] = None,
) -> Path:
    fig, axes = plt.subplots(1, len(CLASSIFIER_ORDER), figsize=(15.4, 4.9), sharey=True)
    fig.subplots_adjust(bottom=0.36, wspace=0.16, top=0.80, left=0.06, right=0.99)
    for i, model in enumerate(CLASSIFIER_ORDER):
        ax = axes[i]
        _plot_mean_ribbon(ax, summary, x_col, metric, model)
        _style_axes(
            ax,
            xlabel=xlabel,
            ylabel=METRIC_DISPLAY[metric] if i == 0 else "",
            title=CLASSIFIER_DISPLAY[model],
        )
        if xticks is not None:
            ax.set_xticks(list(xticks))
        else:
            vals = sorted(summary[x_col].unique())
            ax.set_xticks(vals)
    fig.suptitle(title)
    fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.2, color="#2d3748")
    written = _sample_size_save_figure(fig, path, dpi=160)
    plt.close(fig)
    return written


def _require_summary(item_folder: str) -> pd.DataFrame:
    path = REPO_ROOT / "6_Results" / BUCKET / item_folder / "all_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"results not generated yet: expected {path}. "
            "Run 5_Experiments/Snapshot_Sample_Size/run_shared.py first."
        )
    return pd.read_csv(path)


def visualize_item(item: str) -> List[Path]:
    item_folder = ITEM_FOLDERS[item]
    summary = _require_summary(item_folder)
    viz = visualizations_dir(item_folder)
    win_long_path(viz).mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    note = ITEM_NOTES[item]
    if item == "1":
        written.extend(_plot_item1(summary, viz, note))
    elif item == "2":
        written.extend(_plot_item2(summary, viz, note))
    else:
        written.extend(_plot_item4(summary, viz, note))
    return written


def _plot_item1(summary: pd.DataFrame, viz: Path, note: str) -> List[Path]:
    written = []
    for (folder, protocol), grp in summary.groupby(["folder_name", "protocol"]):
        for metric in ("f1", "accuracy"):
            fig, ax = plt.subplots(figsize=(10.2, 6.8))
            fig.subplots_adjust(bottom=0.38, top=0.88)
            for model in CLASSIFIER_ORDER:
                _plot_mean_trend(ax, grp, "n_snapshots", metric, model)
            pps = int(grp["points_per_snapshot"].iloc[0])
            _style_axes(
                ax,
                xlabel="Number of snapshots (training)",
                ylabel=METRIC_DISPLAY[metric],
                title=(
                    f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
                    f"{METRIC_DISPLAY[metric]} vs number of snapshots\n"
                    f"Number of snapshots on the x-axis; each cloud has {pps} points"
                ),
            )
            _legend_below(ax, ncol=5, bbox_y=-0.20)
            ax.set_xticks(list(N_SNAPSHOTS_GRID))
            _footnote(ax, note)
            path = viz / f"{folder}_{protocol}_{metric}_by_n_snapshots.png"
            written.append(_sample_size_save_figure(fig, path, dpi=160))
            plt.close(fig)
            written.append(
                _plot_ci_panels(
                    grp,
                    viz,
                    x_col="n_snapshots",
                    metric=metric,
                    path=viz / f"{folder}_{protocol}_{metric}_by_n_snapshots_ci_panels.png",
                    xlabel="Number of snapshots (training)",
                    title=(
                        f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
                        f"{METRIC_DISPLAY[metric]} vs number of snapshots, one model per panel "
                        f"(95% ribbon). Number of snapshots on the x-axis; each cloud has {pps} points"
                    ),
                    note=note,
                    xticks=list(N_SNAPSHOTS_GRID),
                )
            )
    written.extend(_plot_cross_dataset_facet(summary, viz, "1", "n_snapshots", note))
    return written


def _plot_item2(summary: pd.DataFrame, viz: Path, note: str) -> List[Path]:
    written = []
    for (folder, protocol), grp in summary.groupby(["folder_name", "protocol"]):
        xticks = sorted(int(v) for v in grp["points_per_snapshot"].unique())
        dropped = ""
        if int(grp["binding_class_count"].iloc[0]) <= 60:
            dropped = (
                f" Binding class count on this arm is "
                f"{int(grp['binding_class_count'].iloc[0])}, so any candidate "
                f"≥ that count was dropped."
            )
        panel_note = note + dropped
        for metric in ("f1", "accuracy"):
            fig, ax = plt.subplots(figsize=(10.2, 7.0))
            fig.subplots_adjust(bottom=0.40, top=0.88)
            for model in CLASSIFIER_ORDER:
                _plot_mean_trend(ax, grp, "points_per_snapshot", metric, model)
            _style_axes(
                ax,
                xlabel="Points per snapshot",
                ylabel=METRIC_DISPLAY[metric],
                title=(
                    f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
                    f"{METRIC_DISPLAY[metric]} vs points per snapshot\n"
                    "Points per snapshot on the x-axis; always 60 snapshots"
                ),
            )
            _legend_below(ax, ncol=5, bbox_y=-0.22)
            ax.set_xticks(xticks)
            _footnote(ax, panel_note)
            path = viz / f"{folder}_{protocol}_{metric}_by_points_per_snapshot.png"
            written.append(_sample_size_save_figure(fig, path, dpi=160))
            plt.close(fig)
            written.append(
                _plot_ci_panels(
                    grp,
                    viz,
                    x_col="points_per_snapshot",
                    metric=metric,
                    path=viz
                    / f"{folder}_{protocol}_{metric}_by_points_per_snapshot_ci_panels.png",
                    xlabel="Points per snapshot",
                    title=(
                        f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
                        f"{METRIC_DISPLAY[metric]} vs points per snapshot, one model per panel "
                        "(95% ribbon). Points per snapshot on the x-axis; always 60 snapshots"
                    ),
                    note=panel_note,
                    xticks=xticks,
                )
            )
    written.extend(
        _plot_cross_dataset_facet(summary, viz, "2", "points_per_snapshot", note)
    )
    return written


def _plot_item4_cloud_size_ci_panels(
    grp: pd.DataFrame,
    viz: Path,
    folder: str,
    protocol: str,
    metric: str,
    note: str,
) -> Path:
    pps_values = [int(v) for v in sorted(grp["points_per_snapshot"].unique())]
    n_pps = max(1, len(pps_values))
    n_models = len(CLASSIFIER_ORDER)
    fig, axes = plt.subplots(
        n_models,
        n_pps,
        figsize=(3.35 * n_pps, 2.45 * n_models),
        sharex=True,
        sharey="row",
    )
    fig.subplots_adjust(bottom=0.14, hspace=0.38, wspace=0.16, top=0.90, left=0.08, right=0.99)
    grid = _as_axes_grid(axes, n_models, n_pps)
    styles = _cloud_size_style(pps_values)
    for row_i, model in enumerate(CLASSIFIER_ORDER):
        for col_i, pps in enumerate(pps_values):
            ax = grid[row_i, col_i]
            sub = grp[grp["points_per_snapshot"] == pps]
            spec = styles[int(pps)]
            _plot_mean_ribbon(
                ax,
                sub,
                "n_snapshots",
                metric,
                model,
                color=spec["color"],
                marker=spec["marker"],
            )
            title = ""
            if row_i == 0:
                title = f"{int(pps)} points per snapshot"
            ylabel = METRIC_DISPLAY[metric] if col_i == 0 else ""
            _style_axes(
                ax,
                xlabel="Number of snapshots" if row_i == n_models - 1 else "",
                ylabel=ylabel,
                title=title,
            )
            if col_i == 0:
                ax.text(
                    -0.28,
                    0.5,
                    CLASSIFIER_DISPLAY[model],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=9,
                    color=MODEL_COLORS[model],
                    fontweight="bold" if model in HIGHLIGHT_MODELS else "normal",
                )
            ax.set_xticks(list(N_SNAPSHOTS_GRID))
    fig.suptitle(
        f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
        f"{METRIC_DISPLAY[metric]} families of cloud size, one 95% ribbon per panel"
    )
    fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.0, color="#2d3748")
    path = viz / f"{folder}_{protocol}_{metric}_cloud_size_ci_panels.png"
    written = _sample_size_save_figure(fig, path, dpi=150)
    plt.close(fig)
    return written


def _plot_item4(summary: pd.DataFrame, viz: Path, note: str) -> List[Path]:
    written = []
    for (folder, protocol), grp in summary.groupby(["folder_name", "protocol"]):
        pps_values = [int(v) for v in sorted(grp["points_per_snapshot"].unique())]
        styles = _cloud_size_style(pps_values)
        # Focus overlay: SVM + Logistic × F1 + accuracy, mean trends only.
        fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6), sharex=True)
        fig.subplots_adjust(bottom=0.30, hspace=0.30, wspace=0.22, top=0.86)
        for row_i, model in enumerate(("svm", "logistic")):
            for col_i, metric in enumerate(("f1", "accuracy")):
                ax = axes[row_i][col_i]
                for pps in pps_values:
                    spec = styles[int(pps)]
                    _plot_pps_trend(
                        ax,
                        grp,
                        metric,
                        model,
                        pps,
                        color=spec["color"],
                        marker=spec["marker"],
                        label=f"{int(pps)} points per snapshot",
                    )
                _style_axes(
                    ax,
                    xlabel="Number of snapshots (training)",
                    ylabel=METRIC_DISPLAY[metric],
                    title=f"{CLASSIFIER_DISPLAY[model]} — {METRIC_DISPLAY[metric]}",
                )
                ax.set_xticks(list(N_SNAPSHOTS_GRID))
        fig.legend(
            handles=_cloud_size_legend_handles(pps_values),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(4, len(pps_values)),
            frameon=False,
        )
        fig.suptitle(
            f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
            "Families of cloud size (item 4). Number of snapshots on the x-axis; "
            "one curve per cloud size. SVM and Logistic Regression are the focus. "
            "Mean trend across 10 repeats; 95% intervals are on the CI panels.",
            y=1.02,
        )
        fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.2, color="#2d3748")
        path = viz / f"{folder}_{protocol}_svm_logistic_cloud_size_families.png"
        written.append(_sample_size_save_figure(fig, path, dpi=160))
        plt.close(fig)

        # Overlay small multiples: 5 classifiers × 2 metrics, mean trends only.
        fig, axes = plt.subplots(5, 2, figsize=(10.8, 13.8), sharex=True)
        fig.subplots_adjust(bottom=0.16, hspace=0.40, wspace=0.22, top=0.91)
        for row_i, model in enumerate(CLASSIFIER_ORDER):
            for col_i, metric in enumerate(("f1", "accuracy")):
                ax = axes[row_i][col_i]
                for pps in pps_values:
                    spec = styles[int(pps)]
                    _plot_pps_trend(
                        ax,
                        grp,
                        metric,
                        model,
                        pps,
                        color=spec["color"],
                        marker=spec["marker"],
                        label=f"{int(pps)} points per snapshot"
                        if row_i == 0 and col_i == 0
                        else None,
                    )
                _style_axes(
                    ax,
                    xlabel="Number of snapshots",
                    ylabel=METRIC_DISPLAY[metric],
                    title=f"{CLASSIFIER_DISPLAY[model]} — {METRIC_DISPLAY[metric]}",
                )
                ax.set_xticks(list(N_SNAPSHOTS_GRID))
        fig.legend(
            handles=_cloud_size_legend_handles(pps_values),
            loc="upper center",
            ncol=min(4, len(pps_values)),
            frameon=False,
        )
        fig.suptitle(
            f"{grp['dataset_display'].iloc[0]} — {grp['protocol_display'].iloc[0]}\n"
            "All five classifiers (SVM / Logistic thicker; KNN, XGBoost, and "
            "Random Forest at full saturation). Number of snapshots on the x-axis; "
            "one curve per cloud size. Mean trend across 10 repeats; 95% intervals "
            "are on the CI panels."
        )
        fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.0, color="#2d3748")
        path = viz / f"{folder}_{protocol}_all_classifiers_small_multiples.png"
        written.append(_sample_size_save_figure(fig, path, dpi=150))
        plt.close(fig)

        for metric in ("f1", "accuracy"):
            written.append(
                _plot_item4_cloud_size_ci_panels(grp, viz, folder, protocol, metric, note)
            )
    return written


def _plot_cross_dataset_facet(
    summary: pd.DataFrame,
    viz: Path,
    item: str,
    x_col: str,
    note: str,
) -> List[Path]:
    written = []
    if x_col == "n_snapshots":
        xlabel = "Number of snapshots (training)"
        axis_note = (
            "Number of snapshots on the x-axis; each cloud uses the dataset-aware "
            "default point count (held fixed)"
        )
    else:
        xlabel = "Points per snapshot"
        axis_note = "Points per snapshot on the x-axis; always 60 snapshots"
    for protocol, proto_grp in summary.groupby("protocol"):
        datasets = list(proto_grp["folder_name"].unique())
        n = len(datasets)
        if n == 0:
            continue
        fig, axes = plt.subplots(n, 2, figsize=(11.2, 3.2 * n), sharex=True)
        if n == 1:
            axes = np.array([axes])
        fig.subplots_adjust(bottom=0.20, hspace=0.48, wspace=0.22, top=0.90)
        for i, folder in enumerate(datasets):
            sub = proto_grp[proto_grp["folder_name"] == folder]
            for j, metric in enumerate(("f1", "accuracy")):
                ax = axes[i][j]
                for model in CLASSIFIER_ORDER:
                    _plot_mean_trend(ax, sub, x_col, metric, model)
                _style_axes(
                    ax,
                    xlabel=xlabel,
                    ylabel=METRIC_DISPLAY[metric],
                    title=f"{sub['dataset_display'].iloc[0]} — {METRIC_DISPLAY[metric]}",
                )
                if x_col == "n_snapshots":
                    ax.set_xticks(list(N_SNAPSHOTS_GRID))
                else:
                    ax.set_xticks(sorted(sub[x_col].unique()))
        fig.legend(
            handles=[_model_legend_handle(m) for m in CLASSIFIER_ORDER],
            loc="upper center",
            ncol=5,
            frameon=False,
        )
        fig.suptitle(
            f"{PROTOCOLS[protocol]['display']} — item {item} across datasets\n{axis_note}"
        )
        fig.text(0.02, 0.01, _wrap_note(note), ha="left", va="bottom", fontsize=7.0, color="#2d3748")
        path = viz / f"cross_dataset_{protocol}_item{item}.png"
        written.append(_sample_size_save_figure(fig, path, dpi=150))
        plt.close(fig)
    return written




# =============================================================================
# REVISED SNAPSHOT PROTOCOL HELPERS (Experiment 9)
# =============================================================================

# -----------------------------------------------------------------------------
# Canonical sweep grids
# -----------------------------------------------------------------------------
DEFAULT_TRAIN_L = 60
DEFAULT_TEST_L = 15
ZANIAR_TRAIN_L = (60, 80, 100)  # 3 points in 60–100
ZANIAR_TEST_L = (15, 22, 30)  # 3 points in 15–30
DCCCD_FULL_L = (60, 75, 90)  # 3 points in 60–90 for the bigger dataset
TARGET_REUSE = 1.0
TARGET_T_OVER_CLASS = 0.20


# =============================================================================
# Concern A — email formula:  l ~ (t / log t)^{2/b}
# =============================================================================


def formula_t_candidates_for_target_l(
    target_l: int,
    b: float,
    t_min: int = 10,
    t_max: int = 200,
) -> List[Dict[str, float]]:
    """Find integer t where formula_l_from_t_b(t,b) is closest to target_l."""
    rows = []
    for t in range(t_min, t_max + 1):
        l_hat = formula_l_from_t_b(t, b)
        rows.append(
            {
                "t": t,
                "b": float(b),
                "l_formula": l_hat,
                "abs_diff_to_target_l": abs(l_hat - target_l),
                "target_l": target_l,
            }
        )
    rows.sort(key=lambda r: r["abs_diff_to_target_l"])
    return rows


# =============================================================================
# Concern B — reuse-ratio / sampling constraints
# =============================================================================


def max_t_for_reuse(
    n_class: int,
    l: int,
    max_reuse: float = TARGET_REUSE,
) -> int:
    """Largest integer t with (t*l)/n_class <= max_reuse and t <= n_class."""
    if n_class < 2 or l < 1:
        return 0
    return int(max(2, min(n_class, math.floor(max_reuse * n_class / l))))


def max_l_for_reuse(
    n_class: int,
    t: int,
    max_reuse: float = TARGET_REUSE,
) -> int:
    """Largest integer l with (t*l)/n_class <= max_reuse."""
    if n_class < 2 or t < 1:
        return 0
    return int(max(1, math.floor(max_reuse * n_class / t)))


def t_over_class(t: int, n_class: int) -> float:
    if n_class <= 0:
        return float("nan")
    return float(t) / float(n_class)


def audit_reuse_constraints(
    n_pos: int,
    n_neg: int,
    t: int,
    l: int,
    max_reuse: float = TARGET_REUSE,
    max_t_frac: float = TARGET_T_OVER_CLASS,
) -> Dict[str, Any]:
    """
    Concern B audit. Independent of the email formula.

    Checks:
      1) t / n_c < max_t_frac for each class c
      2) (t * l) / n_c ≲ max_reuse for each class c
    The binding class is the minority (smaller n_c).
    """
    n_min = min(n_pos, n_neg)
    n_maj = max(n_pos, n_neg)
    r_pos = reuse_ratio(t, l, n_pos)
    r_neg = reuse_ratio(t, l, n_neg)
    return {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n_minority": int(n_min),
        "n_majority": int(n_maj),
        "t": int(t),
        "l": int(l),
        "t_over_n_pos": t_over_class(t, n_pos),
        "t_over_n_neg": t_over_class(t, n_neg),
        "reuse_pos": r_pos,
        "reuse_neg": r_neg,
        "reuse_binding": max(r_pos, r_neg),
        "ok_t_fraction": (t_over_class(t, n_min) < max_t_frac),
        "ok_reuse": (r_pos <= max_reuse and r_neg <= max_reuse),
        "max_t_reuse_ok": max_t_for_reuse(n_min, l, max_reuse),
        "max_l_reuse_ok": max_l_for_reuse(n_min, t, max_reuse),
        "max_reuse_target": max_reuse,
        "max_t_frac_target": max_t_frac,
    }


def recommend_t_l_separated(
    n_pos: int,
    n_neg: int,
    b: float,
    train_l_target: int = DEFAULT_TRAIN_L,
    test_l_target: int = DEFAULT_TEST_L,
    t_candidates: Optional[Sequence[int]] = None,
    t_min_practical: int = 10,
) -> Dict[str, Any]:
    """
    Produce recommendations while keeping Concern A and Concern B separate.

    When minority pools are too small for (t_min_practical, train_l=60) under
    reuse ≤ 1, we *adapt train_l / test_l downward* and document that Concern B
    overrode the meeting default (formula Concern A still reported separately).
    """
    n_min = min(n_pos, n_neg)
    adapted_train_l = int(train_l_target)
    adapted_test_l = int(test_l_target)
    adaptation_notes = []

    # Ensure a practical t exists under reuse at the requested l; else lower l.
    t_at_train = max_t_for_reuse(n_min, adapted_train_l, TARGET_REUSE)
    if t_at_train < t_min_practical:
        # Need l <= n_min / t_min_practical
        adapted_train_l = max(1, int(math.floor(TARGET_REUSE * n_min / t_min_practical)))
        adapted_train_l = min(adapted_train_l, train_l_target)
        adaptation_notes.append(
            f"train_l reduced {train_l_target}→{adapted_train_l} so t≥{t_min_practical} "
            f"keeps reuse≤{TARGET_REUSE} on n_min={n_min}"
        )
        t_at_train = max_t_for_reuse(n_min, adapted_train_l, TARGET_REUSE)

    t_at_test = max_t_for_reuse(n_min, adapted_test_l, TARGET_REUSE)
    if t_at_test < t_min_practical:
        adapted_test_l = max(1, int(math.floor(TARGET_REUSE * n_min / t_min_practical)))
        adapted_test_l = min(adapted_test_l, test_l_target)
        adaptation_notes.append(
            f"test_l reduced {test_l_target}→{adapted_test_l} so t≥{t_min_practical} "
            f"keeps reuse≤{TARGET_REUSE} on n_min={n_min}"
        )
        t_at_test = max_t_for_reuse(n_min, adapted_test_l, TARGET_REUSE)

    # Absolute floor: need t>=3 for formula and PH
    t_hi = max(3, min(t_at_train, n_min))
    if t_candidates is None:
        mid = max(3, t_hi // 2)
        lo = max(3, t_hi // 4)
        extras = [t for t in (10, 20, 40, 60, 80) if t <= t_hi]
        t_candidates = sorted(set([lo, mid, t_hi, *extras]))
        t_candidates = [t for t in t_candidates if 3 <= t <= n_min]

    formula_rows = []
    for t in t_candidates:
        if not np.isfinite(b) or b <= 0:
            l_f = float("nan")
        else:
            l_f = formula_l_from_t_b(t, b)
        formula_rows.append(
            {
                "t": t,
                "b": float(b) if np.isfinite(b) else None,
                "l_formula": l_f,
                "vs_train_default_60": (l_f - train_l_target) if np.isfinite(l_f) else None,
                "vs_test_default_15": (l_f - test_l_target) if np.isfinite(l_f) else None,
            }
        )

    reuse_rows = []
    for t in t_candidates:
        reuse_rows.append(
            {
                "t": t,
                **audit_reuse_constraints(n_pos, n_neg, t, adapted_train_l),
                "split": "train",
            }
        )
        reuse_rows.append(
            {
                "t": t,
                **audit_reuse_constraints(n_pos, n_neg, t, adapted_test_l),
                "split": "test",
            }
        )

    feasible_train = [
        r for r in reuse_rows if r["split"] == "train" and r["ok_reuse"] and r["ok_t_fraction"]
    ]
    if feasible_train:
        chosen_t = max(r["t"] for r in feasible_train)
    else:
        chosen_t = max(3, t_hi)

    chosen_audit_train = audit_reuse_constraints(n_pos, n_neg, chosen_t, adapted_train_l)
    chosen_audit_test = audit_reuse_constraints(n_pos, n_neg, chosen_t, adapted_test_l)
    if np.isfinite(b) and b > 0 and chosen_t >= 3:
        l_formula_at_chosen = formula_l_from_t_b(chosen_t, b)
    else:
        l_formula_at_chosen = float("nan")

    return {
        "concern_A_formula": {
            "definition": "l ≈ (t / log t)^{2/b}",
            "rows": formula_rows,
            "at_chosen_t": {
                "t": chosen_t,
                "b": float(b) if np.isfinite(b) else None,
                "l_formula": l_formula_at_chosen,
                "interpretation": (
                    "Formula-suggested snapshot count at chosen t. "
                    "Compare to meeting defaults (train 60 / test 15) separately "
                    "from reuse feasibility."
                ),
            },
        },
        "concern_B_reuse": {
            "definition": "R=(t*l)/n_class ≲ 1 and t/n_class < 0.20",
            "rows": reuse_rows,
            "max_t_at_train_l": max_t_for_reuse(n_min, adapted_train_l),
            "max_t_at_test_l": max_t_for_reuse(n_min, adapted_test_l),
            "adapted_train_l": adapted_train_l,
            "adapted_test_l": adapted_test_l,
            "adaptation_notes": adaptation_notes,
        },
        "chosen_joint": {
            "t": int(chosen_t),
            "train_l": int(adapted_train_l),
            "test_l": int(adapted_test_l),
            "meeting_train_l_requested": int(train_l_target),
            "meeting_test_l_requested": int(test_l_target),
            "same_t_train_test": True,
            "no_undersampling": True,
            "train_reuse_audit": chosen_audit_train,
            "test_reuse_audit": chosen_audit_test,
            "formula_l_at_chosen_t": l_formula_at_chosen,
            "why": (
                "Pick the largest reuse-safe t at the (possibly adapted) train_l "
                "(binding class = minority). Keep the SAME t for test. "
                "Report formula l separately; do not override meeting 60/15 "
                "with the formula alone — only Concern B (reuse) may reduce l."
            ),
            "adaptation_notes": adaptation_notes,
        },
        "per_class_minima": per_class_safe_minima(n_pos, n_neg, t=chosen_t),
    }


def choose_joint_t_train_test_l(
    train_pos: int,
    train_neg: int,
    test_pos: int,
    test_neg: int,
    target_train_l: int = DEFAULT_TRAIN_L,
    target_test_l: int = DEFAULT_TEST_L,
    t_max_cap: int = 120,
    min_train_l: int = 5,
    min_test_l: int = 3,
) -> Dict[str, Any]:
    """
    Jointly choose a single t and (train_l, test_l) under Concern B.

    Preference order:
      1) reuse ≤ 1 on BOTH train and test minority pools
      2) train_l as close as possible to 60, test_l as close as possible to 15
      3) larger t (richer topology) among ties
      4) if impossible, relax test reuse to ≤ 2.0 (documented), never train reuse > 1
    """
    n_tr = min(train_pos, train_neg)
    n_te = min(test_pos, test_neg)
    t_hi = int(min(n_tr, n_te, t_max_cap))
    candidates = []
    for t in range(t_hi, 4, -1):
        max_tr_l = max_l_for_reuse(n_tr, t, 1.0)
        max_te_l = max_l_for_reuse(n_te, t, 1.0)
        train_l = min(target_train_l, max_tr_l)
        test_l = min(target_test_l, max_te_l)
        if train_l >= min_train_l and test_l >= min_test_l:
            score = (train_l / target_train_l) + (test_l / target_test_l) + 0.001 * t
            candidates.append(
                {
                    "t": t,
                    "train_l": train_l,
                    "test_l": test_l,
                    "score": score,
                    "test_reuse_limit": 1.0,
                    "relaxed_test_reuse": False,
                }
            )
    if not candidates:
        # Relax test reuse to 2.0; keep train reuse ≤ 1
        for t in range(t_hi, 4, -1):
            max_tr_l = max_l_for_reuse(n_tr, t, 1.0)
            max_te_l = max_l_for_reuse(n_te, t, 2.0)
            train_l = min(target_train_l, max_tr_l)
            test_l = min(target_test_l, max_te_l)
            if train_l >= min_train_l and test_l >= min_test_l:
                score = (train_l / target_train_l) + (test_l / target_test_l) + 0.001 * t - 0.5
                candidates.append(
                    {
                        "t": t,
                        "train_l": train_l,
                        "test_l": test_l,
                        "score": score,
                        "test_reuse_limit": 2.0,
                        "relaxed_test_reuse": True,
                    }
                )
    if not candidates:
        # Last resort: smallest workable PH setting
        t = max(5, min(t_hi, 10))
        return {
            "t": t,
            "train_l": max(1, max_l_for_reuse(n_tr, t, 1.0)),
            "test_l": max(1, max_l_for_reuse(n_te, t, 2.0)),
            "score": 0.0,
            "test_reuse_limit": 2.0,
            "relaxed_test_reuse": True,
            "fallback": True,
            "n_train_min": n_tr,
            "n_test_min": n_te,
        }
    best = max(candidates, key=lambda r: r["score"])
    best["n_train_min"] = n_tr
    best["n_test_min"] = n_te
    best["fallback"] = False
    return best


def per_class_safe_minima(
    n_pos: int,
    n_neg: int,
    t: int,
    max_reuse: float = TARGET_REUSE,
) -> Dict[str, Any]:
    """
    Safe upper bounds on l (and implied minima discussion) per class at fixed t.
    Conservative experimental l must not exceed the minority-class bound.
    """
    return {
        "t": int(t),
        "pos": {
            "n": int(n_pos),
            "max_l_reuse_le_1": max_l_for_reuse(n_pos, t, max_reuse),
            "reuse_at_l60": reuse_ratio(t, 60, n_pos),
            "reuse_at_l100": reuse_ratio(t, 100, n_pos),
        },
        "neg": {
            "n": int(n_neg),
            "max_l_reuse_le_1": max_l_for_reuse(n_neg, t, max_reuse),
            "reuse_at_l60": reuse_ratio(t, 60, n_neg),
            "reuse_at_l100": reuse_ratio(t, 100, n_neg),
        },
        "conservative_max_l": int(
            min(max_l_for_reuse(n_pos, t, max_reuse), max_l_for_reuse(n_neg, t, max_reuse))
        ),
    }


# =============================================================================
# Intrinsic dimension (skdim + fallbacks)
# =============================================================================
try:
    import skdim
    HAS_SKDIM = True
except Exception:
    HAS_SKDIM = False


def estimate_intrinsic_dimensions(
    X: np.ndarray,
    n_samples: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Estimate intrinsic dimension b with multiple estimators.

    Primary package: scikit-dimension (skdim) — TwoNN, MLE (Levina–Bickel),
    DANCo (when computationally feasible), lPCA.
    """
    rng = check_random_state(random_state)
    X = np.asarray(X, dtype=float)
    if n_samples is not None and len(X) > n_samples:
        idx = rng.choice(len(X), size=n_samples, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    out: Dict[str, Any] = {
        "n_points_used": int(len(Xs)),
        "n_features": int(Xs.shape[1]),
        "package": "scikit-dimension" if HAS_SKDIM else "fallback_local",
        "estimators": {},
    }

    if HAS_SKDIM:
        try:
            twonn = skdim.id.TwoNN().fit(Xs)
            out["estimators"]["TwoNN"] = float(twonn.dimension_)
        except Exception as exc:  # pragma: no cover
            out["estimators"]["TwoNN"] = f"error:{exc}"
        try:
            mle = skdim.id.MLE(K=20).fit(Xs)
            out["estimators"]["MLE_LevinaBickel"] = float(mle.dimension_)
        except Exception as exc:  # pragma: no cover
            out["estimators"]["MLE_LevinaBickel"] = f"error:{exc}"
        try:
            lpca = skdim.id.lPCA().fit(Xs)
            out["estimators"]["lPCA"] = float(lpca.dimension_)
        except Exception as exc:  # pragma: no cover
            out["estimators"]["lPCA"] = f"error:{exc}"
        if hasattr(skdim.id, "MiND_ML"):
            try:
                mind = skdim.id.MiND_ML().fit(Xs)
                out["estimators"]["MiND_ML"] = float(mind.dimension_)
            except Exception as exc:  # pragma: no cover
                out["estimators"]["MiND_ML"] = f"error:{exc}"
        # DANCo is slower; only on modest samples
        if len(Xs) <= 800:
            try:
                danco = skdim.id.DANCo().fit(Xs)
                out["estimators"]["DANCo"] = float(danco.dimension_)
            except Exception as exc:  # pragma: no cover
                out["estimators"]["DANCo"] = f"error:{exc}"
    else:
        # Minimal Two-NN fallback
        from scipy.spatial.distance import cdist

        D = cdist(Xs, Xs)
        np.fill_diagonal(D, np.inf)
        nn = np.sort(D, axis=1)[:, :2]
        mu = nn[:, 1] / np.maximum(nn[:, 0], 1e-12)
        mu = mu[mu > 1]
        out["estimators"]["TwoNN"] = float(1.0 / np.mean(np.log(mu))) if len(mu) else float("nan")

    numeric = [v for v in out["estimators"].values() if isinstance(v, float) and np.isfinite(v)]
    # Prefer TwoNN as primary b (matches Exp 26 / Facco et al.)
    b_primary = out["estimators"].get("TwoNN")
    if not isinstance(b_primary, float) or not np.isfinite(b_primary):
        b_primary = float(np.median(numeric)) if numeric else float("nan")
    out["b_primary_TwoNN"] = float(b_primary) if np.isfinite(b_primary) else float("nan")
    out["b_median_available"] = float(np.median(numeric)) if numeric else float("nan")
    return out


# =============================================================================
# Fixed-t landmark generation (NO percentage, NO undersampling)
# =============================================================================
def split_classes_no_balance(
    X: pd.DataFrame,
    y: pd.Series,
    positive_label: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Keep full class pools — do not undersample."""
    data = X.copy()
    data["__y__"] = y.values
    pos = data[data["__y__"] == positive_label].drop(columns=["__y__"]).reset_index(drop=True)
    neg = data[data["__y__"] != positive_label].drop(columns=["__y__"]).reset_index(drop=True)
    return {"default": pos, "non-default": neg}




def split_classes_maybe_balance(
    X: pd.DataFrame,
    y: pd.Series,
    undersample: bool,
    positive_label: int = 1,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    if undersample:
        X, y = undersample_xy(X, y, positive_label=positive_label, random_state=random_state)
    return split_classes_no_balance(X, y, positive_label=positive_label)


def revised_snapshot_late_split_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Late-split first-class mode: impute / scale / PCA on the FULL table,
    then stratified 80/20 on the reduced customers. Leaky by design
    (matches Historical / No_Undersampling PCA timing).
    """
    from sklearn.impute import SimpleImputer

    miss = X.isna().astype(float)
    miss.columns = [f"miss_{c}" for c in X.columns]
    keep = [c for c in miss.columns if miss[c].sum() > 0]
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    if keep:
        X_imp = pd.concat([X_imp, miss[keep]], axis=1)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X_imp)
    n_comp = min(n_components, max(1, Xs.shape[0] - 1), Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    cols = [f"PCA_{i}" for i in range(1, n_comp + 1)]
    Xp = pd.DataFrame(pca.fit_transform(Xs), columns=cols, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(
        Xp, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, float(pca.explained_variance_ratio_.sum())


def prepare_protocol_clouds(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    split_timing: str,
    undersample: bool,
    positive_label: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    First-class protocol factory for the four split/undersample pairs.

    early and undersample=False  → Early_Split_No_Undersample_H0_And_H1
    early and undersample=True   → Early_Split_And_Undersample_H0_And_H1
    late  and undersample=True   → Late_Split_And_Undersample_H0_And_H1
    late  and undersample=False  → Late_Split_No_Undersample_H0_And_H1
    """
    split_timing = str(split_timing).strip().lower()
    if split_timing not in {"early", "late"}:
        raise ValueError(f"split_timing must be 'early' or 'late', got {split_timing!r}")

    if split_timing == "early":
        X_train, X_test, y_train, y_test, var = revised_snapshot_early_split_pca(
            X, y, n_components=n_components, test_size=test_size, random_state=random_state
        )
        pca_fit = "train_only"
        if undersample:
            X_train, y_train = undersample_xy(
                X_train, y_train, positive_label=positive_label, random_state=random_state
            )
            X_test, y_test = undersample_xy(
                X_test, y_test, positive_label=positive_label, random_state=random_state
            )
    else:
        if undersample:
            # Balance on the raw table first so PCA sees the same balanced cloud
            # the landmarks will be drawn from, then full-table PCA, then split.
            X, y = undersample_xy(X, y, positive_label=positive_label, random_state=random_state)
        X_train, X_test, y_train, y_test, var = revised_snapshot_late_split_pca(
            X, y, n_components=n_components, test_size=test_size, random_state=random_state
        )
        pca_fit = "full_table"

    meta = {
        "split_timing": split_timing,
        "undersample": bool(undersample),
        "pca_fit": pca_fit,
        "variance_retained": float(var),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_pos": int((y_train == positive_label).sum()),
        "train_neg": int((y_train != positive_label).sum()),
        "test_pos": int((y_test == positive_label).sum()),
        "test_neg": int((y_test != positive_label).sum()),
    }
    return X_train, X_test, y_train, y_test, meta


def generate_fixed_t_snapshots(
    class_frames: Dict[str, pd.DataFrame],
    t: int,
    l: int,
    output_root: Path,
    tag: str,
    random_state: int = 42,
    store_index_sets: bool = True,
    undersample: bool = False,
) -> Dict[str, Any]:
    """
    Draw l snapshots of exactly t points (without replacement within a snapshot)
    from each class pool. Same absolute t for every class.

    Saves:
      output_root / {class}_T{t} / landmarks_{i}.csv
      optional index JSONs for overlap analysis
    """
    rng = check_random_state(random_state)
    meta: Dict[str, Any] = {
        "t": int(t),
        "l": int(l),
        "tag": tag,
        "undersample": bool(undersample),
        "no_undersampling": not bool(undersample),
        "classes": {},
        "index_sets": {},
    }
    output_root = win_long_path(Path(output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    for class_name, frame in class_frames.items():
        n = len(frame)
        if t > n:
            raise ValueError(
                f"Cannot draw t={t} from class '{class_name}' with only n={n} rows "
                f"(no undersampling / no replacement within a snapshot)."
            )
        class_dir = win_long_path(output_root / f"{class_name}_T{t}")
        class_dir.mkdir(parents=True, exist_ok=True)
        index_sets = []
        for i in range(l):
            # Independent snapshot seeds derived from master seed
            local_rng = check_random_state(rng.randint(0, 2**31 - 1))
            idx = local_rng.choice(n, size=t, replace=False)
            snap = frame.iloc[idx]
            snap.to_csv(class_dir / f"landmarks_{i}.csv", index=False)
            index_sets.append(sorted(map(int, idx.tolist())))
        meta["classes"][class_name] = {"n_pool": n, "n_snapshots": l, "t": t}
        if store_index_sets:
            meta["index_sets"][class_name] = index_sets
            with open(class_dir / "snapshot_index_sets.json", "w", encoding="utf-8") as f:
                json.dump(index_sets, f)
    with open(output_root / f"snapshot_meta_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in meta.items() if k != "index_sets"}, f, indent=2)
        # store indices separately (can be large)
    with open(output_root / f"snapshot_indices_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta["index_sets"], f)
    return meta


def pairwise_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def pairwise_overlap_fraction(a: Sequence[int], b: Sequence[int], t: int) -> float:
    """|A∩B| / t  (fraction of a snapshot's points shared with another)."""
    if t <= 0:
        return float("nan")
    return len(set(a) & set(b)) / float(t)


def analyze_snapshot_overlap(
    index_sets: List[List[int]],
    t: int,
    n_pool: int,
    n_pair_sample: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Pairwise snapshot overlap summary for one class.
    If C(l,2) is large, subsample pairs.
    """
    rng = check_random_state(random_state)
    l = len(index_sets)
    all_pairs = list(combinations(range(l), 2))
    if len(all_pairs) > n_pair_sample:
        chosen = rng.choice(len(all_pairs), size=n_pair_sample, replace=False)
        pairs = [all_pairs[i] for i in chosen]
        sampled = True
    else:
        pairs = all_pairs
        sampled = False

    jaccards = []
    overlaps = []
    for i, j in pairs:
        jaccards.append(pairwise_jaccard(index_sets[i], index_sets[j]))
        overlaps.append(pairwise_overlap_fraction(index_sets[i], index_sets[j], t))

    # Theoretical expected overlap fraction for two independent samples without replacement:
    # E[|A∩B|]/t = t/n  (approximately, exact hypergeometric mean is t*(t/n) wait)
    # Exact: E[|A∩B|] = t * (t / n) = t^2 / n   => E[frac] = t/n
    expected_overlap_frac = t / n_pool if n_pool else float("nan")
    expected_jaccard = (
        expected_overlap_frac / (2 - expected_overlap_frac)
        if np.isfinite(expected_overlap_frac) and expected_overlap_frac < 2
        else float("nan")
    )

    return {
        "n_snapshots": l,
        "n_pairs_evaluated": len(pairs),
        "pairs_sampled": sampled,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
        "std_jaccard": float(np.std(jaccards)) if jaccards else float("nan"),
        "mean_overlap_frac": float(np.mean(overlaps)) if overlaps else float("nan"),
        "std_overlap_frac": float(np.std(overlaps)) if overlaps else float("nan"),
        "expected_overlap_frac_indep": float(expected_overlap_frac),
        "expected_jaccard_approx": float(expected_jaccard),
        "reuse_ratio_tl_over_n": reuse_ratio(t, l, n_pool),
        "jaccard_values": jaccards,
        "overlap_frac_values": overlaps,
    }


def overlap_significance_tests(
    index_sets: List[List[int]],
    t: int,
    n_pool: int,
    n_permutations: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Formal tests that observed pairwise overlap is consistent with independent
    uniform sampling without replacement (null), vs systematically too high.

    1) One-sided permutation / Monte-Carlo test on mean overlap fraction.
    2) Mann–Whitney U: observed pair overlaps vs null-simulated pair overlaps.
    """
    rng = check_random_state(random_state)
    observed = analyze_snapshot_overlap(
        index_sets, t=t, n_pool=n_pool, n_pair_sample=300, random_state=random_state
    )
    obs_mean = observed["mean_overlap_frac"]

    null_means = []
    null_pair_values = []
    for _ in range(n_permutations):
        sim_sets = [rng.choice(n_pool, size=t, replace=False).tolist() for _ in range(len(index_sets))]
        sim = analyze_snapshot_overlap(
            sim_sets, t=t, n_pool=n_pool, n_pair_sample=200, random_state=rng.randint(0, 2**31 - 1)
        )
        null_means.append(sim["mean_overlap_frac"])
        null_pair_values.extend(sim["overlap_frac_values"])

    # p-value: fraction of null means >= observed (more overlap than chance)
    p_mean = (1 + sum(m >= obs_mean for m in null_means)) / (n_permutations + 1)

    obs_pairs = observed["overlap_frac_values"]
    if obs_pairs and null_pair_values:
        # alternative: observed overlaps stochastically greater than null
        u_stat, p_mw = mannwhitneyu(obs_pairs, null_pair_values, alternative="greater")
    else:
        u_stat, p_mw = float("nan"), float("nan")

    return {
        "observed_mean_overlap_frac": obs_mean,
        "null_mean_of_means": float(np.mean(null_means)),
        "null_std_of_means": float(np.std(null_means)),
        "expected_overlap_frac_theory": observed["expected_overlap_frac_indep"],
        "p_value_mean_overlap_greater_than_null": float(p_mean),
        "mannwhitney_U": float(u_stat) if np.isfinite(u_stat) else float("nan"),
        "mannwhitney_p_greater": float(p_mw) if np.isfinite(p_mw) else float("nan"),
        "n_permutations": n_permutations,
        "interpretation": (
            "Large p-values support the null that snapshots behave like independent "
            "uniform draws. Small p-values suggest excess overlap beyond chance."
        ),
        "summary_without_raw": {
            k: v
            for k, v in observed.items()
            if k not in ("jaccard_values", "overlap_frac_values")
        },
    }


# =============================================================================
# Barcodes + ML
# =============================================================================
def compute_barcode_stats_for_snapshot_dir(
    snapshot_dir: Path,
    label: int,
    dim: int = 2,
) -> pd.DataFrame:
    rows = []
    files = sorted(snapshot_dir.glob("landmarks_*.csv"))
    for fp in files:
        pts = pd.read_csv(fp).values
        dgms = ripser(pts, maxdim=dim - 1)["dgms"]
        row = []
        for d in range(dim):
            row.extend(compute_barcode_statistics(dgms[d]))
        row.append(label)
        rows.append(row)
    cols = [f"g{i}_{j}" for j in range(dim) for i in range(1, 13)] + ["label"]
    return pd.DataFrame(rows, columns=cols)


def build_barcode_matrix_for_tag(
    landmarks_root: Path,
    t: int,
    label_map: Dict[int, str] = None,
) -> pd.DataFrame:
    if label_map is None:
        label_map = {1: "default", 0: "non-default"}
    frames = []
    for lab, name in label_map.items():
        d = win_long_path(landmarks_root / f"{name}_T{t}")
        if not d.exists():
            raise FileNotFoundError(d)
        frames.append(compute_barcode_stats_for_snapshot_dir(d, label=lab))
    return pd.concat(frames, ignore_index=True)


def fit_simple_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    X_tr = train_df.drop(columns=["label"]).values
    y_tr = train_df["label"].values
    X_te = test_df.drop(columns=["label"]).values
    y_te = test_df["label"].values

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    models = {
        "logistic": LogisticRegression(max_iter=2000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, class_weight="balanced"
        ),
        "xgb": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "svm": SVC(probability=True, random_state=random_state, class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_te, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
        }
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_te)[:, 1]
            try:
                row["roc_auc"] = float(roc_auc_score(y_te, proba))
            except Exception:
                row["roc_auc"] = float("nan")
            try:
                row["average_precision"] = float(average_precision_score(y_te, proba))
            except Exception:
                row["average_precision"] = float("nan")
        else:
            row["roc_auc"] = float("nan")
            row["average_precision"] = float("nan")
        rows.append(row)
    return rows


def revised_snapshot_early_split_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Protocol B: split first, then impute / scale / PCA on train only.
    Median imputation + missing indicators keep Polish/PKDD usable without leakage.
    """
    from sklearn.impute import SimpleImputer

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Missing indicators (train-fit columns) then median impute
    miss_tr = X_train.isna().astype(float)
    miss_te = X_test.isna().astype(float)
    miss_tr.columns = [f"miss_{c}" for c in X_train.columns]
    miss_te.columns = [f"miss_{c}" for c in X_test.columns]
    # Drop all-zero indicator columns (no missingness in train)
    keep = [c for c in miss_tr.columns if miss_tr[c].sum() > 0]
    miss_tr = miss_tr[keep]
    miss_te = miss_te[keep]

    imputer = SimpleImputer(strategy="median")
    Xtr_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    Xte_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    if keep:
        Xtr_imp = pd.concat([Xtr_imp, miss_tr], axis=1)
        Xte_imp = pd.concat([Xte_imp, miss_te], axis=1)

    scaler = MinMaxScaler()
    Xtr_s = scaler.fit_transform(Xtr_imp)
    Xte_s = scaler.transform(Xte_imp)

    n_comp = min(n_components, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    cols = [f"PCA_{i}" for i in range(1, n_comp + 1)]
    Xtr_p = pd.DataFrame(pca.fit_transform(Xtr_s), columns=cols, index=X_train.index)
    Xte_p = pd.DataFrame(pca.transform(Xte_s), columns=cols, index=X_test.index)
    return Xtr_p, Xte_p, y_train, y_test, float(pca.explained_variance_ratio_.sum())




# =============================================================================
# REVISED SNAPSHOT PROTOCOL STAGES (Experiment 9)
# =============================================================================

def set_revised_snapshot_arm(protocol_bucket, split_timing, undersample):
    """Dataset scripts call this once so Experiment 9 helpers know which arm they are in."""
    global PROTOCOL_BUCKET, SPLIT_TIMING, UNDERSAMPLE, EXP_NAME
    global RESULTS, DATA_LANDMARKS, DATA_TDA, DATA_BARCODES
    PROTOCOL_BUCKET = protocol_bucket
    SPLIT_TIMING = split_timing
    UNDERSAMPLE = undersample
    EXP_NAME = "9_Revised_Snapshot_Protocol"
    RESULTS = REPO_ROOT / "6_Results" / PROTOCOL_BUCKET / EXP_NAME
    DATA_LANDMARKS = REPO_ROOT / "1_Data" / "Landmark_Sets" / PROTOCOL_BUCKET / EXP_NAME
    DATA_TDA = REPO_ROOT / "1_Data" / "TDA_Datasets" / PROTOCOL_BUCKET / EXP_NAME
    DATA_BARCODES = REPO_ROOT / "1_Data" / "Barcode_Statistics" / PROTOCOL_BUCKET / EXP_NAME


PROTOCOL_BUCKET = "Late_Split_And_Undersample_H0_And_H1"
SPLIT_TIMING = "late"
UNDERSAMPLE = True
EXP_NAME = "9_Revised_Snapshot_Protocol"
RESULTS = REPO_ROOT / "6_Results" / PROTOCOL_BUCKET / EXP_NAME
DATA_LANDMARKS = REPO_ROOT / "1_Data" / "Landmark_Sets" / PROTOCOL_BUCKET / EXP_NAME
DATA_TDA = REPO_ROOT / "1_Data" / "TDA_Datasets" / PROTOCOL_BUCKET / EXP_NAME
DATA_BARCODES = REPO_ROOT / "1_Data" / "Barcode_Statistics" / PROTOCOL_BUCKET / EXP_NAME

DATASET_SPECS = {
    "credit_card_default": {
        "path": REPO_ROOT
        / "1_Data/Processed_Datasets/Default_Of_Credit_Card_Client_Data/processed_data.xlsx",
        "run_full_nonsplit": True,
    },
    "statlog_german": {
        "path": REPO_ROOT
        / "1_Data/Processed_Datasets/Statlog_German_Credit_Data/processed_data.xlsx",
        "run_full_nonsplit": False,
    },
    "south_german_credit": {
        "path": REPO_ROOT / "1_Data/Processed_Datasets/South_German_Credit/processed_data.csv",
        "run_full_nonsplit": False,
    },
    "pkdd_czech": {
        "path": REPO_ROOT / "1_Data/Processed_Datasets/PKDD_Czech_Financial/processed_data.csv",
        "run_full_nonsplit": False,
    },
    "polish_bankruptcy": {
        "path": REPO_ROOT
        / "1_Data/Processed_Datasets/Polish_Bankruptcy_3Year/processed_data.csv",
        "run_full_nonsplit": False,
    },
    "taiwan_bankruptcy": {
        "path": REPO_ROOT / "1_Data/Processed_Datasets/Taiwan_Bankruptcy/processed_data.csv",
        "run_full_nonsplit": False,
    },
}


def _pca_rank(dataset_key: str) -> int:
    return int(get_dataset_config(dataset_key).notes["pca_n_components_exp3"])


def load_xy(dataset_key: str):
    cfg = get_dataset_config(dataset_key)
    spec = DATASET_SPECS[dataset_key]
    path = spec["path"]
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    target = cfg.target_column
    if target not in df.columns:
        # common alternates
        for alt in ("Class", "class", "target", "Target", "default payment next month", "y"):
            if alt in df.columns:
                target = alt
                break
    drop = [c for c in ("Unnamed: 0", "id", "ID", target) if c in df.columns]
    X = df.drop(columns=drop)
    # keep numeric only for PCA/PH
    X = X.select_dtypes(include=[np.number]).copy()
    y = df[target].astype(int)
    # map south german if needed (already 0/1 in processed)
    return X, y, cfg, spec


def design_for_dataset(dataset_key: str, X=None, y=None) -> dict:
    from sklearn.impute import SimpleImputer

    cfg = get_dataset_config(dataset_key)
    spec = DATASET_SPECS[dataset_key]
    if X is None or y is None:
        X, y, cfg, spec = load_xy(dataset_key)
    pos = int((y == cfg.positive_label).sum())
    neg = int((y != cfg.positive_label).sum())

    # ID on median-imputed raw numeric (no split leakage for exploratory ID)
    X_id = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X), columns=X.columns
    )
    id_raw = estimate_intrinsic_dimensions(X_id.values, n_samples=min(2000, len(X_id)))

    pca_n = _pca_rank(dataset_key)
    Xtr, Xte, ytr, yte, cloud = prepare_protocol_clouds(
        X,
        y,
        n_components=pca_n,
        split_timing=SPLIT_TIMING,
        undersample=UNDERSAMPLE,
        positive_label=cfg.positive_label,
        random_state=42,
    )
    var = cloud["variance_retained"]
    id_pca = estimate_intrinsic_dimensions(Xtr.values, n_samples=min(2000, len(Xtr)))
    b = id_pca["b_primary_TwoNN"]
    if not np.isfinite(b):
        b = id_raw["b_primary_TwoNN"]

    train_pos = int(cloud["train_pos"])
    train_neg = int(cloud["train_neg"])
    test_pos = int(cloud["test_pos"])
    test_neg = int(cloud["test_neg"])

    # Concern A/B tables still reported on each pool at meeting targets
    rec_train = recommend_t_l_separated(
        train_pos, train_neg, b=b, train_l_target=DEFAULT_TRAIN_L, test_l_target=DEFAULT_TEST_L
    )
    rec_test_pool = recommend_t_l_separated(
        test_pos, test_neg, b=b, train_l_target=DEFAULT_TRAIN_L, test_l_target=DEFAULT_TEST_L
    )

    # Joint choice: one t, train_l≈60, test_l≈15, reuse-safe on both pools
    joint = choose_joint_t_train_test_l(
        train_pos, train_neg, test_pos, test_neg,
        target_train_l=DEFAULT_TRAIN_L,
        target_test_l=DEFAULT_TEST_L,
    )
    chosen_t = int(joint["t"])
    eff_train_l = int(joint["train_l"])
    eff_test_l = int(joint["test_l"])

    notes = []
    if eff_train_l < DEFAULT_TRAIN_L or eff_test_l < DEFAULT_TEST_L:
        notes.append(
            f"Concern B joint choice set train_l={eff_train_l}, test_l={eff_test_l} "
            f"(meeting asked 60/15) at t={chosen_t}; "
            f"train_min={joint['n_train_min']}, test_min={joint['n_test_min']}"
        )
    if joint.get("relaxed_test_reuse"):
        notes.append(
            f"Test reuse limit relaxed to {joint['test_reuse_limit']} because "
            "strict reuse<=1 could not support min_test_l with a usable t"
        )

    t_cap = chosen_t
    flo = max(5, t_cap // 3) if t_cap >= 15 else max(3, t_cap // 3)
    mid = max(flo, (2 * t_cap) // 3)
    t_sweep = sorted({flo, mid, t_cap})

    full_rec = recommend_t_l_separated(pos, neg, b=b, train_l_target=60, test_l_target=15)

    design = {
        "dataset_key": dataset_key,
        "display_name": cfg.display_name,
        "n_total": int(len(X)),
        "n_pos": pos,
        "n_neg": neg,
        "default_rate": pos / len(X),
        "pca_components": pca_n,
        "pca_variance_retained": var,
        "pca_fit": cloud["pca_fit"],
        "protocol_bucket": PROTOCOL_BUCKET,
        "split_timing": SPLIT_TIMING,
        "undersample": UNDERSAMPLE,
        "intrinsic_dim_raw": id_raw,
        "intrinsic_dim_pca_train": id_pca,
        "b_used": float(b),
        "split_counts": {
            "train_pos": train_pos,
            "train_neg": train_neg,
            "test_pos": test_pos,
            "test_neg": test_neg,
        },
        "meeting_defaults": {"train_l": DEFAULT_TRAIN_L, "test_l": DEFAULT_TEST_L},
        "effective_defaults": {
            "train_l": eff_train_l,
            "test_l": eff_test_l,
            "joint_choice": joint,
            "notes": notes,
        },
        "zaniar_sweep": {"train_l": list(ZANIAR_TRAIN_L), "test_l": list(ZANIAR_TEST_L)},
        # Largest t that can host the full Zaniar corner (train_l=100, test_l=30) under R<=1
        "zaniar_t": int(
            max(
                0,
                min(
                    max_t_for_reuse(min(train_pos, train_neg), max(ZANIAR_TRAIN_L)),
                    max_t_for_reuse(min(test_pos, test_neg), max(ZANIAR_TEST_L)),
                ),
            )
        ),
        "dcccd_full_l": list(DCCCD_FULL_L),
        "concern_A_and_B_train_pool": rec_train,
        "concern_A_and_B_test_pool": rec_test_pool,
        "chosen_t": chosen_t,
        "t_sweep": t_sweep,
        "formula_at_chosen_t": (
            formula_l_from_t_b(chosen_t, b) if (chosen_t >= 3 and np.isfinite(b) and b > 0) else None
        ),
        "full_data_rec": full_rec,
        "worked_examples": _worked_examples(
            chosen_t, b, train_pos, train_neg, test_pos, test_neg, eff_train_l, eff_test_l
        ),
    }

    out_dir = RESULTS / cfg.folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "design.json", design)

    # Flat tables for the report
    formula_df = pd.DataFrame(rec_train["concern_A_formula"]["rows"])
    formula_df.to_csv(out_dir / "concern_A_formula_rows.csv", index=False)
    reuse_df = pd.DataFrame(rec_train["concern_B_reuse"]["rows"])
    reuse_df.to_csv(out_dir / "concern_B_reuse_rows.csv", index=False)

    # Worked calculation table
    pd.DataFrame(design["worked_examples"]).to_csv(out_dir / "worked_calculations.csv", index=False)
    return design


def _worked_examples(t, b, train_pos, train_neg, test_pos, test_neg, train_l=None, test_l=None):
    rows = []
    train_l = DEFAULT_TRAIN_L if train_l is None else int(train_l)
    test_l = DEFAULT_TEST_L if test_l is None else int(test_l)
    if t < 3 or not np.isfinite(b) or b <= 0:
        return rows
    l_f = formula_l_from_t_b(t, b)
    rows.append(
        {
            "step": "A1_formula",
            "expression": "l = (t / ln(t))^(2/b)",
            "t": t,
            "b": b,
            "ln_t": float(np.log(t)),
            "t_over_ln_t": float(t / np.log(t)),
            "exponent_2_over_b": float(2.0 / b),
            "result_l_formula": l_f,
            "notes": "Concern A only — theoretical snapshot count suggestion",
        }
    )
    for split, npos, nneg, l in (
        ("train", train_pos, train_neg, train_l),
        ("test", test_pos, test_neg, test_l),
    ):
        audit = audit_reuse_constraints(npos, nneg, t, l)
        rows.append(
            {
                "step": f"B_{split}_reuse",
                "expression": "R = (t*l)/n_class",
                "t": t,
                "l": l,
                "n_pos": npos,
                "n_neg": nneg,
                "reuse_pos": audit["reuse_pos"],
                "reuse_neg": audit["reuse_neg"],
                "ok_reuse": audit["ok_reuse"],
                "ok_t_fraction": audit["ok_t_fraction"],
                "notes": "Concern B only — sampling reuse feasibility at effective l",
            }
        )
    return rows


def _run_key_done(results_csv: Path, run_key: str) -> bool:
    if not results_csv.exists():
        return False
    df = pd.read_csv(results_csv)
    return run_key in set(df.get("run_key", []).astype(str))


def run_split_setting(
    dataset_key: str,
    design: dict,
    t: int,
    train_l: int,
    test_l: int,
    mode: str = "default",
) -> None:
    X, y, cfg, spec = load_xy(dataset_key)
    folder = cfg.folder_name
    run_key = f"{dataset_key}|split|t{t}|train{train_l}|test{test_l}|{mode}"
    results_csv = RESULTS / folder / "ml_results.csv"
    if _run_key_done(results_csv, run_key):
        print(f"[skip] {run_key}")
        return

    print(f"\n=== {run_key} ===")
    Xtr, Xte, ytr, yte, cloud = prepare_protocol_clouds(
        X,
        y,
        n_components=_pca_rank(dataset_key),
        split_timing=SPLIT_TIMING,
        undersample=UNDERSAMPLE,
        positive_label=cfg.positive_label,
        random_state=42,
    )
    var = cloud["variance_retained"]
    train_classes = split_classes_no_balance(Xtr, ytr, positive_label=cfg.positive_label)
    test_classes = split_classes_no_balance(Xte, yte, positive_label=cfg.positive_label)

    # Feasibility guard
    for name, frame in {**{f"train/{k}": v for k, v in train_classes.items()}, **{f"test/{k}": v for k, v in test_classes.items()}}.items():
        if len(frame) < t:
            raise ValueError(f"{run_key}: pool {name} has {len(frame)} < t={t}")

    lm_train = win_long_path(DATA_LANDMARKS / folder / f"split_t{t}_tr{train_l}_te{test_l}" / "train")
    lm_test = win_long_path(DATA_LANDMARKS / folder / f"split_t{t}_tr{train_l}_te{test_l}" / "test")
    tda_dir = win_long_path(DATA_TDA / folder / f"split_t{t}_tr{train_l}_te{test_l}")
    bar_dir = win_long_path(DATA_BARCODES / folder / f"split_t{t}_tr{train_l}_te{test_l}")
    tda_dir.mkdir(parents=True, exist_ok=True)
    bar_dir.mkdir(parents=True, exist_ok=True)

    meta_tr = generate_fixed_t_snapshots(
        train_classes, t=t, l=train_l, output_root=lm_train, tag="train", random_state=42, undersample=UNDERSAMPLE
    )
    meta_te = generate_fixed_t_snapshots(
        test_classes, t=t, l=test_l, output_root=lm_test, tag="test", random_state=43, undersample=UNDERSAMPLE
    )

    # Overlap + significance on train default class (and non-default)
    overlap_report = {}
    for cname in ("default", "non-default"):
        idx = meta_tr["index_sets"][cname]
        n_pool = meta_tr["classes"][cname]["n_pool"]
        summary = analyze_snapshot_overlap(idx, t=t, n_pool=n_pool, random_state=42)
        # strip heavy arrays for JSON
        summary_light = {k: v for k, v in summary.items() if k not in ("jaccard_values", "overlap_frac_values")}
        sig = overlap_significance_tests(
            idx, t=t, n_pool=n_pool, n_permutations=150, random_state=42
        )
        overlap_report[f"train_{cname}"] = {"summary": summary_light, "significance": sig}

    save_json(RESULTS / folder / f"overlap_{run_key.replace('|', '_')}.json", overlap_report)

    train_bar = build_barcode_matrix_for_tag(lm_train, t=t)
    test_bar = build_barcode_matrix_for_tag(lm_test, t=t)
    train_bar.to_csv(tda_dir / "train_barcodes.csv", index=False)
    test_bar.to_csv(tda_dir / "test_barcodes.csv", index=False)
    train_bar.to_csv(bar_dir / "train_barcodes.csv", index=False)
    test_bar.to_csv(bar_dir / "test_barcodes.csv", index=False)

    ml_rows = fit_simple_models(train_bar, test_bar, random_state=42)
    for row in ml_rows:
        row.update(
            {
                "run_key": run_key,
                "dataset": dataset_key,
                "protocol": f"{SPLIT_TIMING}_split_{'undersample' if UNDERSAMPLE else 'no_undersample'}_fixed_t",
                "protocol_bucket": PROTOCOL_BUCKET,
                "mode": mode,
                "t": t,
                "train_l": train_l,
                "test_l": test_l,
                "b_used": design["b_used"],
                "formula_l": design.get("formula_at_chosen_t"),
                "pca_variance": var,
                "n_train_snapshots": len(train_bar),
                "n_test_snapshots": len(test_bar),
                "train_class_balance": train_bar["label"].value_counts().to_dict(),
                "reuse_train_default": audit_reuse_constraints(
                    design["split_counts"]["train_pos"],
                    design["split_counts"]["train_neg"],
                    t,
                    train_l,
                )["reuse_pos"],
                "reuse_train_nondefault": audit_reuse_constraints(
                    design["split_counts"]["train_pos"],
                    design["split_counts"]["train_neg"],
                    t,
                    train_l,
                )["reuse_neg"],
            }
        )

    out_df = pd.DataFrame(ml_rows)
    if results_csv.exists():
        out_df = pd.concat([pd.read_csv(results_csv), out_df], ignore_index=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(results_csv, index=False)
    print(f"Saved ML results -> {results_csv}")


def run_full_nonsplit(dataset_key: str, design: dict, t: int, l: int) -> None:
    """Non-split full-data snapshots (DCCCD 60–90). Evaluate via internal 80/20 on barcodes."""
    X, y, cfg, spec = load_xy(dataset_key)
    folder = cfg.folder_name
    run_key = f"{dataset_key}|full|t{t}|l{l}"
    results_csv = RESULTS / folder / "ml_results.csv"
    if _run_key_done(results_csv, run_key):
        print(f"[skip] {run_key}")
        return

    print(f"\n=== {run_key} ===")
    # Scale+PCA on FULL data intentionally for this non-split sensitivity arm
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA

    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X_imp)
    pca_n = _pca_rank(dataset_key)
    n_comp = min(pca_n, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    Xp = pd.DataFrame(
        pca.fit_transform(Xs), columns=[f"PCA_{i}" for i in range(1, n_comp + 1)]
    )
    if UNDERSAMPLE:
        Xp, y = undersample_xy(Xp, y, positive_label=cfg.positive_label, random_state=42)
    classes = split_classes_no_balance(Xp, y, positive_label=cfg.positive_label)

    lm = win_long_path(DATA_LANDMARKS / folder / f"full_t{t}_l{l}")
    meta = generate_fixed_t_snapshots(classes, t=t, l=l, output_root=lm, tag="full", random_state=42, undersample=UNDERSAMPLE)

    overlap_report = {}
    for cname in ("default", "non-default"):
        idx = meta["index_sets"][cname]
        n_pool = meta["classes"][cname]["n_pool"]
        summary = analyze_snapshot_overlap(idx, t=t, n_pool=n_pool)
        summary_light = {k: v for k, v in summary.items() if k not in ("jaccard_values", "overlap_frac_values")}
        sig = overlap_significance_tests(idx, t=t, n_pool=n_pool, n_permutations=150)
        overlap_report[cname] = {"summary": summary_light, "significance": sig}
    save_json(RESULTS / folder / f"overlap_{run_key.replace('|', '_')}.json", overlap_report)

    bar = build_barcode_matrix_for_tag(lm, t=t)
    tda_dir = win_long_path(DATA_TDA / folder / f"full_t{t}_l{l}")
    bar_dir = win_long_path(DATA_BARCODES / folder / f"full_t{t}_l{l}")
    tda_dir.mkdir(parents=True, exist_ok=True)
    bar_dir.mkdir(parents=True, exist_ok=True)
    bar.to_csv(tda_dir / "all_barcodes.csv", index=False)
    bar.to_csv(bar_dir / "all_barcodes.csv", index=False)

    # Stratified split on barcode rows (snapshot-level)
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(bar, test_size=0.2, random_state=42, stratify=bar["label"])
    ml_rows = fit_simple_models(tr, te, random_state=42)
    for row in ml_rows:
        row.update(
            {
                "run_key": run_key,
                "dataset": dataset_key,
                "protocol": "full_data_nonsplit_then_barcode_split",
                "mode": "full_dcccd_range",
                "t": t,
                "train_l": l,
                "test_l": None,
                "b_used": design["b_used"],
                "formula_l": formula_l_from_t_b(t, design["b_used"]),
                "pca_variance": float(pca.explained_variance_ratio_.sum()),
                "n_train_snapshots": len(tr),
                "n_test_snapshots": len(te),
                "reuse_binding": audit_reuse_constraints(
                    design["n_pos"], design["n_neg"], t, l
                )["reuse_binding"],
            }
        )
    out_df = pd.DataFrame(ml_rows)
    if results_csv.exists():
        out_df = pd.concat([pd.read_csv(results_csv), out_df], ignore_index=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(results_csv, index=False)
    print(f"Saved ML results -> {results_csv}")


def run_design_all(keys):
    designs = {}
    for k in keys:
        print(f"\n[design] {k}")
        try:
            designs[k] = design_for_dataset(k)
            print(f"  b={designs[k]['b_used']:.4f}  chosen_t={designs[k]['chosen_t']}  t_sweep={designs[k]['t_sweep']}")
        except Exception as exc:
            print(f"  DESIGN FAILED: {exc}")
            traceback.print_exc()
    save_json(RESULTS / "all_designs.json", {k: _strip_heavy(v) for k, v in designs.items()})
    return designs


def _strip_heavy(design: dict) -> dict:
    # keep JSON lighter
    d = dict(design)
    return d


def run_split_ml(designs, keys, sweep: bool = True):
    for k in keys:
        if k not in designs:
            continue
        d = designs[k]
        t0 = d["chosen_t"]
        eff_tr = int(d.get("effective_defaults", {}).get("train_l", DEFAULT_TRAIN_L))
        eff_te = int(d.get("effective_defaults", {}).get("test_l", DEFAULT_TEST_L))
        # Default (meeting 60/15, or Concern-B-adapted) at each t sweep point
        for t in d["t_sweep"]:
            try:
                run_split_setting(
                    k, d, t=t, train_l=eff_tr, test_l=eff_te, mode="default_60_15"
                )
            except Exception as exc:
                print(f"FAILED default {k} t={t}: {exc}")
                traceback.print_exc()
        if sweep:
            # Zaniar 3x3 at a t that can actually host the upper corner under R<=1.
            # (Using max chosen_t often makes the entire grid infeasible — e.g. DCCCD t=88.)
            t_z = int(d.get("zaniar_t") or 0)
            if t_z < 3:
                # fall back: largest t_sweep point that admits at least one non-default cell
                t_z = max(d["t_sweep"])
            print(f"[zaniar] {k}: using t={t_z} for sweep grid (chosen_t was {t0})")
            for tr_l in ZANIAR_TRAIN_L:
                for te_l in ZANIAR_TEST_L:
                    if tr_l == eff_tr and te_l == eff_te and t_z == t0:
                        continue
                    sc = d["split_counts"]
                    audit_tr = audit_reuse_constraints(sc["train_pos"], sc["train_neg"], t_z, tr_l)
                    audit_te = audit_reuse_constraints(sc["test_pos"], sc["test_neg"], t_z, te_l)
                    if not audit_tr["ok_reuse"] or not audit_te["ok_reuse"]:
                        print(
                            f"[reuse-skip] {k} train_l={tr_l} test_l={te_l} t={t_z} "
                            f"reuse_tr={audit_tr['reuse_binding']:.3f} "
                            f"reuse_te={audit_te['reuse_binding']:.3f}"
                        )
                        skip_path = RESULTS / get_dataset_config(k).folder_name / "reuse_skips.csv"
                        row = pd.DataFrame(
                            [
                                {
                                    "dataset": k,
                                    "t": t_z,
                                    "train_l": tr_l,
                                    "test_l": te_l,
                                    "reuse_train_binding": audit_tr["reuse_binding"],
                                    "reuse_test_binding": audit_te["reuse_binding"],
                                    "reason": "reuse_>1_on_train_or_test",
                                }
                            ]
                        )
                        if skip_path.exists():
                            row = pd.concat([pd.read_csv(skip_path), row], ignore_index=True)
                        skip_path.parent.mkdir(parents=True, exist_ok=True)
                        row.to_csv(skip_path, index=False)
                        continue
                    try:
                        run_split_setting(
                            k, d, t=t_z, train_l=tr_l, test_l=te_l, mode="zaniar_sweep"
                        )
                    except Exception as exc:
                        print(f"FAILED sweep {k} {tr_l}/{te_l}: {exc}")
                        traceback.print_exc()


def run_full_ml(designs, keys):
    for k in keys:
        if k not in designs:
            continue
        if not DATASET_SPECS[k].get("run_full_nonsplit"):
            continue
        d = designs[k]
        t = d["chosen_t"]
        for l in DCCCD_FULL_L:
            audit = audit_reuse_constraints(d["n_pos"], d["n_neg"], t, l)
            if not audit["ok_reuse"]:
                print(f"[reuse-skip full] {k} l={l} reuse={audit['reuse_binding']:.3f}")
                continue
            try:
                run_full_nonsplit(k, d, t=t, l=l)
            except Exception as exc:
                print(f"FAILED full {k} l={l}: {exc}")
                traceback.print_exc()



