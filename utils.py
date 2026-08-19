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
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import kmapper as km
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pyvis.network import Network
from itertools import product
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from ripser import ripser
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from kmapper import KeplerMapper
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
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
                             confusion_matrix)

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

ACTIVE_TDA_PROTOCOL_BUCKETS = (
    "Historical_Late_Split_Balanced_TDA",
    "Early_Split_TDA",
    "No_Undersampling",
    "Early_Split_TDA_And_No_Undersampling",
)

ACTIVE_TDA_EXPERIMENT_NAMES = (
    "1_PH_Default_Parameters",
    "2_PH_Tuned_Parameters",
    "3_H0_Only",
    "4_Dropping_Correlated_Barcode_Statistics_Columns",
    "5_Linear_Regression_For_Prediction",
    "6_Sampling_Ratio_Audit",
    "7_Snapshot_Mean_Variance",
    "8_Null_Hypothesis_Algorithm2",
    "9_Revised_Snapshot_Protocol",
)

TDA_PROTOCOL_SPECS: Dict[str, Dict[str, Any]] = {
    "Historical_Late_Split_Balanced_TDA": {
        "split_timing": "late",
        "undersample": True,
        "description": "Scale/PCA on the full table, undersample majority to minority count, then 80/20 on barcode rows.",
    },
    "Early_Split_TDA": {
        "split_timing": "early",
        "undersample": True,
        "description": "Stratified 80/20 on customers first; train-only scaler/PCA; undersample within each split.",
    },
    "No_Undersampling": {
        "split_timing": "late",
        "undersample": False,
        "description": "Late-split historical geometry without majority downsample. t = floor(n_class * L / 100) per class.",
    },
    "Early_Split_TDA_And_No_Undersampling": {
        "split_timing": "early",
        "undersample": False,
        "description": "Early customer split, train-only scaler/PCA, full class pools (no undersample).",
    },
}


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
    path = REPO_ROOT / "1_Data" / kind / protocol_bucket / experiment_name / dataset_folder
    for part in extra:
        if part:
            path = path / part
    return win_long_path(path)


def tda_results_dir(protocol_bucket: str, experiment_name: str, dataset_folder: str) -> Path:
    return win_long_path(REPO_ROOT / "6_Results" / protocol_bucket / experiment_name / dataset_folder)


def get_tda_protocol(protocol_bucket: str) -> Dict[str, Any]:
    if protocol_bucket not in TDA_PROTOCOL_SPECS:
        raise ValueError(
            f"Unknown TDA protocol bucket '{protocol_bucket}'. "
            f"Known: {', '.join(TDA_PROTOCOL_SPECS)}"
        )
    spec = dict(TDA_PROTOCOL_SPECS[protocol_bucket])
    spec["bucket"] = protocol_bucket
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
            "landmark_reason": "Original paper percents. n1=300, so 30%/60% are required to get t=90/180 points.",
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
            "landmark_reason": "Original paper percents. n1=6630, so 5% already gives t=331 points.",
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
            "landmark_reason": "Shared new-table percents. 5% would give t=3 on n1=76 (too small for PH); 30% would over-reuse.",
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
            "landmark_reason": "Shared new-table percents. Not Statlog's 30/60: this is a sensitivity table, kept on the same L10/L20 grid as the other new sets.",
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
    abs_save_dir = Path(os.path.abspath(path))
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
    abs_save_dir = Path(os.path.abspath(path))
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
        for model_name, model_info in model_results[data_key].items():
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

def improved_visualize_model_results(
    model_results: dict,
    save_dir: str = "results/visualizations",
    export_metrics: bool = True,
    plot_precision_recall: bool = False,
    hide_axis_labels: bool = False,
    compare_datasets: bool = False,
    colormap: str = "tab10"  # New: supports color schemes like 'viridis', 'OrRd'
):
    """
    Enhanced visualization of model results with dataset comparison, custom colormaps, and layout improvements.
    """
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_titles = ["Accuracy", "Precision", "Recall", "F1 Score"]

    datasets = list(model_results.keys())
    all_models = sorted({model for data in model_results.values() for model in data})

    # Export to CSV if requested
    if export_metrics:
        for metric in metrics:
            rows = []
            for dataset_name, results in model_results.items():
                for model_name, model_stats in results.items():
                    value = model_stats.get(metric, None)
                    if value is not None:
                        rows.append({
                            "Dataset": dataset_name,
                            "Model": model_name,
                            metric.capitalize(): value
                        })
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(save_dir, f"{metric}_summary.csv"), index=False)

    if compare_datasets:
        # Grouped bar chart with datasets grouped per model
        num_metrics = len(metrics)
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Model Performance Comparison (Grouped by Dataset)", fontsize=18)

        x = np.arange(len(all_models))
        width = 0.8 / len(datasets)

        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axs[i // 2, i % 2]
            for idx, dataset_name in enumerate(datasets):
                values = [model_results[dataset_name].get(model, {}).get(metric, 0) for model in all_models]
                offset = (idx - (len(datasets) - 1) / 2) * width
                bars = ax.bar(x + offset, values, width, label=dataset_name)

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                            f"{height:.2f}", ha='center', va='bottom', fontsize=9)

            ax.set_title(title, fontsize=14)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x)
            ax.set_xticklabels(all_models, rotation=45)

            if not hide_axis_labels:
                ax.set_xlabel("Model", fontsize=10)
            ax.set_ylabel(title, fontsize=10)

        # Add legend below the figure (prevents overlapping)
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
                   ncol=len(datasets), fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plot_path = os.path.join(save_dir, "model_comparison_grouped.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path

    else:
        # One chart per dataset
        paths = []
        for dataset_name in datasets:
            fig, axs = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f"Model Performance - {dataset_name}", fontsize=18)

            for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
                ax = axs[i // 2, i % 2]
                plot_data = [
                    (model, model_results[dataset_name].get(model, {}).get(metric, None))
                    for model in all_models
                ]
                plot_data = [item for item in plot_data if item[1] is not None]
                if not plot_data:
                    ax.set_title(f"{title} (No Data)", fontsize=14)
                    continue

                labels, values = zip(*plot_data)
                cmap = plt.get_cmap(colormap)
                colors = cmap(np.linspace(0, 1, len(labels)))

                bars = ax.bar(labels, values, color=colors)
                ax.set_title(title, fontsize=14)
                ax.set_ylim(0, 1.05)

                if not hide_axis_labels:
                    ax.set_xlabel("Model", fontsize=10)
                ax.set_ylabel(title, fontsize=10)
                ax.tick_params(axis='x', rotation=45)

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                            f"{height:.2f}", ha='center', va='bottom', fontsize=9)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plot_path = os.path.join(save_dir, f"model_comparison_{dataset_name}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            paths.append(plot_path)
        return paths
    
def visualize_cross_validation_detailed(
    cross_val_results: Dict[str, Dict[str, Dict[str, Any]]],
    save_dir: str = "results/visualizations",
    colormap: str = "tab10",
    compare_models: bool = False
):
    """
    Visualizes cross-validation results in detail.
    If compare_models is True:
        - One figure per dataset.
        - Each figure compares models via subplots.
    Otherwise:
        - One figure per dataset-model with fold-wise and summary plots.
    """
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    cmap = plt.get_cmap(colormap)
    paths = []

    for dataset_idx, (dataset_name, models_data) in enumerate(cross_val_results.items()):
        model_names = list(models_data.keys())

        if compare_models:
            n_models = len(model_names)
            cols = 2
            rows = (n_models + 1) // cols

            fig, axs = plt.subplots(rows, cols, figsize=(14, 4 * rows))
            fig.suptitle(f"Cross-Validation Comparison — {os.path.basename(dataset_name)}", fontsize=18)

            axs = axs.flatten()
            for idx, model_name in enumerate(model_names):
                ax = axs[idx]
                stats = models_data[model_name]
                scores = stats.get("cross_val_scores", [])
                mean_score = stats.get("mean_accuracy", stats.get("mean_accracy", 0))
                std_score = stats.get("std_accuracy", 0)

                bars = ax.bar([f"Fold {i+1}" for i in range(len(scores))], scores,
                              color=cmap(idx / n_models))
                ax.set_ylim(0, 1.05)
                ax.set_title(f"{model_name}", fontsize=13)
                ax.set_ylabel("Accuracy")

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                            f"{height:.2f}", ha='center', va='bottom', fontsize=9)

                # Add summary as text
                ax.text(0.5, 0.02, f"Mean ± Std: {mean_score:.3f} ± {std_score:.3f}",
                        transform=ax.transAxes, ha='center', fontsize=10, bbox=dict(boxstyle="round", facecolor="#f0f0f0"))

            # Hide any unused subplots
            for j in range(idx + 1, len(axs)):
                fig.delaxes(axs[j])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            filename = f"cv_compare_models_{os.path.basename(dataset_name).replace('.csv','')}.png"
            path = os.path.join(save_dir, filename)
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            paths.append(path)

        else:
            # Individual detailed plots
            for model_idx, (model_name, stats) in enumerate(models_data.items()):
                scores = stats.get("cross_val_scores", [])
                mean_score = stats.get("mean_accuracy", stats.get("mean_accracy", 0))
                std_score = stats.get("std_accuracy", 0)

                fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 2]})
                fig.suptitle(f"Cross-Validation — {model_name} on {os.path.basename(dataset_name)}", fontsize=16)

                # Subplot 1: Fold scores
                ax1 = axs[0]
                bar_colors = cmap(np.linspace(0, 1, len(scores)))
                bars = ax1.bar([f"Fold {i+1}" for i in range(len(scores))], scores, color=bar_colors)
                ax1.set_ylim(0, 1.05)
                ax1.set_ylabel("Accuracy")
                ax1.set_title("Fold-wise Accuracy")

                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                             f"{height:.2f}", ha='center', va='bottom', fontsize=9)

                # Subplot 2: Summary
                ax2 = axs[1]
                ax2.bar(["Mean Accuracy"], [mean_score], yerr=[std_score], capsize=10,
                        color=cmap(0.6), edgecolor='black', linewidth=1.2)
                ax2.set_ylim(0, 1.05)
                ax2.set_title("Mean Accuracy with Std Deviation")
                ax2.set_ylabel("Accuracy")

                ax2.text(0, mean_score + std_score + 0.02,
                         f"Mean ± Std = {mean_score:.3f} ± {std_score:.3f}",
                         ha='center', fontsize=10, fontweight='bold')

                plt.tight_layout(rect=[0, 0, 1, 0.94])
                filename = f"cv_detail_{os.path.basename(dataset_name).replace('.csv','')}_{model_name}.png"
                path = os.path.join(save_dir, filename)
                plt.savefig(path, bbox_inches="tight")
                plt.close()
                paths.append(path)

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
    Audit sampling ratios from Zaniar's statistical checklist.

    n  = n1 + n2  (class sizes used for landmark generation, typically after balancing)
    t  = points per landmark snapshot
    l  = number of landmark files (snapshots) per class
    """
    n = n1 + n2
    ratios = {
        "n": n,
        "n1": n1,
        "n2": n2,
        "t": t,
        "l": l,
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
    """Consumer: keep H0 columns from experiment-1 matrices. No Ripser."""
    protocol = get_tda_protocol(protocol_bucket)
    cfg = get_dataset_config(dataset_key)
    percentages = dataset_landmark_percentages(dataset_key)
    folder = cfg.folder_name
    exp1 = "1_PH_Default_Parameters"
    exp3 = "3_H0_Only"
    dest_root = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp3, folder)
    if protocol["split_timing"] == "early":
        pairs = {}
        for p in percentages:
            token = _percent_token(p)
            src_train = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp1, folder, "train", f"data_L{token}.csv")
            src_test = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp1, folder, "test", f"data_L{token}.csv")
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
            src = tda_artefact_dir("TDA_Datasets", protocol_bucket, exp1, folder, f"data_L{token}.csv")
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
        path=str(tda_results_dir(protocol_bucket, exp3, folder)),
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

