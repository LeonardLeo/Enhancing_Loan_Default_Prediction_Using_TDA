# -*- coding: utf-8 -*-
"""
Experiment 22 — Visualizing Persistence Diagrams
Dataset: South German Credit

Results: 6_Results/Archives/22_Visualizing_Persistence_Diagrams/South_German_Credit/
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.chdir(Path(__file__).resolve().parent)

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils import data_preprocessing_pipeline

warnings.filterwarnings("ignore")

TARGET = "target"
FOLDER = "South_German_Credit"
DATA_PATH = "../../../../1_Data/Processed_Datasets/South_German_Credit/processed_data.csv"
PCA_COMPONENTS = 10
PERCENTAGES = [10, 20]
RANDOM_STATE = 42
DATASET_KEY = "south_german_credit"
DATASET_TO_USE = "south_german_credit"

import matplotlib.pyplot as plt
from persim import plot_diagrams
from ripser import ripser

# =============================================================================
# Load and preprocess data
# =============================================================================
dataset = pd.read_csv(os.path.abspath(DATA_PATH))
dataset = data_preprocessing_pipeline(
    dataset,
    log_col=["hoehe", "laufzeit"],
)
X = dataset.drop(columns=[TARGET])
y = dataset[TARGET]

# =============================================================================
# Normalize features
# =============================================================================
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# =============================================================================
# Apply PCA
# =============================================================================
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_reduced = pd.DataFrame(
    pca.fit_transform(X_normalized),
    columns=[f"PCA_{num}" for num in range(1, PCA_COMPONENTS + 1)],
)
variance_ratio = pca.explained_variance_ratio_.sum()
print(f"Variance retained with PCA components: {variance_ratio:.2%}")

reduced_data = X_reduced.copy()
reduced_data["Class"] = y

default_data = reduced_data[reduced_data["Class"] == 1].reset_index(drop=True)
non_default_data = reduced_data[reduced_data["Class"] == 0].reset_index(drop=True)
n_samples = len(default_data)
balanced_non_default = non_default_data.sample(n=n_samples, random_state=RANDOM_STATE)


# =============================================================================
# Default class persistence diagram
# =============================================================================
data_default = default_data.drop("Class", axis=1)
diagrams_default = ripser(data_default)["dgms"]
fig, ax = plt.subplots(figsize=(8, 6))
plot_diagrams(diagrams_default, ax=ax, title="Styled Persistence Diagram for Default Data", legend=True)
ax.set_title(f"Styled Persistence Diagram - Default Data ({DATASET_KEY})", fontsize=16)
ax.set_xlabel("Birth", fontsize=14)
ax.set_ylabel("Death", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.5)
ax.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
plt.tight_layout()
default_fig_path = os.path.abspath("persistence_diagram_default.png")
plt.savefig(default_fig_path, dpi=300)
print(f"Saved default diagram to {default_fig_path}")
plt.close(fig)

# =============================================================================
# Non-default class persistence diagram
# =============================================================================
data_non_default = balanced_non_default.drop("Class", axis=1)
diagrams_non_default = ripser(data_non_default)["dgms"]
fig, ax = plt.subplots(figsize=(8, 6))
plot_diagrams(
    diagrams_non_default,
    ax=ax,
    title="Styled Persistence Diagram for Non-Default Data",
    legend=True,
)
ax.set_title(f"Styled Persistence Diagram - Non-Default Data ({DATASET_KEY})", fontsize=16)
ax.set_xlabel("Birth", fontsize=14)
ax.set_ylabel("Death", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.5)
ax.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
plt.tight_layout()
non_default_fig_path = os.path.abspath("persistence_diagram_non_default.png")
plt.savefig(non_default_fig_path, dpi=300)
print(f"Saved non-default diagram to {non_default_fig_path}")
plt.close(fig)
