# -*- coding: utf-8 -*-
"""
Experiment 22 — Persistence diagrams for Statlog German Credit.

Loads the Statlog processed table (not DCCCD), reduces with PCA(15),
and plots one persistence diagram per class.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import plot_diagrams
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# =============================================================================
# Load Statlog processed data
# =============================================================================
data = pd.read_excel(
    r"../../../../1_Data/Processed_Datasets/Statlog_German_Credit_Data/processed_data.xlsx"
)

X = data.drop(columns=["Class"] + [c for c in data.columns if str(c).startswith("Unnamed")])
y = data["Class"]

scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

pca = PCA(n_components=15)
X_reduced = pd.DataFrame(
    pca.fit_transform(X_normalized),
    columns=[f"PCA_{num}" for num in range(1, 16)],
)
variance_ratio = pca.explained_variance_ratio_.sum()
print(f"Variance retained with PCA components: {variance_ratio:.2%}")

reduced_data = X_reduced.copy()
reduced_data["Class"] = y

default_data = reduced_data[reduced_data["Class"] == 1].reset_index(drop=True)
non_default_data = reduced_data[reduced_data["Class"] == 0].reset_index(drop=True)

n_samples = len(default_data)
balanced_non_default = non_default_data.sample(n=n_samples, random_state=42)

# =============================================================================
# Default class
# =============================================================================
data_default = default_data.drop("Class", axis=1)
diagrams_default = ripser(data_default)["dgms"]

fig, ax = plt.subplots(figsize=(8, 6))
plot_diagrams(diagrams_default, ax=ax, title="Styled Persistence Diagram for Default Data", legend=True)
ax.set_title("Styled Persistence Diagram - Default Data (Statlog)", fontsize=16)
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
# Non-default class
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
ax.set_title("Styled Persistence Diagram - Non-Default Data (Statlog)", fontsize=16)
ax.set_xlabel("Birth", fontsize=14)
ax.set_ylabel("Death", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.5)
ax.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
plt.tight_layout()
non_default_fig_path = os.path.abspath("persistence_diagram_non_default.png")
plt.savefig(non_default_fig_path, dpi=300)
print(f"Saved non-default diagram to {non_default_fig_path}")
plt.close(fig)
